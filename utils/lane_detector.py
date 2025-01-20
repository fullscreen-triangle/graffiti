# lane_detector.py
import numpy as np
import cv2
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List


@dataclass
class LaneInfo:
    left_lane: Optional[np.ndarray]
    right_lane: Optional[np.ndarray]
    lane_width: Optional[float]
    curvature: Optional[float]


class LaneDetector:
    def __init__(self, config: dict):
        self.config = config

        # Default parameters if not specified in config
        self.roi_vertices = config.get('roi_vertices', np.array([
            [0, 720],
            [450, 450],
            [830, 450],
            [1280, 720]
        ]))

        self.pixel_to_meter_x = config.get('pixel_to_meter_x', 3.7 / 800)  # meters per pixel in x dimension
        self.pixel_to_meter_y = config.get('pixel_to_meter_y', 30.0 / 720)  # meters per pixel in y dimension

        # Sliding window parameters
        self.nwindows = config.get('nwindows', 9)
        self.margin = config.get('margin', 100)
        self.minpix = config.get('minpix', 50)

    def detect(self, frame: np.ndarray) -> Optional[LaneInfo]:
        """Main lane detection pipeline."""
        try:
            # Preprocess the image
            binary = self._preprocess_frame(frame)

            # Apply perspective transform
            warped = self._perspective_transform(binary)

            # Find lane line pixels
            left_x, left_y, right_x, right_y = self._find_lane_pixels(warped)

            if not (len(left_x) and len(right_x)):
                return None

            # Fit polynomial curves
            left_fit = np.polyfit(left_y, left_x, 2)
            right_fit = np.polyfit(right_y, right_x, 2)

            # Calculate lane width and curvature
            lane_width = self._calculate_lane_width(left_fit, right_fit)
            curvature = self._calculate_curvature(left_fit, right_fit)

            return LaneInfo(
                left_lane=left_fit,
                right_lane=right_fit,
                lane_width=lane_width,
                curvature=curvature
            )

        except Exception as e:
            print(f"Lane detection failed: {str(e)}")
            return None

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply preprocessing steps to prepare image for lane detection."""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply Sobel operator
        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        abs_sobelx = np.absolute(sobelx)
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

        # Apply threshold
        binary = np.zeros_like(scaled_sobel)
        binary[(scaled_sobel >= 20) & (scaled_sobel <= 100)] = 1

        # Apply ROI mask
        mask = np.zeros_like(binary)
        cv2.fillPoly(mask, [self.roi_vertices], 1)
        masked = cv2.bitwise_and(binary, mask)

        return masked

    def _perspective_transform(self, binary_img: np.ndarray) -> np.ndarray:
        """Apply perspective transform to get bird's eye view."""
        img_size = (binary_img.shape[1], binary_img.shape[0])

        src = np.float32([
            [585, 460],
            [695, 460],
            [1127, 720],
            [203, 720]
        ])

        dst = np.float32([
            [320, 0],
            [960, 0],
            [960, 720],
            [320, 720]
        ])

        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(binary_img, M, img_size)

        return warped

    def _find_lane_pixels(self, binary_warped: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Find pixels belonging to left and right lanes using sliding window."""
        histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)

        midpoint = len(histogram) // 2
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        window_height = np.int_(binary_warped.shape[0] // self.nwindows)

        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        leftx_current = leftx_base
        rightx_current = rightx_base

        left_lane_inds = []
        right_lane_inds = []

        for window in range(self.nwindows):
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height

            win_xleft_low = leftx_current - self.margin
            win_xleft_high = leftx_current + self.margin
            win_xright_low = rightx_current - self.margin
            win_xright_high = rightx_current + self.margin

            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            if len(good_left_inds) > self.minpix:
                leftx_current = np.int_(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > self.minpix:
                rightx_current = np.int_(np.mean(nonzerox[good_right_inds]))

        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        return leftx, lefty, rightx, righty

    def _calculate_lane_width(self, left_fit: np.ndarray, right_fit: np.ndarray) -> float:
        """Calculate the lane width in meters."""
        y_eval = 720  # Height of the image
        left_x = left_fit[0] * y_eval ** 2 + left_fit[1] * y_eval + left_fit[2]
        right_x = right_fit[0] * y_eval ** 2 + right_fit[1] * y_eval + right_fit[2]

        # Convert pixel width to meters
        lane_width = (right_x - left_x) * self.pixel_to_meter_x

        return lane_width

    def _calculate_curvature(self, left_fit: np.ndarray, right_fit: np.ndarray) -> float:
        """Calculate the radius of curvature in meters."""
        y_eval = 720  # Height of the image

        # Convert to world space
        left_fit_cr = left_fit * [self.pixel_to_meter_x / (self.pixel_to_meter_y ** 2),
                                  self.pixel_to_meter_x / self.pixel_to_meter_y,
                                  self.pixel_to_meter_x]
        right_fit_cr = right_fit * [self.pixel_to_meter_x / (self.pixel_to_meter_y ** 2),
                                    self.pixel_to_meter_x / self.pixel_to_meter_y,
                                    self.pixel_to_meter_x]

        # Calculate radius of curvature
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval + left_fit_cr[1]) ** 2) ** 1.5) \
                        / np.absolute(2 * left_fit_cr[0])
        right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval + right_fit_cr[1]) ** 2) ** 1.5) \
                         / np.absolute(2 * right_fit_cr[0])

        # Return average curvature
        return (left_curverad + right_curverad) / 2

