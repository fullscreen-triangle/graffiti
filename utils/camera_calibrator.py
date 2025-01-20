import cv2
import numpy as np


class CameraCalibrator:
    def __init__(self):
        self.calibration_points = []
        self.feature_detector = cv2.SIFT_create()
        self.feature_matcher = cv2.BFMatcher()
        self.previous_frame = None
        self.previous_keypoints = None
        self.previous_descriptors = None

    def detect_calibration_points(self, frame):
        """Automatically detect track corners for calibration"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(gray, 25, 0.01, 10)
        corners = np.int0(corners)

        # Filter corners to find track corners
        track_corners = self.filter_track_corners(corners)
        return track_corners

    def filter_track_corners(self, corners):
        """
        Filter detected corners to identify likely track corners

        Args:
            corners: Array of detected corner points

        Returns:
            Array of filtered corner points that are likely track corners
        """
        if len(corners) < 4:
            return corners

        # Convert corners to list of points
        points = [corner.ravel() for corner in corners]

        # Sort points by x and y coordinates
        points_x = sorted(points, key=lambda p: p[0])
        points_y = sorted(points, key=lambda p: p[1])

        # Find corners that form rectangular pattern
        track_corners = []
        min_dist = 20  # Minimum distance between corners

        for i in range(len(points)):
            point = points[i]

            # Check if point forms rectangular pattern with other points
            matches = []
            for j in range(len(points)):
                if i != j:
                    dist = np.sqrt((point[0] - points[j][0]) ** 2 +
                                   (point[1] - points[j][1]) ** 2)
                    if dist > min_dist:
                        matches.append(points[j])

            if len(matches) >= 3:  # Point has at least 3 distant neighbors
                track_corners.append(point)

        return np.array(track_corners)

    def track_calibration_points(self, frame):
        """Track calibration points across frames"""
        if self.previous_frame is None:
            self.previous_frame = frame
            self.previous_keypoints, self.previous_descriptors = self.feature_detector.detectAndCompute(frame, None)
            return None

        # Detect features in current frame
        current_keypoints, current_descriptors = self.feature_detector.detectAndCompute(frame, None)

        # Match features
        matches = self.feature_matcher.knnMatch(self.previous_descriptors, current_descriptors, k=2)

        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        # Update calibration points
        if len(good_matches) > 10:
            src_pts = np.float32([self.previous_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([current_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # Calculate transformation matrix
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            # Update calibration points
            if M is not None:
                self.update_calibration_points(M)

        # Update previous frame data
        self.previous_frame = frame
        self.previous_keypoints = current_keypoints
        self.previous_descriptors = current_descriptors

    def update_calibration_points(self, homography_matrix):
        """
        Update calibration points using homography transformation

        Args:
            homography_matrix: 3x3 homography transformation matrix
        """
        if not self.calibration_points:
            return

        # Convert points to homogeneous coordinates
        points = np.float32(self.calibration_points).reshape(-1, 1, 2)

        # Apply homography transformation
        transformed_points = cv2.perspectiveTransform(points, homography_matrix)

        # Update calibration points
        self.calibration_points = transformed_points.reshape(-1, 2).tolist()

    def get_camera_parameters(self):
        """
        Calculate camera parameters from calibration points

        Returns:
            Dictionary containing camera matrix and distortion coefficients
        """
        if len(self.calibration_points) < 4:
            return None

        # Convert calibration points to appropriate format
        object_points = []  # 3D points in real world space
        image_points = []  # 2D points in image plane

        # Assume calibration points lie on a plane at z=0
        for point in self.calibration_points:
            object_points.append([point[0], point[1], 0])
            image_points.append(point)

        object_points = np.array(object_points, dtype=np.float32)
        image_points = np.array(image_points, dtype=np.float32)

        # Get image size from previous frame
        if self.previous_frame is None:
            return None

        img_size = self.previous_frame.shape[:2]

        # Calculate camera calibration parameters
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            [object_points], [image_points], img_size, None, None
        )

        return {
            'camera_matrix': camera_matrix,
            'dist_coeffs': dist_coeffs,
            'rvecs': rvecs,
            'tvecs': tvecs
        }
