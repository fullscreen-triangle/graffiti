import numpy as np
import torch
from typing import Optional, Dict
import cv2


class PoseDetector:
    def __init__(self, model_path: str, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load the pose detection model
        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()

    def _load_model(self, model_path: str) -> torch.nn.Module:
        try:
            model = torch.load(model_path, map_location=self.device)
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}: {str(e)}")

    def detect(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect human pose keypoints in the input frame.

        Args:
            frame: Input image frame as numpy array (H, W, C)

        Returns:
            Numpy array of shape (N, K, 3) where:
                N is number of detected persons
                K is number of keypoints
                3 represents (x, y, confidence) for each keypoint
        """
        if frame is None:
            return None

        # Preprocess the frame
        input_tensor = self._preprocess_frame(frame)

        # Run inference
        with torch.no_grad():
            predictions = self.model(input_tensor)

        # Post-process predictions
        keypoints = self._postprocess_predictions(predictions)

        # Filter by confidence
        if keypoints is not None:
            mask = keypoints[..., 2] > self.confidence_threshold
            if not mask.any():
                return None
            keypoints = keypoints[mask.any(axis=1)]

        return keypoints

    def _preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess frame for model input"""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Normalize
        frame_normalized = frame_rgb.astype(np.float32) / 255.0

        # Add batch dimension and convert to tensor
        input_tensor = torch.from_numpy(frame_normalized).permute(2, 0, 1).unsqueeze(0)

        return input_tensor.to(self.device)

    def _postprocess_predictions(self, predictions: torch.Tensor) -> Optional[np.ndarray]:
        """Convert model predictions to keypoint coordinates"""
        if predictions is None:
            return None

        # Convert to numpy array
        keypoints = predictions.cpu().numpy()

        # Reshape to (N, K, 3) format
        if len(keypoints.shape) == 4:  # If batch dimension exists
            keypoints = keypoints[0]  # Remove batch dimension

        return keypoints

    def get_model_info(self) -> Dict:
        """Return information about the loaded model"""
        return {
            'device': str(self.device),
            'confidence_threshold': self.confidence_threshold,
            'model_parameters': sum(p.numel() for p in self.model.parameters()),
        }
