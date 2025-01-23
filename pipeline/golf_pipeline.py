from pathlib import Path
import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
import mediapipe as mp
from dataclasses import dataclass
import logging

from golf.ClubPathDrawer import ClubPathDrawer
from golf.GolfSwing import GolfSwinger
from golf.PoseDrawer import PoseDrawer
from .biomechanics_analysis import BiomechanicsAnalysisPipeline


@dataclass
class SwingPhase:
    frame_idx: int
    phase_name: str
    pose_data: Dict
    timestamp: float


@dataclass
class GolfSwingResult:
    setup_metrics: Dict
    swing_phases: List[SwingPhase]
    swing_metrics: Dict
    club_metrics: Dict
    ball_metrics: Optional[Dict]
    video_path: str
    duration: float


class GolfSwingPipeline(BiomechanicsAnalysisPipeline):
    """Pipeline for golf swing analysis that builds upon biomechanical analysis"""
    def __init__(self, config: dict):
        super().__init__(config)
        
        # Golf-specific components
        self.club_tracker = cv2.TrackerCSRT_create()
        self.pose_drawer = PoseDrawer()
        self.club_drawer = ClubPathDrawer()
        # ... initialize other golf components ...

    def _process_frame(self, frame: np.ndarray, frame_idx: int, prev_frame: Optional[np.ndarray]) -> Dict:
        """Override frame processing to add golf-specific analysis"""
        # Get base biomechanical analysis results
        base_result = super()._process_frame(frame, frame_idx, prev_frame)
        
        # Add golf-specific analysis
        if frame_idx == 0:
            club_bbox = self._initialize_club_tracking(frame)
            self.club_tracker.init(frame, club_bbox)
        
        success, bbox = self.club_tracker.update(frame)
        club_position = self._get_bbox_center(bbox) if success else None

        return {
            **base_result,
            'club_tracking': {
                'position': club_position,
                'success': success
            }
        }

    def analyze_video(self, video_path: str) -> Dict:
        """Override video analysis to include golf-specific results"""
        # Get base biomechanical analysis results
        base_results = super().analyze_video(video_path)
        
        # Add golf-specific analysis
        swing_phases = self._detect_swing_phases(base_results['frame_results'])
        setup_metrics = self._analyze_setup(base_results['frame_results'][0])
        
        return {
            **base_results,
            'golf_analysis': {
                'swing_phases': swing_phases,
                'setup_metrics': setup_metrics,
                'club_metrics': self._analyze_club_metrics(base_results['frame_results']),
                'ball_metrics': self._analyze_ball_flight(base_results['frame_results'], swing_phases)
            }
        }

    def analyze_swing(self, video_path: str) -> GolfSwingResult:
        """Main method to analyze a golf swing video."""
        self.logger.info(f"Starting golf swing analysis for {video_path}")

        # Load video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        frames = []
        pose_sequence = []
        club_positions = []

        # Process each frame
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frames.append(frame)

            # Detect pose
            pose_results = self._detect_pose(frame)
            pose_sequence.append(pose_results)

            # Track club head
            if len(club_positions) == 0:
                # Initialize club head tracking
                club_bbox = self._initialize_club_tracking(frame)
                self.club_tracker.init(frame, club_bbox)
            else:
                success, bbox = self.club_tracker.update(frame)
                if success:
                    club_positions.append(self._get_bbox_center(bbox))

        cap.release()

        # Detect swing phases
        swing_phases = self._detect_swing_phases(pose_sequence)

        # Calculate metrics
        setup_metrics = self._analyze_setup(pose_sequence[0])
        swing_metrics = self._analyze_swing_mechanics(pose_sequence, swing_phases)
        club_metrics = self._analyze_club_metrics(club_positions)
        ball_metrics = self._analyze_ball_flight(frames, swing_phases)

        duration = len(frames) / cap.get(cv2.CAP_PROP_FPS)

        return GolfSwingResult(
            setup_metrics=setup_metrics,
            swing_phases=swing_phases,
            swing_metrics=swing_metrics,
            club_metrics=club_metrics,
            ball_metrics=ball_metrics,
            video_path=video_path,
            duration=duration
        )

    def _detect_pose(self, frame: np.ndarray) -> Dict:
        """Detect pose in frame using MediaPipe."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        if results.pose_landmarks:
            return {
                'landmarks': results.pose_landmarks,
                'world_landmarks': results.pose_world_landmarks
            }
        return None

    def _initialize_club_tracking(self, frame: np.ndarray) -> Tuple[int, int, int, int]:
        """Initialize club head tracking."""
        # This is a simplified version - you might want to implement more sophisticated
        # club head detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(
            gray,
            self.config['club_tracking']['max_points'],
            self.config['club_tracking']['quality_level'],
            self.config['club_tracking']['min_distance']
        )

        if corners is not None:
            x_min = min(corners[:, 0, 0])
            x_max = max(corners[:, 0, 0])
            y_min = min(corners[:, 0, 1])
            y_max = max(corners[:, 0, 1])
            return (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))

        raise ValueError("Could not initialize club tracking")

    def _detect_swing_phases(self, pose_sequence: List[Dict]) -> List[SwingPhase]:
        """Detect key phases in the golf swing."""
        phases = []

        # Implement phase detection logic here
        # This should detect: setup, takeaway, top of backswing, impact, and follow-through
        # You'll need to analyze pose angles and positions to determine these

        return phases

    def _analyze_setup(self, pose_data: Dict) -> Dict:
        """Analyze the setup position."""
        # Implement setup analysis
        # Check spine angle, knee flex, posture, etc.
        return {}

    def _analyze_swing_mechanics(self, pose_sequence: List[Dict],
                                 phases: List[SwingPhase]) -> Dict:
        """Analyze swing mechanics throughout the swing."""
        # Implement swing mechanics analysis
        # Calculate rotation angles, positions, timing, etc.
        return {}

    def _analyze_club_metrics(self, club_positions: List[Tuple[int, int]]) -> Dict:
        """Analyze club metrics."""
        # Implement club metrics analysis
        # Calculate club head speed, path, face angle, etc.
        return {}

    def _analyze_ball_flight(self, frames: List[np.ndarray],
                             phases: List[SwingPhase]) -> Optional[Dict]:
        """Analyze ball flight."""
        # Implement ball flight analysis
        # Track ball after impact, calculate launch conditions
        return None

    @staticmethod
    def _get_bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        """Get center point of bounding box."""
        x, y, w, h = bbox
        return (int(x + w / 2), int(y + h / 2))

    def generate_analysis_video(self, frames: List[np.ndarray],
                                result: GolfSwingResult) -> str:
        """Generate video with analysis overlays."""
        output_path = Path(result.video_path).with_name(
            f"{Path(result.video_path).stem}_analysis.mp4"
        )

        # Set up video writer
        height, width = frames[0].shape[:2]
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*'mp4v'),
            self.config['visualization']['output_fps'],
            (width, height)
        )

        for i, frame in enumerate(frames):
            # Add visualization overlays
            if self.config['visualization']['draw_pose']:
                frame = self.pose_drawer.draw_pose(frame, result.swing_metrics)

            if self.config['visualization']['draw_club_path']:
                frame = self.club_drawer.draw_club_path(frame, result.club_metrics)

            writer.write(frame)

        writer.release()
        return str(output_path)

    def _calculate_stance_width(self, frame_data: Dict) -> float:
        """Calculate the width of the golfer's stance using ankle keypoints."""
        pose_data = frame_data.get('pose_data', {})
        if not pose_data or 'landmarks' not in pose_data:
            return 0.0

        # Get ankle keypoints (assuming MediaPipe pose landmarks)
        left_ankle = pose_data['landmarks'].landmark[27]  # Left ankle index
        right_ankle = pose_data['landmarks'].landmark[28]  # Right ankle index
        
        # Calculate horizontal distance between ankles in pixels
        stance_width = abs(left_ankle.x - right_ankle.x)
        
        # Convert to real-world units (assuming average height for scaling)
        height_pixels = abs(pose_data['landmarks'].landmark[0].y - pose_data['landmarks'].landmark[27].y)
        pixels_to_meters = 1.7 / height_pixels  # Assuming average height of 1.7m
        
        return stance_width * pixels_to_meters

    def _calculate_spine_angle(self, frame_data: Dict) -> float:
        """Calculate the spine angle relative to vertical."""
        pose_data = frame_data.get('pose_data', {})
        if not pose_data or 'landmarks' not in pose_data:
            return 0.0

        # Get relevant keypoints
        hip_mid = np.mean([
            [pose_data['landmarks'].landmark[23].x, pose_data['landmarks'].landmark[23].y],  # Left hip
            [pose_data['landmarks'].landmark[24].x, pose_data['landmarks'].landmark[24].y]   # Right hip
        ], axis=0)
        
        shoulder_mid = np.mean([
            [pose_data['landmarks'].landmark[11].x, pose_data['landmarks'].landmark[11].y],  # Left shoulder
            [pose_data['landmarks'].landmark[12].x, pose_data['landmarks'].landmark[12].y]   # Right shoulder
        ], axis=0)
        
        # Calculate angle between spine line and vertical
        spine_vector = shoulder_mid - hip_mid
        vertical_vector = np.array([0, -1])
        
        cos_angle = np.dot(spine_vector, vertical_vector) / (np.linalg.norm(spine_vector) * np.linalg.norm(vertical_vector))
        angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        
        return np.degrees(angle_rad)

    def _analyze_swing_plane(self, frame_results: List[Dict]) -> Dict:
        """Analyze the swing plane characteristics using club head positions."""
        if not self.club_positions or len(self.club_positions) < 3:
            return {'flatness': 0.0, 'consistency': 0.0}

        # Convert positions to numpy array
        points = np.array(self.club_positions)
        
        # Fit a plane to the points using SVD
        centroid = np.mean(points, axis=0)
        centered_points = points - centroid
        _, singular_values, vh = np.linalg.svd(centered_points)
        
        # Normal vector of the fitted plane
        normal = vh[2]
        
        # Calculate plane flatness (how well points fit the plane)
        distances = np.abs(np.dot(centered_points, normal))
        flatness_score = 1.0 - (np.mean(distances) / np.linalg.norm(centered_points, axis=1).mean())
        
        # Calculate consistency (variation in plane angle during swing)
        swing_segments = np.array_split(points, min(len(points), 5))
        segment_angles = []
        
        for segment in swing_segments:
            if len(segment) >= 3:
                _, _, vh = np.linalg.svd(segment - np.mean(segment, axis=0))
                segment_normal = vh[2]
                angle = np.arccos(np.abs(np.dot(normal, segment_normal)))
                segment_angles.append(np.degrees(angle))
        
        consistency_score = 1.0 - (np.std(segment_angles) / 90.0 if segment_angles else 0.0)
        
        return {
            'flatness': float(flatness_score),
            'consistency': float(consistency_score),
            'plane_normal': normal.tolist()
        }

    def _calculate_club_speed(self, positions: List[Tuple[int, int]]) -> float:
        """Calculate club head speed using finite differences."""
        if len(positions) < 2:
            return 0.0
        
        # Convert positions to numpy array
        points = np.array(positions)
        
        # Calculate velocities using central differences
        velocities = np.zeros(len(points))
        dt = 1.0 / self.config['video_processing']['fps']
        
        # First point: forward difference
        velocities[0] = np.linalg.norm(points[1] - points[0]) / dt
        
        # Middle points: central difference
        for i in range(1, len(points)-1):
            velocities[i] = np.linalg.norm(points[i+1] - points[i-1]) / (2*dt)
        
        # Last point: backward difference
        velocities[-1] = np.linalg.norm(points[-1] - points[-2]) / dt
        
        # Convert from pixels/s to mph (approximate conversion)
        pixels_to_meters = 0.1  # This should be calibrated based on known distances
        speed_mph = np.max(velocities) * pixels_to_meters * 2.23694  # Convert m/s to mph
        
        return float(speed_mph)

    def _analyze_kinematic_sequence(self, frame_results: List[Dict]) -> Dict:
        """Analyze the kinematic sequence of the swing."""
        if not frame_results:
            return {}

        # Extract relevant joint angles over time
        sequence_data = []
        for frame in frame_results:
            pose_data = frame.get('pose_data', {})
            if not pose_data or 'landmarks' not in pose_data:
                continue

            # Calculate angular velocities for key joints
            hips = self._calculate_hip_rotation(pose_data)
            torso = self._calculate_torso_rotation(pose_data)
            arms = self._calculate_arm_rotation(pose_data)
            
            sequence_data.append({
                'hips': hips,
                'torso': torso,
                'arms': arms
            })

        if not sequence_data:
            return {}

        # Analyze the sequence timing
        peak_times = {
            'hips': self._find_peak_time(sequence_data, 'hips'),
            'torso': self._find_peak_time(sequence_data, 'torso'),
            'arms': self._find_peak_time(sequence_data, 'arms')
        }

        # Check if sequence is correct (hips → torso → arms)
        correct_sequence = (peak_times['hips'] <= peak_times['torso'] <= peak_times['arms'])
        
        # Calculate timing gaps
        timing_gaps = {
            'hips_to_torso': peak_times['torso'] - peak_times['hips'],
            'torso_to_arms': peak_times['arms'] - peak_times['torso']
        }

        return {
            'sequence_correct': correct_sequence,
            'peak_times': peak_times,
            'timing_gaps': timing_gaps,
            'efficiency_score': self._calculate_sequence_efficiency(timing_gaps)
        }

    def _calculate_launch_angle(self, frame_results: List[Dict]) -> float:
        """Calculate the ball launch angle using initial trajectory."""
        if not self.ball_tracker or len(frame_results) < 3:
            return 0.0

        # Get first few frames of ball trajectory
        ball_positions = []
        for frame in frame_results:
            if 'ball_tracking' in frame and frame['ball_tracking']['position']:
                ball_positions.append(frame['ball_tracking']['position'])
                if len(ball_positions) >= 3:
                    break

        if len(ball_positions) < 3:
            return 0.0

        # Fit a quadratic curve to the initial trajectory
        positions = np.array(ball_positions)
        t = np.arange(len(positions))
        
        # Separate x and y coordinates
        x = positions[:, 0]
        y = positions[:, 1]
        
        # Fit quadratic to y coordinates
        coeffs = np.polyfit(t, y, 2)
        
        # Calculate initial velocity vector
        dx = (x[1] - x[0])
        dy = -coeffs[1]  # Negative because y increases downward in image
        
        # Calculate angle
        angle_rad = np.arctan2(dy, dx)
        return float(np.degrees(angle_rad))

    def _evaluate_posture(self, frame_data: Dict) -> float:
        """Evaluate the overall posture score based on multiple factors."""
        if not frame_data or 'pose_data' not in frame_data:
            return 0.0

        pose_data = frame_data['pose_data']
        
        # Calculate individual posture components
        spine_angle = self._calculate_spine_angle(frame_data)
        knee_flex = self._calculate_knee_flex(pose_data)
        weight_distribution = self._calculate_weight_distribution(pose_data)
        head_position = self._evaluate_head_position(pose_data)
        
        # Weight factors for different components
        weights = {
            'spine_angle': 0.35,
            'knee_flex': 0.25,
            'weight_distribution': 0.25,
            'head_position': 0.15
        }
        
        # Score each component (0-1 scale)
        scores = {
            'spine_angle': self._score_spine_angle(spine_angle),
            'knee_flex': self._score_knee_flex(knee_flex),
            'weight_distribution': self._score_weight_distribution(weight_distribution),
            'head_position': head_position
        }
        
        # Calculate weighted average
        total_score = sum(weights[k] * scores[k] for k in weights.keys())
        
        return float(total_score)

    def _calculate_knee_flex(self, pose_data: Dict) -> float:
        """Calculate knee flexion angle."""
        if not pose_data or 'landmarks' not in pose_data:
            return 0.0

        # Get knee angles for both legs
        left_knee = self._calculate_joint_angle(
            pose_data['landmarks'].landmark[23],  # Hip
            pose_data['landmarks'].landmark[25],  # Knee
            pose_data['landmarks'].landmark[27]   # Ankle
        )
        
        right_knee = self._calculate_joint_angle(
            pose_data['landmarks'].landmark[24],  # Hip
            pose_data['landmarks'].landmark[26],  # Knee
            pose_data['landmarks'].landmark[28]   # Ankle
        )
        
        return (left_knee + right_knee) / 2

    def _calculate_joint_angle(self, p1, p2, p3) -> float:
        """Calculate angle between three points."""
        v1 = np.array([p1.x - p2.x, p1.y - p2.y])
        v2 = np.array([p3.x - p2.x, p3.y - p2.y])
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        
        return np.degrees(angle)

    def _score_spine_angle(self, angle: float) -> float:
        """Score spine angle (ideal range: 35-45 degrees)."""
        if 35 <= angle <= 45:
            return 1.0
        elif angle < 35:
            return 1.0 - (35 - angle) / 35
        else:
            return 1.0 - (angle - 45) / 45

    def _score_knee_flex(self, angle: float) -> float:
        """Score knee flexion (ideal range: 20-30 degrees)."""
        if 20 <= angle <= 30:
            return 1.0
        elif angle < 20:
            return 1.0 - (20 - angle) / 20
        else:
            return 1.0 - (angle - 30) / 30

    def _score_weight_distribution(self, distribution: float) -> float:
        """Score weight distribution (ideal: close to 50/50)."""
        return 1.0 - abs(0.5 - distribution)


def main():
    # Set up basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    try:
        # Set your paths here
        config_path = 'golf_config.yaml'
        video_path = 'golf_swing.mp4'  # or your video path

        # Initialize the golf swing analyzer
        logger.info("Initializing Golf Swing Analyzer...")
        swinger = GolfSwinger(config_path)

        # Analyze the swing
        results = swinger.analyze_swing(video_path)

        # Print results
        print(f"\nAnalysis Results:")
        print(f"Swing duration: {results['duration']:.2f} seconds")
        print("Swing phases:", results['swing_phases'])
        print("Club head speed:", results['metrics']['club']['speed'])

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()
