import concurrent
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import logging


class MannequinConversionPipeline:
    def __init__(self, config: Optional[Dict] = None):
        self.config = self._load_default_config()
        if config:
            self.config.update(config)

        self.logger = self._setup_logger()
        self._init_joint_mapping()

    def _load_default_config(self) -> Dict:
        return {
            'max_workers': mp.cpu_count(),
            'chunk_size': 1000,
            'use_threading': False,  # False for CPU-intensive tasks
            'output_format': 'json',
            'compression': False,
            'batch_processing': True,
            'validation': True,
            'error_handling': 'skip',  # 'skip' or 'raise'
            'output_directory': 'mannequin_output'
        }

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger('MannequinConversionPipeline')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def _init_joint_mapping(self):
        self.joint_mapping = {
            'hip_l': {'mannequin_name': 'l_leg', 'axis_order': 'xyz'},
            'hip_r': {'mannequin_name': 'r_leg', 'axis_order': 'xyz'},
            'knee_l': {'mannequin_name': 'l_knee', 'axis_order': 'zxy'},
            'knee_r': {'mannequin_name': 'r_knee', 'axis_order': 'zxy'},
            'ankle_l': {'mannequin_name': 'l_ankle', 'axis_order': 'xyz'},
            'ankle_r': {'mannequin_name': 'r_ankle', 'axis_order': 'xyz'},
            'shoulder_l': {'mannequin_name': 'l_arm', 'axis_order': 'xyz'},
            'shoulder_r': {'mannequin_name': 'r_arm', 'axis_order': 'xyz'},
            'elbow_l': {'mannequin_name': 'l_elbow', 'axis_order': 'zxy'},
            'elbow_r': {'mannequin_name': 'r_elbow', 'axis_order': 'zxy'},
            'wrist_l': {'mannequin_name': 'l_wrist', 'axis_order': 'xyz'},
            'wrist_r': {'mannequin_name': 'r_wrist', 'axis_order': 'xyz'}
        }

    def process_batch(self,
                      input_data: List[Dict],
                      output_path: Optional[str] = None) -> Dict:
        """
        Process a batch of frames in parallel
        """
        self.logger.info(f"Starting batch processing of {len(input_data)} frames")
        start_time = time.time()

        # Split data into chunks for parallel processing
        chunks = self._split_into_chunks(input_data, self.config['chunk_size'])

        # Choose between ProcessPoolExecutor and ThreadPoolExecutor
        executor_class = ThreadPoolExecutor if self.config['use_threading'] else ProcessPoolExecutor

        converted_frames = []
        with executor_class(max_workers=self.config['max_workers']) as executor:
            future_to_chunk = {
                executor.submit(self._process_chunk, chunk): i
                for i, chunk in enumerate(chunks)
            }

            for future in concurrent.futures.as_completed(future_to_chunk):
                chunk_id = future_to_chunk[future]
                try:
                    chunk_results = future.result()
                    converted_frames.extend(chunk_results)
                except Exception as e:
                    self.logger.error(f"Error processing chunk {chunk_id}: {str(e)}")
                    if self.config['error_handling'] == 'raise':
                        raise e

        # Sort frames by frame number
        converted_frames.sort(key=lambda x: x['frame_number'])

        result = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'version': '1.0',
                'frame_count': len(converted_frames),
                'processing_time': time.time() - start_time,
                'config': self.config
            },
            'frames': converted_frames
        }

        if output_path:
            self._save_results(result, output_path)

        self.logger.info(f"Batch processing completed in {time.time() - start_time:.2f} seconds")
        return result

    def _process_chunk(self, chunk: List[Dict]) -> List[Dict]:
        """Process a chunk of frames"""
        converted_frames = []
        for frame_data in chunk:
            try:
                converted_frame = self._convert_frame(frame_data)
                if converted_frame:
                    converted_frames.append(converted_frame)
            except Exception as e:
                self.logger.error(f"Error converting frame {frame_data.get('frame_number')}: {str(e)}")
                if self.config['error_handling'] == 'raise':
                    raise e
        return converted_frames

    def _convert_frame(self, frame_data: Dict) -> Dict:
        """Convert a single frame to mannequin.js format"""
        frame = {
            'frame_number': frame_data['frame_number'],
            'timestamp': frame_data['timestamp'],
            'athletes': []
        }

        for athlete in frame_data.get('athletes', []):
            if athlete.get('skeleton'):
                mannequin_data = self._create_mannequin_data(athlete['skeleton'])
                athlete_data = {
                    'track_id': athlete['track_id'],
                    'mannequin': mannequin_data
                }
                frame['athletes'].append(athlete_data)

        return frame

    def _create_mannequin_data(self, skeleton: Dict) -> Dict:
        """Create mannequin.js formatted joint data"""
        mannequin_data = []

        # Base position and rotation
        mannequin_data.append(skeleton.get('position', [0, 0, 0]))
        mannequin_data.append(skeleton.get('rotation', [0, -90, 0]))

        # Process joint angles
        angles = skeleton.get('angles', {})
        for joint_name, mapping in self.joint_mapping.items():
            angle_data = self._process_joint_angles(
                angles.get(joint_name, {}),
                mapping['axis_order']
            )
            mannequin_data.append(angle_data)

        return {
            "version": 7,
            "data": mannequin_data
        }

    def _process_joint_angles(self,
                              angle_data: Union[Dict, float],
                              axis_order: str) -> List[float]:
        """Process joint angles based on axis order"""
        if isinstance(angle_data, (float, int)):
            return [float(angle_data), 0, 0]

        angles = [0.0, 0.0, 0.0]
        if isinstance(angle_data, dict):
            for i, axis in enumerate(axis_order):
                angles[i] = float(angle_data.get(axis, 0.0))
        return angles

    def _split_into_chunks(self, data: List, chunk_size: int) -> List[List]:
        """Split data into chunks for parallel processing"""
        return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

    def _save_results(self, results: Dict, output_path: str) -> None:
        """Save results to file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2 if not self.config['compression'] else None)


# Example usage
if __name__ == "__main__":
    config = {
        'max_workers': 8,
        'chunk_size': 500,
        'use_threading': False,
        'output_format': 'json',
        'compression': False,
        'validation': True,
        'error_handling': 'skip',
        'output_directory': 'output/mannequin'
    }

    pipeline = MannequinConversionPipeline(config)

    # Example processing
    input_data = [...]  # Your analysis results here
    output_path = "output/mannequin/converted_results.json"

    results = pipeline.process_batch(input_data, output_path)
