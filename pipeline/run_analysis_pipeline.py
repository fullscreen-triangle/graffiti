import yaml
import logging
from pathlib import Path
import cv2
import numpy as np
from typing import Dict
import json

from pipeline.biomechanics_analysis import BiomechanicsAnalysisPipeline
from pipeline.motion_analysis import MotionAnalysisPipeline
from pipeline.posture_converter import MannequinConversionPipeline
from pipeline.scene_analysis import SceneAnalysisPipeline


class VideoAnalysisPipeline:
    def __init__(self, config_path: Path):
        self.base_dir = config_path.parent.parent
        self.output_dir = self.base_dir / 'output'
        
        # Setup logging
        self._setup_logging()
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Initialize pipelines
        self.scene_pipeline = SceneAnalysisPipeline(self.config)
        self.motion_pipeline = MotionAnalysisPipeline(self.config)
        self.biomech_pipeline = BiomechanicsAnalysisPipeline(self.config)
        self.mannequin_pipeline = MannequinConversionPipeline(self.config.get('mannequin', {}))
        
        self.logger = logging.getLogger(__name__)

    def _setup_logging(self):
        log_dir = self.output_dir / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'pipeline.log'),
                logging.StreamHandler()
            ]
        )

    def _save_results(self, results: Dict, stage: str, video_name: str):
        """Save analysis results to JSON"""
        output_file = self.output_dir / stage / f"{video_name}_{stage}_analysis.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Saved {stage} results to {output_file}")
        return output_file

    def _create_annotated_video(self, original_video: Path, results: Dict) -> Path:
        """Reconstruct video with annotations from all analyses"""
        cap = cv2.VideoCapture(str(original_video))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        output_path = self.output_dir / 'videos' / f"{original_video.stem}_annotated.mp4"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Add scene analysis annotations
            if 'scene' in results:
                scene_data = results['scene'].get(frame_idx, {})
                # Add scene annotations (lanes, stability metrics, etc.)
                
            # Add motion analysis annotations
            if 'motion' in results:
                motion_data = results['motion'].get(frame_idx, {})
                # Add motion annotations (pose keypoints, trajectories, etc.)
                
            # Add biomechanics annotations
            if 'biomechanics' in results:
                biomech_data = results['biomechanics'].get(frame_idx, {})
                # Add biomechanics annotations (joint angles, forces, etc.)
                
            # Write the annotated frame
            out.write(frame)
            frame_idx += 1
            
        cap.release()
        out.release()
        
        self.logger.info(f"Created annotated video: {output_path}")
        return output_path

    def process_video(self, video_path: Path):
        """Run complete analysis pipeline on video"""
        self.logger.info(f"Starting analysis pipeline for {video_path}")
        video_name = video_path.stem
        
        try:
            # 1. Scene Analysis
            self.logger.info("Running scene analysis...")
            scene_results = self.scene_pipeline.analyze_video(str(video_path))
            self._save_results(scene_results, 'scene', video_name)
            
            # 2. Motion Analysis
            self.logger.info("Running motion analysis...")
            motion_results = self.motion_pipeline.analyze_video(str(video_path))
            self._save_results(motion_results, 'motion', video_name)
            
            # 3. Biomechanics Analysis
            self.logger.info("Running biomechanics analysis...")
            biomech_results = self.biomech_pipeline.analyze_video(str(video_path))
            self._save_results(biomech_results, 'biomechanics', video_name)
            
            # 4. Mannequin Conversion
            self.logger.info("Converting to mannequin format...")
            mannequin_file = self.output_dir / 'mannequin' / f"{video_name}_mannequin.json"
            self.mannequin_pipeline.save_for_viewer(
                biomech_results['athletes'], 
                str(mannequin_file)
            )
            
            # 5. Combine all results
            all_results = {
                'scene': scene_results,
                'motion': motion_results,
                'biomechanics': biomech_results
            }
            
            # 6. Create annotated video
            self.logger.info("Creating annotated video...")
            annotated_video = self._create_annotated_video(video_path, all_results)
            
            self.logger.info("Pipeline completed successfully!")
            return annotated_video
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise

def main():
    base_dir = Path(__file__).parent.parent
    config_path = base_dir / 'config' / 'config.yaml'
    video_path = base_dir / 'public' / 'bolt_100m_record_side.mp4'
    
    pipeline = VideoAnalysisPipeline(config_path)
    pipeline.process_video(video_path)

if __name__ == "__main__":
    main() 