import os
import yaml
import logging
from pathlib import Path
from typing import Dict
import json
from pipeline.scene_analysis import SceneAnalysisPipeline
from pipeline.motion_analysis import MotionAnalysisPipeline
from pipeline.biomechanics_analysis import BiomechanicsAnalysisPipeline
from pipeline.posture_converter import MannequinConversionPipeline

def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_logging(output_dir: str) -> logging.Logger:
    """Setup logging configuration"""
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'analysis.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def process_video(video_path: str, config: Dict, output_dir: str) -> Dict:
    """Process a single video through all pipelines"""
    logger = logging.getLogger(__name__)
    
    # Create pipeline instances
    scene_pipeline = SceneAnalysisPipeline(config)
    motion_pipeline = MotionAnalysisPipeline(config)
    biomech_pipeline = BiomechanicsAnalysisPipeline(config)
    mannequin_pipeline = MannequinConversionPipeline(config.get('mannequin', {}))
    
    # Process video through each pipeline
    logger.info(f"Processing video: {video_path}")
    
    try:
        # Scene analysis
        scene_results = scene_pipeline.analyze_video(video_path)
        
        # Motion analysis
        motion_results = motion_pipeline.analyze_video(video_path)
        
        # Biomechanics analysis
        biomech_results = biomech_pipeline.analyze_video(video_path)
        
        # Convert to mannequin format
        mannequin_output = os.path.join(output_dir, 'mannequin', f"{Path(video_path).stem}_mannequin.json")
        mannequin_pipeline.save_for_viewer(biomech_results['athletes'], mannequin_output)
        
        # Combine results
        results = {
            'video_path': video_path,
            'scene_analysis': scene_results,
            'motion_analysis': motion_results,
            'biomechanics_analysis': biomech_results,
            'mannequin_output': mannequin_output
        }
        
        # Save combined results
        output_path = os.path.join(output_dir, f"{Path(video_path).stem}_analysis.json")
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        return results
        
    except Exception as e:
        logger.error(f"Error processing video {video_path}: {str(e)}")
        raise

def main():
    # Setup paths
    base_dir = Path(__file__).parent
    config_path = base_dir / 'config.yaml'
    videos_dir = base_dir / 'videos'
    output_dir = base_dir / 'output'
    
    # Create output directories
    output_dir.mkdir(exist_ok=True)
    (output_dir / 'mannequin').mkdir(exist_ok=True)
    
    # Load configuration
    config = load_config(config_path)
    
    # Setup logging
    logger = setup_logging(str(output_dir))
    
    # Process all videos in the videos directory
    video_extensions = ('.mp4', '.avi', '.mov')
    videos = [f for f in videos_dir.glob('*') if f.suffix.lower() in video_extensions]
    
    if not videos:
        logger.warning(f"No videos found in {videos_dir}")
        return
    
    for video_path in videos:
        try:
            logger.info(f"Starting analysis of {video_path}")
            results = process_video(str(video_path), config, str(output_dir))
            logger.info(f"Successfully processed {video_path}")
        except Exception as e:
            logger.error(f"Failed to process {video_path}: {str(e)}")

if __name__ == "__main__":
    main() 