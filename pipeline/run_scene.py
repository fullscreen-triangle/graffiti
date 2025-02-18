import yaml
import logging
from pathlib import Path
import json
from pipeline.scene_analysis import SceneAnalysisPipeline

def main():
    base_dir = Path(__file__).parent
    config_path = base_dir / 'config.yaml'
    video_path = base_dir / 'videos' / 'sample.mp4'  # Change this to your video file
    output_dir = base_dir / 'output' / 'scene'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / 'scene.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    try:
        pipeline = SceneAnalysisPipeline(config)
        results = pipeline.analyze_video(str(video_path))
        
        output_file = output_dir / f"{video_path.stem}_scene_analysis.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Analysis completed. Results saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")

if __name__ == "__main__":
    main() 