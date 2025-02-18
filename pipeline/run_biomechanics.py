import yaml
import logging
import json
from pathlib import Path
import sys

from pipeline.biomechanics_analysis import BiomechanicsAnalysisPipeline

# Add the project root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


def main():
    # Setup paths
    base_dir = Path(__file__).parent.parent  # Go up one level to get to project root
    config_path = base_dir / 'config' / 'config.yaml'
    video_path = base_dir / 'public' / 'last_stretch.mp4'  # Change this to your video file
    output_dir = base_dir / 'output' / 'biomechanics'
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / 'biomechanics.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    try:
        # Initialize and run pipeline
        pipeline = BiomechanicsAnalysisPipeline(config)
        results = pipeline.analyze_video(str(video_path))
        
        # Save results
        output_file = output_dir / f"{video_path.stem}_biomech_analysis.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Analysis completed. Results saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")




if __name__ == "__main__":
    main() 