import yaml
import logging
from pathlib import Path
from pipeline.posture_converter import MannequinConversionPipeline
from pipeline.biomechanics_analysis import BiomechanicsAnalysisPipeline
import json

def main():
    base_dir = Path(__file__).parent.parent  # Go up one level to get to project root
    config_path = base_dir / 'config' / 'config.yaml'
    video_path = base_dir / 'public' / 'bolt_100m_record_side.mp4'
    output_dir = base_dir / 'output' / 'mannequin'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / 'posture.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # First run biomechanics analysis
        logger.info("Running biomechanics analysis...")
        biomech_pipeline = BiomechanicsAnalysisPipeline(config)
        biomech_results = biomech_pipeline.analyze_video(str(video_path))
        
        # Then convert to mannequin format
        logger.info("Converting to mannequin format...")
        mannequin_pipeline = MannequinConversionPipeline(config.get('mannequin', {}))
        
        # Convert and save
        output_file = output_dir / f"{video_path.stem}_mannequin.json"
        mannequin_pipeline.save_for_viewer(biomech_results['athletes'], str(output_file))
            
        logger.info(f"Conversion completed. Results saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Error during conversion: {str(e)}")
        raise  # Add this to see the full error traceback

if __name__ == "__main__":
    main() 