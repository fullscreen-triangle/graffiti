"""
Environmental Query Processing Demo

Demonstration of twelve-dimensional environmental measurement and integration
with atmospheric molecular processing and cross-modal catalysis.
"""

import asyncio
import argparse
import json
import sys
import time
import logging
from typing import Dict, List, Any, Optional

from graffiti.core.types import Query, QueryId, EnvironmentalState, UserContext
from graffiti.environmental.twelve_dimensional import (
    EnvironmentalMeasurement, 
    EnvironmentalIntegrator,
    EnvironmentalDimension
)
from graffiti.environmental.atmospheric_molecular import AtmosphericProcessingNetwork
from graffiti.environmental.cross_modal_bmv import CrossModalBMDValidator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnvironmentalQueryDemo:
    """
    Environmental query processing demonstration.
    
    Demonstrates twelve-dimensional environmental measurement, atmospheric
    molecular processing, and environmental consciousness integration.
    """
    
    def __init__(self, dimensions: int = 12):
        """Initialize environmental query demo."""
        self.dimensions = min(dimensions, 12)  # Cap at 12 dimensions
        self.environmental_measurement = EnvironmentalMeasurement()
        self.environmental_integrator = EnvironmentalIntegrator()
        self.atmospheric_network = AtmosphericProcessingNetwork()
        self.cross_modal_validator = CrossModalBMDValidator()
        
        self.query_history: List[Dict[str, Any]] = []
        
        logger.info(f"Environmental Query Demo initialized with {self.dimensions}-dimensional measurement")
    
    async def process_environmental_query(self, query_text: str,
                                        atmospheric_processing: bool = True,
                                        cross_modal_validation: bool = True,
                                        full_dimensions: bool = True) -> Dict[str, Any]:
        """
        Process query through environmental frameworks.
        
        Args:
            query_text: Query text to process
            atmospheric_processing: Enable atmospheric molecular processing
            cross_modal_validation: Enable cross-modal BMD validation
            full_dimensions: Use all 12 dimensions vs simplified measurement
            
        Returns:
            Environmental processing result
        """
        start_time = time.time()
        
        logger.info(f"Processing environmental query: '{query_text}'")
        
        # Phase 1: Environmental measurement
        logger.info("Phase 1: Twelve-dimensional environmental measurement...")
        environmental_state = await self.environmental_measurement.measure_environment()
        
        # Get measurement statistics
        measurement_stats = self.environmental_measurement.get_measurement_statistics()
        
        # Phase 2: Environmental integration
        logger.info("Phase 2: Environmental context integration...")
        integration_result = await self.environmental_integrator.integrate_environmental_context(query_text)
        
        # Phase 3: Atmospheric processing (if enabled)
        atmospheric_result = None
        if atmospheric_processing:
            logger.info("Phase 3: Atmospheric molecular processing...")
            query = Query(content=query_text, environmental_context=environmental_state)
            atmospheric_result = await self.atmospheric_network.process_query(query, environmental_state)
        
        # Phase 4: Cross-modal validation (if enabled) 
        cross_modal_result = None
        if cross_modal_validation:
            logger.info("Phase 4: Cross-modal BMD validation...")
            cross_modal_result = self.cross_modal_validator.validate_cross_modal_equivalence(
                query_text, environmental_state
            )
        
        processing_time = time.time() - start_time
        
        # Compile comprehensive result
        result = {
            'query_text': query_text,
            'environmental_measurement': {
                'state': self._environmental_state_to_dict(environmental_state),
                'uniqueness_factor': environmental_state.calculate_uniqueness(),
                'measurement_stats': measurement_stats,
                'dimensions_active': self.dimensions
            },
            'environmental_integration': {
                'integration_quality': integration_result['integration_quality'],
                'correlation_factor': integration_result['correlation_factor'],
                'content_complexity': integration_result['content_complexity'],
                'environmental_enhancement': integration_result['environmental_enhancement'],
                'uniqueness_amplification': integration_result['uniqueness_amplification']
            },
            'atmospheric_processing': atmospheric_result,
            'cross_modal_validation': self._format_cross_modal_result(cross_modal_result),
            'processing_time': processing_time,
            'processing_modes': {
                'atmospheric_processing': atmospheric_processing,
                'cross_modal_validation': cross_modal_validation,
                'full_dimensions': full_dimensions
            }
        }
        
        # Store in history
        self.query_history.append(result)
        
        logger.info(f"Environmental query processing completed in {processing_time:.3f}s")
        
        return result
    
    def _environmental_state_to_dict(self, env_state: EnvironmentalState) -> Dict[str, Any]:
        """Convert environmental state to dictionary format."""
        return {
            'biometric': {
                'physiological_arousal': env_state.biometric.physiological_arousal,
                'cognitive_load': env_state.biometric.cognitive_load,
                'attention_state': env_state.biometric.attention_state,
                'stress_level': env_state.biometric.stress_level
            },
            'spatial': {
                'location_x': env_state.spatial.location_x,
                'location_y': env_state.spatial.location_y,
                'location_z': env_state.spatial.location_z,
                'orientation': env_state.spatial.orientation
            },
            'atmospheric_pressure': env_state.atmospheric_pressure,
            'temperature': env_state.temperature,
            'humidity': env_state.humidity,
            'light_level': env_state.light_level,
            'sound_level': env_state.sound_level,
            'electromagnetic_field': env_state.electromagnetic_field,
            'cosmic_background': env_state.cosmic_background,
            'quantum_coherence': env_state.quantum_coherence,
            'temporal_flow': env_state.temporal_flow
        }
    
    def _format_cross_modal_result(self, cross_modal_result) -> Optional[Dict[str, Any]]:
        """Format cross-modal validation result."""
        if not cross_modal_result:
            return None
        
        from graffiti.environmental.cross_modal_bmv import ModalityType
        
        formatted = {
            'modality_results': {},
            'equivalence_assessment': {}
        }
        
        for modality, result in cross_modal_result.items():
            formatted['modality_results'][modality.value] = {
                'catalysis_strength': result.catalysis_strength,
                'information_content': result.information_content,
                'environmental_coupling': result.environmental_coupling,
                'consciousness_optimization': result.consciousness_optimization
            }
        
        # Calculate overall equivalence
        catalysis_values = [r.catalysis_strength for r in cross_modal_result.values()]
        if catalysis_values:
            import numpy as np
            mean_catalysis = np.mean(catalysis_values)
            std_catalysis = np.std(catalysis_values)
            equivalence_score = 1.0 / (1.0 + std_catalysis / (mean_catalysis + 0.001))
            
            formatted['equivalence_assessment'] = {
                'equivalence_score': equivalence_score,
                'mean_catalysis': mean_catalysis,
                'catalysis_variance': std_catalysis,
                'modalities_validated': len(catalysis_values)
            }
        
        return formatted
    
    async def demonstrate_dimensional_measurement(self) -> Dict[str, Any]:
        """Demonstrate detailed dimensional measurement."""
        logger.info("Demonstrating twelve-dimensional measurement...")
        
        # Take multiple measurements to show variation
        measurements = []
        for i in range(3):
            env_state = await self.environmental_measurement.measure_environment()
            measurements.append({
                'measurement_id': i + 1,
                'uniqueness': env_state.calculate_uniqueness(),
                'quantum_coherence': env_state.quantum_coherence,
                'temporal_flow': env_state.temporal_flow,
                'atmospheric_pressure': env_state.atmospheric_pressure,
                'temperature': env_state.temperature,
                'light_level': env_state.light_level,
                'sound_level': env_state.sound_level
            })
            
            # Brief delay between measurements
            await asyncio.sleep(0.1)
        
        # Calculate measurement statistics
        import numpy as np
        uniqueness_values = [m['uniqueness'] for m in measurements]
        coherence_values = [m['quantum_coherence'] for m in measurements]
        
        return {
            'measurements': measurements,
            'statistical_analysis': {
                'uniqueness_mean': np.mean(uniqueness_values),
                'uniqueness_std': np.std(uniqueness_values),
                'coherence_mean': np.mean(coherence_values),
                'coherence_std': np.std(coherence_values),
                'measurement_stability': 1.0 - np.std(uniqueness_values)
            },
            'dimensions_measured': len(EnvironmentalDimension),
            'demonstration': 'Twelve-dimensional environmental measurement with temporal variation'
        }
    
    async def demonstrate_atmospheric_consensus(self, query_text: str) -> Dict[str, Any]:
        """Demonstrate atmospheric molecular consensus."""
        logger.info("Demonstrating atmospheric molecular consensus...")
        
        # Create query and environmental context
        env_state = await self.environmental_measurement.measure_environment()
        query = Query(content=query_text, environmental_context=env_state)
        
        # Process through atmospheric network
        atmospheric_result = await self.atmospheric_network.process_query(query, env_state)
        
        # Get network statistics
        network_stats = self.atmospheric_network.get_network_statistics()
        
        return {
            'query_text': query_text,
            'atmospheric_processing': atmospheric_result,
            'network_statistics': network_stats,
            'molecular_simulation_scale': self.atmospheric_network.target_molecules,
            'demonstration': 'Atmospheric molecular processing achieving consensus across molecular clusters'
        }
    
    def analyze_environmental_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in environmental query processing."""
        if not self.query_history:
            return {'message': 'No environmental queries processed yet'}
        
        # Extract metrics from history
        integration_qualities = []
        uniqueness_factors = []
        processing_times = []
        atmospheric_consensus_levels = []
        
        for query_result in self.query_history:
            integration_qualities.append(query_result['environmental_integration']['integration_quality'])
            uniqueness_factors.append(query_result['environmental_measurement']['uniqueness_factor'])
            processing_times.append(query_result['processing_time'])
            
            if query_result['atmospheric_processing']:
                consensus = query_result['atmospheric_processing']['consensus_result']
                atmospheric_consensus_levels.append(consensus.consensus_level)
        
        import numpy as np
        
        # Pattern analysis
        integration_trend = np.polyfit(range(len(integration_qualities)), integration_qualities, 1)[0] if len(integration_qualities) > 1 else 0.0
        
        analysis = {
            'total_environmental_queries': len(self.query_history),
            'average_integration_quality': np.mean(integration_qualities),
            'integration_improvement_trend': integration_trend,
            'average_uniqueness_factor': np.mean(uniqueness_factors),
            'uniqueness_stability': 1.0 - np.std(uniqueness_factors),
            'average_processing_time': np.mean(processing_times),
            'atmospheric_processing_usage': len(atmospheric_consensus_levels) / len(self.query_history),
            'average_atmospheric_consensus': np.mean(atmospheric_consensus_levels) if atmospheric_consensus_levels else 0.0,
            'environmental_pattern_quality': np.mean(integration_qualities) * (1.0 - np.std(uniqueness_factors))
        }
        
        return analysis
    
    def get_environmental_statistics(self) -> Dict[str, Any]:
        """Get comprehensive environmental processing statistics."""
        if not self.query_history:
            return {'message': 'No environmental queries processed yet'}
        
        # Measurement system statistics
        measurement_stats = self.environmental_measurement.get_measurement_statistics()
        
        # Integration system statistics  
        integration_patterns = self.environmental_integrator.analyze_environmental_patterns()
        
        # Atmospheric network statistics
        network_stats = self.atmospheric_network.get_network_statistics()
        
        # Cross-modal validation statistics
        validation_stats = self.cross_modal_validator.get_validation_statistics()
        
        return {
            'measurement_system': measurement_stats,
            'integration_patterns': integration_patterns,
            'atmospheric_network': network_stats,
            'cross_modal_validation': validation_stats,
            'query_processing': self.analyze_environmental_patterns(),
            'dimensions_supported': self.dimensions,
            'total_environmental_operations': len(self.query_history)
        }


def main():
    """Main entry point for environmental query demo."""
    parser = argparse.ArgumentParser(description="Graffiti Environmental Query Processing Demo")
    parser.add_argument("--query", type=str, required=True,
                       help="Query for environmental processing")
    parser.add_argument("--dimensions", type=int, default=12,
                       help="Number of environmental dimensions (1-12)")
    parser.add_argument("--no-atmospheric", action='store_true',
                       help="Disable atmospheric molecular processing")
    parser.add_argument("--no-validation", action='store_true',
                       help="Disable cross-modal validation")
    parser.add_argument("--demo-measurement", action='store_true',
                       help="Demonstrate dimensional measurement")
    parser.add_argument("--demo-atmospheric", action='store_true',
                       help="Demonstrate atmospheric consensus")
    parser.add_argument("--analyze-patterns", action='store_true',
                       help="Analyze environmental patterns")
    parser.add_argument("--stats", action='store_true',
                       help="Show environmental statistics")
    
    args = parser.parse_args()
    
    async def run_environmental_demo():
        demo = EnvironmentalQueryDemo(dimensions=args.dimensions)
        
        if args.stats:
            stats = demo.get_environmental_statistics()
            print("Environmental Processing Statistics:")
            print(json.dumps(stats, indent=2))
            return
        
        if args.analyze_patterns:
            analysis = demo.analyze_environmental_patterns()
            print("Environmental Pattern Analysis:")
            print(json.dumps(analysis, indent=2))
            return
        
        if args.demo_measurement:
            result = await demo.demonstrate_dimensional_measurement()
            print("Twelve-Dimensional Measurement Demonstration:")
            print(json.dumps(result, indent=2))
            return
        
        if args.demo_atmospheric:
            result = await demo.demonstrate_atmospheric_consensus(args.query)
            print("Atmospheric Molecular Consensus Demonstration:")
            print(json.dumps(result, indent=2))
            return
        
        # Main environmental query processing
        result = await demo.process_environmental_query(
            args.query,
            atmospheric_processing=not args.no_atmospheric,
            cross_modal_validation=not args.no_validation,
            full_dimensions=True
        )
        
        print("ENVIRONMENTAL QUERY PROCESSING RESULT")
        print("="*60)
        
        # Display query and environmental context
        print(f"Query: '{result['query_text']}'")
        print(f"Dimensions: {result['environmental_measurement']['dimensions_active']}")
        
        # Environmental measurement results
        print(f"\nEnvironmental Measurement:")
        env_measurement = result['environmental_measurement']
        print(f"  Uniqueness Factor: {env_measurement['uniqueness_factor']:.6f}")
        if env_measurement['measurement_stats']:
            stats = env_measurement['measurement_stats']
            print(f"  Reliability Rate: {stats.get('reliability_rate', 0):.1%}")
            print(f"  Integration Score: {stats.get('average_integration_score', 0):.3f}")
        
        # Environmental integration results
        print(f"\nEnvironmental Integration:")
        integration = result['environmental_integration']
        print(f"  Integration Quality: {integration['integration_quality']:.3f}")
        print(f"  Content Complexity: {integration['content_complexity']:.3f}")
        print(f"  Environmental Enhancement: {integration['environmental_enhancement']:.3f}")
        print(f"  Uniqueness Amplification: {integration['uniqueness_amplification']:.3f}")
        
        # Atmospheric processing results
        if result['atmospheric_processing']:
            print(f"\nAtmospheric Molecular Processing:")
            atmospheric = result['atmospheric_processing']
            consensus = atmospheric['consensus_result']
            print(f"  Consensus Reached: {consensus.consensus_reached}")
            print(f"  Consensus Level: {consensus.consensus_level:.3f}")
            print(f"  Participating Molecules: {consensus.participating_molecules:,}")
            print(f"  Processing Time: {consensus.processing_time:.3f}s")
            print(f"  Environmental Coupling: {consensus.environmental_coupling:.3f}")
        
        # Cross-modal validation results
        if result['cross_modal_validation']:
            print(f"\nCross-Modal BMD Validation:")
            validation = result['cross_modal_validation']
            if 'equivalence_assessment' in validation:
                equiv = validation['equivalence_assessment']
                print(f"  Equivalence Score: {equiv['equivalence_score']:.3f}")
                print(f"  Mean Catalysis: {equiv['mean_catalysis']:.3f}")
                print(f"  Modalities Validated: {equiv['modalities_validated']}")
        
        print(f"\nTotal Processing Time: {result['processing_time']:.3f}s")
        print("Environmental consciousness integration through twelve-dimensional measurement complete.")
    
    asyncio.run(run_environmental_demo())


if __name__ == "__main__":
    main()
