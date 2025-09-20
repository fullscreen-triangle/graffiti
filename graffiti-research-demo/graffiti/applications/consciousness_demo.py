"""
Consciousness Processing Demo Application

Demonstration of BMD consciousness processing through predetermined frame
selection, sanity checking, and cross-modal information catalysis.
"""

import asyncio
import argparse
import json
import sys
import time
import logging
import random
from typing import Dict, List, Any, Optional

from graffiti.core.types import (
    Query,
    QueryId,
    EnvironmentalState,
    UserContext,
    ResponseType,
    Urgency,
)
from graffiti.core.bmd_operations import BiologicalMaxwellDemon
from graffiti.environmental.twelve_dimensional import EnvironmentalMeasurement
from graffiti.environmental.cross_modal_bmv import BMDInformationCatalyst

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ConsciousnessDemo:
    """
    Consciousness processing demonstration system.
    
    Demonstrates BMD consciousness processing, frame selection, sanity checking,
    and cross-modal information catalysis.
    """
    
    def __init__(self):
        """Initialize consciousness demo."""
        self.bmd_processor = BiologicalMaxwellDemon()
        self.environmental_measurement = EnvironmentalMeasurement()
        self.information_catalyst = BMDInformationCatalyst()
        
        self.processing_history: List[Dict[str, Any]] = []
        
        logger.info("Consciousness Demo initialized with BMD processing capabilities")
    
    async def process_consciousness_query(self, input_text: str,
                                        miracle_mode: bool = False,
                                        catalyst_mode: bool = True) -> Dict[str, Any]:
        """
        Process input through consciousness frameworks.
        
        Args:
            input_text: Input text to process
            miracle_mode: Enable miracle processing for weak inputs
            catalyst_mode: Enable cross-modal catalysis
            
        Returns:
            Consciousness processing result
        """
        start_time = time.time()
        
        logger.info(f"Processing consciousness query: '{input_text}'")
        
        # Phase 1: Environmental measurement
        logger.info("Phase 1: Measuring environmental context...")
        environmental_state = await self.environmental_measurement.measure_environment()
        
        # Phase 2: BMD consciousness processing
        logger.info("Phase 2: BMD consciousness processing...")
        bmd_result = self.bmd_processor.process_full_bmd_cycle(environmental_state, input_text)
        
        # Phase 3: Cross-modal catalysis (if enabled)
        catalysis_result = None
        if catalyst_mode:
            logger.info("Phase 3: Cross-modal information catalysis...")
            catalysis_result = self.information_catalyst.catalyze_information(
                input_text, environmental_state
            )
        
        # Phase 4: Miracle processing for weak queries (if enabled)
        miracle_enhancement = None
        if miracle_mode:
            logger.info("Phase 4: Miracle enhancement for weak input...")
            miracle_enhancement = await self._process_miracle_enhancement(
                input_text, bmd_result, environmental_state
            )
        
        processing_time = time.time() - start_time
        
        # Compile results
        result = {
            'input_text': input_text,
            'environmental_context': {
                'uniqueness': environmental_state.calculate_uniqueness(),
                'quantum_coherence': environmental_state.quantum_coherence,
                'temporal_flow': environmental_state.temporal_flow,
                'biometric_arousal': environmental_state.biometric.physiological_arousal,
                'attention_state': environmental_state.biometric.attention_state
            },
            'bmd_processing': {
                'frames_activated': len(bmd_result['frame_selection'].selected_frames),
                'selection_confidence': bmd_result['frame_selection'].selection_confidence,
                'processing_efficiency': bmd_result['bmd_efficiency'],
                'sanity_status': bmd_result['sanity_check'].sanity_status,
                'correspondence_score': bmd_result['sanity_check'].correspondence_score,
                'conscious_content': bmd_result['conscious_state']['conscious_content']
            },
            'cross_modal_catalysis': catalysis_result,
            'miracle_enhancement': miracle_enhancement,
            'processing_time': processing_time,
            'modes_enabled': {
                'miracle_mode': miracle_mode,
                'catalyst_mode': catalyst_mode
            }
        }
        
        # Store in history
        self.processing_history.append(result)
        
        logger.info(f"Consciousness processing completed in {processing_time:.3f}s")
        
        return result
    
    async def _process_miracle_enhancement(self, input_text: str,
                                         bmd_result: Dict[str, Any],
                                         environmental_state: EnvironmentalState) -> Dict[str, Any]:
        """Process miracle enhancement for weak queries."""
        
        # Assess input weakness
        weakness_factors = []
        
        # Length factor
        if len(input_text.strip()) < 10:
            weakness_factors.append("very_short")
        
        # Ambiguity factor
        ambiguous_words = ['something', 'maybe', 'kind of', 'sort of', 'possibly', 'perhaps']
        if any(word in input_text.lower() for word in ambiguous_words):
            weakness_factors.append("high_ambiguity")
        
        # Uncertainty factor
        uncertainty_words = ['?', 'unsure', 'dont know', "don't know", 'confused']
        if any(word in input_text.lower() for word in uncertainty_words):
            weakness_factors.append("high_uncertainty")
        
        # Calculate miracle potential
        weakness_level = len(weakness_factors) / 3.0  # Normalize to [0, 1]
        miracle_potential = weakness_level * 0.8 + random.uniform(0.1, 0.2)
        
        # Generate miracle enhancement
        if miracle_potential > 0.4:
            enhanced_interpretation = self._generate_miracle_interpretation(
                input_text, weakness_factors, environmental_state
            )
            
            miracle_result = {
                'miracle_triggered': True,
                'weakness_factors': weakness_factors,
                'miracle_potential': miracle_potential,
                'enhanced_interpretation': enhanced_interpretation,
                'enhancement_confidence': min(0.9, miracle_potential * 1.2)
            }
        else:
            miracle_result = {
                'miracle_triggered': False,
                'weakness_factors': weakness_factors,
                'miracle_potential': miracle_potential,
                'reason': 'Input not sufficiently weak to trigger miracle enhancement'
            }
        
        return miracle_result
    
    def _generate_miracle_interpretation(self, input_text: str, 
                                       weakness_factors: List[str],
                                       environmental_state: EnvironmentalState) -> str:
        """Generate miracle-enhanced interpretation."""
        
        interpretation = f"Miracle Enhancement of '{input_text}':\n\n"
        
        if 'very_short' in weakness_factors:
            interpretation += "• Length Amplification: Short input suggests focused intent requiring expansion\n"
        
        if 'high_ambiguity' in weakness_factors:
            interpretation += "• Ambiguity Resolution: Uncertain language indicates exploration of possibilities\n"
        
        if 'high_uncertainty' in weakness_factors:
            interpretation += "• Uncertainty Processing: Confusion signals need for clarity and direction\n"
        
        # Environmental miracle factors
        env_uniqueness = environmental_state.calculate_uniqueness()
        if env_uniqueness > 0.5:
            interpretation += f"• Environmental Enhancement: Unique environmental context ({env_uniqueness:.3f}) amplifies weak query potential\n"
        
        interpretation += f"\nMiracle Interpretation:\n"
        interpretation += f"The weak query becomes an opportunity for enhanced exploration. "
        interpretation += f"Undefined victory conditions allow the system to discover optimal outcomes "
        interpretation += f"through environmental context integration. The weakness transforms into "
        interpretation += f"strength through miracle processing that finds potential in uncertainty."
        
        return interpretation
    
    def demonstrate_frame_selection(self, input_text: str) -> Dict[str, Any]:
        """Demonstrate BMD frame selection mechanism."""
        logger.info("Demonstrating BMD frame selection...")
        
        # Create environmental context
        environmental_state = EnvironmentalState()
        
        # Process experience
        processed_experience = self.bmd_processor.process_experience(environmental_state, input_text)
        
        # Select frames
        frame_selection = self.bmd_processor.select_frames(processed_experience)
        
        return {
            'input_text': input_text,
            'processed_experience': {
                'total_information': processed_experience['total_information'],
                'sufficiency_met': processed_experience['sufficiency_met'],
                'enhancement_factor': processed_experience['enhancement_factor']
            },
            'frame_selection': {
                'selected_frames': frame_selection.selected_frames,
                'frame_relevance_scores': frame_selection.frame_relevance_scores,
                'selection_confidence': frame_selection.selection_confidence,
                'processing_efficiency': frame_selection.processing_efficiency
            },
            'demonstration': 'BMD frame selection through loaded dice mechanism'
        }
    
    def demonstrate_sanity_checking(self, conscious_state: Dict[str, Any]) -> Dict[str, Any]:
        """Demonstrate sanity checking process."""
        logger.info("Demonstrating sanity checking...")
        
        # Perform sanity check
        sanity_result = self.bmd_processor.perform_sanity_check(conscious_state)
        
        return {
            'conscious_state_intensity': conscious_state.get('state_intensity', 0),
            'fusion_quality': conscious_state.get('fusion_quality', 0),
            'sanity_check': {
                'correspondence_score': sanity_result.correspondence_score,
                'belief_alignment': sanity_result.belief_alignment,
                'collective_validation': sanity_result.collective_validation,
                'functional_threshold_met': sanity_result.functional_threshold_met,
                'sanity_status': sanity_result.sanity_status
            },
            'demonstration': 'Sanity checking through correspondence validation rather than meaning extraction'
        }
    
    async def demonstrate_cross_modal_catalysis(self, input_text: str) -> Dict[str, Any]:
        """Demonstrate cross-modal information catalysis."""
        logger.info("Demonstrating cross-modal catalysis...")
        
        environmental_state = await self.environmental_measurement.measure_environment()
        
        catalysis_result = self.information_catalyst.catalyze_information(
            input_text, environmental_state
        )
        
        return {
            'input_text': input_text,
            'catalysis_result': catalysis_result,
            'modalities_processed': len(catalysis_result['modality_results']),
            'catalysis_effectiveness': catalysis_result['catalysis_effectiveness'],
            'integration_quality': catalysis_result['integration_quality'],
            'demonstration': 'Cross-modal BMD information catalysis across visual, audio, and semantic modalities'
        }
    
    def get_consciousness_statistics(self) -> Dict[str, Any]:
        """Get consciousness processing statistics."""
        if not self.processing_history:
            return {'message': 'No consciousness processing performed yet'}
        
        # Extract metrics
        processing_times = [p['processing_time'] for p in self.processing_history]
        selection_confidences = [p['bmd_processing']['selection_confidence'] for p in self.processing_history]
        sanity_scores = [p['bmd_processing']['correspondence_score'] for p in self.processing_history]
        
        # Count successful processing
        successful_processing = [p for p in self.processing_history 
                               if p['bmd_processing']['sanity_status'] == 'functional']
        success_rate = len(successful_processing) / len(self.processing_history)
        
        return {
            'total_consciousness_operations': len(self.processing_history),
            'processing_success_rate': success_rate,
            'average_processing_time': sum(processing_times) / len(processing_times),
            'average_selection_confidence': sum(selection_confidences) / len(selection_confidences),
            'average_sanity_score': sum(sanity_scores) / len(sanity_scores),
            'miracle_mode_usage': sum(1 for p in self.processing_history if p['modes_enabled']['miracle_mode']) / len(self.processing_history),
            'catalyst_mode_usage': sum(1 for p in self.processing_history if p['modes_enabled']['catalyst_mode']) / len(self.processing_history)
        }


def main():
    """Main entry point for consciousness demo."""
    parser = argparse.ArgumentParser(description="Graffiti Consciousness Processing Demo")
    parser.add_argument("--input", type=str, required=True,
                       help="Input text for consciousness processing")
    parser.add_argument("--miracle-mode", action='store_true',
                       help="Enable miracle processing for weak inputs")
    parser.add_argument("--no-catalyst", action='store_true',
                       help="Disable cross-modal catalysis")
    parser.add_argument("--demo-frame-selection", action='store_true',
                       help="Demonstrate frame selection mechanism")
    parser.add_argument("--demo-catalysis", action='store_true',
                       help="Demonstrate cross-modal catalysis")
    parser.add_argument("--stats", action='store_true',
                       help="Show consciousness processing statistics")
    
    args = parser.parse_args()
    
    async def run_consciousness_demo():
        demo = ConsciousnessDemo()
        
        if args.stats:
            stats = demo.get_consciousness_statistics()
            print("Consciousness Processing Statistics:")
            print(json.dumps(stats, indent=2))
            return
        
        if args.demo_frame_selection:
            result = demo.demonstrate_frame_selection(args.input)
            print("BMD Frame Selection Demonstration:")
            print(json.dumps(result, indent=2))
            return
        
        if args.demo_catalysis:
            result = await demo.demonstrate_cross_modal_catalysis(args.input)
            print("Cross-Modal Catalysis Demonstration:")
            print(json.dumps(result, indent=2))
            return
        
        # Main consciousness processing
        result = await demo.process_consciousness_query(
            args.input,
            miracle_mode=args.miracle_mode,
            catalyst_mode=not args.no_catalyst
        )
        
        print("CONSCIOUSNESS PROCESSING RESULT")
        print("="*50)
        
        # Display conscious content
        print(f"Input: '{result['input_text']}'")
        print(f"\nBMD Processing:")
        bmd = result['bmd_processing']
        print(f"  Frames Activated: {bmd['frames_activated']}")
        print(f"  Selection Confidence: {bmd['selection_confidence']:.3f}")
        print(f"  Sanity Status: {bmd['sanity_status']}")
        print(f"  Correspondence Score: {bmd['correspondence_score']:.3f}")
        
        print(f"\nConscious Content:")
        print(bmd['conscious_content'])
        
        # Display environmental context
        print(f"\nEnvironmental Context:")
        env = result['environmental_context']
        print(f"  Uniqueness: {env['uniqueness']:.6f}")
        print(f"  Quantum Coherence: {env['quantum_coherence']:.3f}")
        print(f"  Attention State: {env['attention_state']:.3f}")
        
        # Display catalysis results if enabled
        if result['cross_modal_catalysis']:
            catalysis = result['cross_modal_catalysis']
            print(f"\nCross-Modal Catalysis:")
            print(f"  Effectiveness: {catalysis['catalysis_effectiveness']:.3f}")
            print(f"  Integration Quality: {catalysis['integration_quality']:.3f}")
            print(f"  Modalities: {len(catalysis['modality_results'])}")
        
        # Display miracle enhancement if enabled
        if result['miracle_enhancement'] and result['miracle_enhancement']['miracle_triggered']:
            miracle = result['miracle_enhancement']
            print(f"\nMiracle Enhancement:")
            print(f"  Miracle Potential: {miracle['miracle_potential']:.3f}")
            print(f"  Weakness Factors: {', '.join(miracle['weakness_factors'])}")
            print(f"  Enhancement Confidence: {miracle['enhancement_confidence']:.3f}")
        
        print(f"\nProcessing Time: {result['processing_time']:.3f}s")
        print("Consciousness processing demonstrates BMD frame selection rather than meaning extraction.")
    
    asyncio.run(run_consciousness_demo())


if __name__ == "__main__":
    main()
