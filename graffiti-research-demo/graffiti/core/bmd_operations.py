"""
Biological Maxwell Demon (BMD) Operations

Implementation of consciousness as predetermined frame selection through BMD
information catalysis and selective attention mechanisms.

Based on theoretical framework: BMD selectively fuses experience with memory frames
to create conscious states, operating as information catalyst through predetermined
cognitive frameworks.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import logging
from dataclasses import dataclass, field
from enum import Enum
import time
import random
import math

from graffiti.core.types import (
    EnvironmentalState,
    Query,
    BMDFrameSelection,
    BiometricDimension,
)

logger = logging.getLogger(__name__)


class FrameType(Enum):
    """Types of memory frames for BMD selection."""
    KNOWLEDGE_FRAME = "knowledge_frame"      # Correlations and counterfactuals
    TRUTH_FRAME = "truth_frame"              # Collective constructions
    EXPERIENCE_FRAME = "experience_frame"    # Processed sensory input
    BELIEF_FRAME = "belief_frame"           # Validation frameworks
    CONTEXTUAL_FRAME = "contextual_frame"   # Situational adaptations


@dataclass
class MemoryFrame:
    """Memory frame containing knowledge and truth components."""
    frame_id: str
    frame_type: FrameType
    knowledge_component: Dict[str, Any]  # Correlations, counterfactuals, relationships
    truth_component: Dict[str, Any]      # Collective agreements, social constructions
    relevance_score: float = 0.0
    recency_score: float = 0.0
    emotional_weight: float = 0.0
    belief_compatibility: float = 0.0
    counterfactual_richness: float = 0.0
    activation_threshold: float = 0.5
    
    def calculate_utility(self, context: Dict[str, Any]) -> float:
        """Calculate frame utility for given context."""
        utility = 0.0
        
        # Weight factors from theoretical framework
        w1, w2, w3, w4, w5 = 0.25, 0.15, 0.2, 0.25, 0.15
        
        utility += w1 * self.relevance_score
        utility += w2 * self.recency_score  
        utility += w3 * self.emotional_weight
        utility += w4 * self.belief_compatibility
        utility += w5 * self.counterfactual_richness
        
        return utility
    
    def is_activated(self, context: Dict[str, Any]) -> bool:
        """Check if frame is activated given context."""
        return self.calculate_utility(context) >= self.activation_threshold


@dataclass
class SanityCheckResult:
    """Result of sanity checking process."""
    correspondence_score: float
    belief_alignment: float  
    collective_validation: float
    functional_threshold_met: bool
    sanity_status: str  # "functional", "dysfunction"


class BiologicalMaxwellDemon:
    """
    Biological Maxwell Demon implementing consciousness through frame selection.
    
    Operates through three core components:
    1. Experience processing (sufficient environmental information)
    2. Memory frame selection (predetermined cognitive frameworks)  
    3. Fusion process (experience + frames → conscious state)
    """
    
    def __init__(self):
        """Initialize Biological Maxwell Demon."""
        self.frame_store: Dict[str, MemoryFrame] = {}
        self.selection_history: List[BMDFrameSelection] = []
        self.loaded_dice_weights: Dict[str, float] = {
            "relevance": 0.3,
            "recency": 0.2, 
            "emotional": 0.25,
            "belief": 0.15,
            "counterfactual": 0.1
        }
        
        # Initialize basic frame store
        self._initialize_frame_store()
    
    def _initialize_frame_store(self):
        """Initialize BMD memory frame store with common cognitive patterns."""
        
        # Knowledge frames - correlations and counterfactuals
        knowledge_frames = [
            MemoryFrame(
                frame_id="pattern_recognition",
                frame_type=FrameType.KNOWLEDGE_FRAME,
                knowledge_component={
                    "correlations": ["similarity_patterns", "structural_relationships"],
                    "counterfactuals": ["alternative_patterns", "pattern_variations"],
                    "causal_relationships": ["cause_effect_chains", "pattern_emergence"]
                },
                truth_component={
                    "collective_agreements": ["pattern_naming_conventions"],
                    "social_constructions": ["shared_pattern_recognition"]
                },
                relevance_score=0.8,
                recency_score=0.6,
                emotional_weight=0.4,
                belief_compatibility=0.7,
                counterfactual_richness=0.9
            ),
            MemoryFrame(
                frame_id="analytical_reasoning",
                frame_type=FrameType.KNOWLEDGE_FRAME,
                knowledge_component={
                    "correlations": ["logical_relationships", "systematic_analysis"],
                    "counterfactuals": ["alternative_reasoning_paths"],
                    "causal_relationships": ["logical_inference_chains"]
                },
                truth_component={
                    "collective_agreements": ["reasoning_standards"],
                    "social_constructions": ["analytical_frameworks"]
                },
                relevance_score=0.75,
                recency_score=0.5,
                emotional_weight=0.2,
                belief_compatibility=0.8,
                counterfactual_richness=0.7
            )
        ]
        
        # Truth frames - collective constructions
        truth_frames = [
            MemoryFrame(
                frame_id="social_validation",
                frame_type=FrameType.TRUTH_FRAME,
                knowledge_component={
                    "correlations": ["social_consensus_patterns"],
                    "counterfactuals": ["alternative_social_frameworks"]
                },
                truth_component={
                    "collective_agreements": ["shared_beliefs", "common_understandings"],
                    "social_constructions": ["cultural_frameworks", "collective_meaning"]
                },
                relevance_score=0.6,
                recency_score=0.8,
                emotional_weight=0.7,
                belief_compatibility=0.9,
                counterfactual_richness=0.5
            )
        ]
        
        # Combine all frames
        all_frames = knowledge_frames + truth_frames
        
        for frame in all_frames:
            self.frame_store[frame.frame_id] = frame
    
    def process_experience(self, environmental_input: EnvironmentalState, 
                          query_content: str) -> Dict[str, Any]:
        """
        Process environmental information to determine sufficiency for consciousness.
        
        Args:
            environmental_input: Environmental state information
            query_content: Content requiring conscious processing
            
        Returns:
            Processed experience with sufficiency assessment
        """
        # Calculate environmental information content
        env_uniqueness = environmental_input.calculate_uniqueness()
        
        # Assess information sufficiency
        base_information = len(query_content) / 1000.0  # Normalize content length
        environmental_enhancement = env_uniqueness * 2.0
        biometric_factor = (
            environmental_input.biometric.physiological_arousal +
            environmental_input.biometric.cognitive_load +
            environmental_input.biometric.attention_state
        ) / 3.0
        
        total_information = base_information + environmental_enhancement + biometric_factor
        
        # Determine sufficiency (threshold-based)
        sufficiency_threshold = 0.5
        is_sufficient = total_information >= sufficiency_threshold
        
        return {
            "environmental_info": environmental_input,
            "content_info": query_content,
            "total_information": total_information,
            "sufficiency_met": is_sufficient,
            "enhancement_factor": environmental_enhancement,
            "biometric_contribution": biometric_factor,
            "processing_context": {
                "uniqueness": env_uniqueness,
                "attention_state": environmental_input.biometric.attention_state,
                "cognitive_load": environmental_input.biometric.cognitive_load
            }
        }
    
    def select_frames(self, processed_experience: Dict[str, Any]) -> BMDFrameSelection:
        """
        Select memory frames through BMD loaded dice mechanism.
        
        Args:
            processed_experience: Processed experience from environmental input
            
        Returns:
            Frame selection result
        """
        if not processed_experience["sufficiency_met"]:
            # Insufficient information - minimal frame activation
            return BMDFrameSelection(
                selected_frames=[],
                frame_relevance_scores=[],
                selection_confidence=0.1,
                processing_efficiency=0.2
            )
        
        # Context for frame selection
        context = {
            "information_level": processed_experience["total_information"],
            "environmental_uniqueness": processed_experience["processing_context"]["uniqueness"],
            "attention_state": processed_experience["processing_context"]["attention_state"],
            "cognitive_load": processed_experience["processing_context"]["cognitive_load"],
            "content_complexity": len(processed_experience["content_info"].split()) / 50.0
        }
        
        # Calculate selection probabilities using loaded dice
        frame_probabilities = self._calculate_frame_probabilities(context)
        
        # Select frames based on probabilities (BMD biased selection)
        selected_frames = []
        relevance_scores = []
        
        for frame_id, probability in frame_probabilities.items():
            # Biased selection based on BMD loaded dice weights
            if random.random() < probability:
                selected_frames.append(frame_id)
                relevance_scores.append(probability)
        
        # Ensure at least one frame is selected for functional processing
        if not selected_frames:
            # Emergency frame selection - choose highest probability frame
            best_frame_id = max(frame_probabilities.keys(), key=lambda k: frame_probabilities[k])
            selected_frames = [best_frame_id]
            relevance_scores = [frame_probabilities[best_frame_id]]
        
        # Calculate selection confidence and processing efficiency
        avg_relevance = np.mean(relevance_scores) if relevance_scores else 0.0
        selection_confidence = min(1.0, avg_relevance * 1.2)
        processing_efficiency = selection_confidence * 0.8  # Efficiency correlates with confidence
        
        result = BMDFrameSelection(
            selected_frames=selected_frames,
            frame_relevance_scores=relevance_scores,
            selection_confidence=selection_confidence,
            processing_efficiency=processing_efficiency
        )
        
        # Record selection history
        self.selection_history.append(result)
        
        return result
    
    def _calculate_frame_probabilities(self, context: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate frame selection probabilities using BMD loaded dice mechanism.
        
        Args:
            context: Processing context
            
        Returns:
            Frame probabilities dictionary
        """
        frame_probabilities = {}
        
        for frame_id, frame in self.frame_store.items():
            # Calculate utility based on theoretical framework
            utility = frame.calculate_utility(context)
            
            # Apply loaded dice weighting (systematic bias for functional selection)
            weighted_utility = utility
            
            # Enhance probability based on context relevance
            if context["information_level"] > 0.7 and frame.frame_type == FrameType.KNOWLEDGE_FRAME:
                weighted_utility *= 1.3  # Boost knowledge frames for complex information
            
            if context["attention_state"] > 0.7 and frame.frame_type == FrameType.TRUTH_FRAME:
                weighted_utility *= 1.2  # Boost truth frames for high attention
            
            # Convert utility to probability using softmax-like function
            probability = 1.0 / (1.0 + np.exp(-5.0 * (weighted_utility - 0.5)))
            
            frame_probabilities[frame_id] = probability
        
        return frame_probabilities
    
    def fuse_experience_and_frames(self, processed_experience: Dict[str, Any],
                                  frame_selection: BMDFrameSelection) -> Dict[str, Any]:
        """
        Fuse processed experience with selected memory frames.
        
        Args:
            processed_experience: Processed environmental experience
            frame_selection: Selected memory frames
            
        Returns:
            Fused conscious state
        """
        # Fusion parameters from theoretical framework
        alpha = 0.4  # Experience weight
        beta = 0.4   # Frame weight  
        gamma = 0.2  # Interaction weight
        
        # Extract experience information
        experience_info = processed_experience["total_information"]
        environmental_context = processed_experience["processing_context"]
        
        # Extract frame information
        selected_frames = [self.frame_store[frame_id] for frame_id in frame_selection.selected_frames]
        frame_info = np.mean(frame_selection.frame_relevance_scores) if frame_selection.frame_relevance_scores else 0.0
        
        # Calculate interaction term (contextual modulation)
        interaction_term = experience_info * frame_info * environmental_context["uniqueness"]
        
        # Fuse components
        conscious_state_intensity = alpha * experience_info + beta * frame_info + gamma * interaction_term
        
        # Generate conscious content from fusion
        conscious_content = self._generate_conscious_content(
            processed_experience["content_info"],
            selected_frames,
            conscious_state_intensity
        )
        
        return {
            "conscious_content": conscious_content,
            "state_intensity": conscious_state_intensity,
            "experience_contribution": alpha * experience_info,
            "frame_contribution": beta * frame_info,
            "interaction_contribution": gamma * interaction_term,
            "selected_frame_ids": frame_selection.selected_frames,
            "fusion_quality": frame_selection.selection_confidence
        }
    
    def _generate_conscious_content(self, original_content: str, 
                                  selected_frames: List[MemoryFrame],
                                  intensity: float) -> str:
        """Generate conscious content from experience-frame fusion."""
        content = f"BMD conscious processing of '{original_content}':\n\n"
        
        if not selected_frames:
            content += "Minimal frame activation - basic conscious awareness with limited interpretation.\n"
            return content
        
        content += f"Frame selection through BMD loaded dice mechanism activated {len(selected_frames)} cognitive frameworks:\n\n"
        
        for frame in selected_frames:
            content += f"• {frame.frame_type.value.replace('_', ' ').title()}:\n"
            if frame.knowledge_component.get("correlations"):
                content += f"  - Correlations: {', '.join(frame.knowledge_component['correlations'])}\n"
            if frame.truth_component.get("collective_agreements"):
                content += f"  - Truth validation: {', '.join(frame.truth_component['collective_agreements'])}\n"
            content += f"  - Relevance: {frame.relevance_score:.2f}\n\n"
        
        content += f"Conscious state fusion achieved intensity {intensity:.3f} through:\n"
        content += "- Experience processing with environmental context integration\n"
        content += "- Biased frame selection (loaded dice mechanism)\n"
        content += "- Experience-frame-interaction fusion\n"
        content += "- Collective truth validation\n\n"
        
        content += "This demonstrates consciousness as predetermined frame selection rather than meaning extraction."
        
        return content
    
    def perform_sanity_check(self, conscious_state: Dict[str, Any],
                           belief_systems: Optional[Dict[str, Any]] = None) -> SanityCheckResult:
        """
        Perform sanity check on conscious state through correspondence validation.
        
        Args:
            conscious_state: Fused conscious state
            belief_systems: Optional belief system context
            
        Returns:
            Sanity check result
        """
        if belief_systems is None:
            belief_systems = {
                "internal_beliefs": {"consistency": 0.8, "coherence": 0.7},
                "external_beliefs": {"social_acceptance": 0.75, "cultural_fit": 0.8}
            }
        
        # Calculate correspondence between conscious state and belief systems
        state_intensity = conscious_state["state_intensity"]
        fusion_quality = conscious_state["fusion_quality"]
        
        # Internal belief correspondence
        internal_correspondence = state_intensity * belief_systems["internal_beliefs"]["consistency"]
        
        # External belief correspondence  
        external_correspondence = fusion_quality * belief_systems["external_beliefs"]["social_acceptance"]
        
        # Overall correspondence score
        correspondence_score = (internal_correspondence + external_correspondence) / 2.0
        
        # Belief alignment assessment
        belief_alignment = min(1.0, correspondence_score * 1.2)
        
        # Collective validation (truth system validation)
        collective_validation = belief_systems["external_beliefs"]["cultural_fit"]
        
        # Functional threshold check
        functional_threshold = 0.6
        threshold_met = correspondence_score >= functional_threshold
        
        # Sanity status determination
        if threshold_met:
            sanity_status = "functional"
        else:
            sanity_status = "dysfunction"
        
        return SanityCheckResult(
            correspondence_score=correspondence_score,
            belief_alignment=belief_alignment,
            collective_validation=collective_validation,
            functional_threshold_met=threshold_met,
            sanity_status=sanity_status
        )
    
    def process_full_bmd_cycle(self, environmental_input: EnvironmentalState,
                              query_content: str) -> Dict[str, Any]:
        """
        Execute complete BMD processing cycle.
        
        Args:
            environmental_input: Environmental state
            query_content: Content to process
            
        Returns:
            Complete BMD processing result
        """
        start_time = time.time()
        
        # Phase 1: Experience processing
        processed_experience = self.process_experience(environmental_input, query_content)
        
        # Phase 2: Frame selection
        frame_selection = self.select_frames(processed_experience)
        
        # Phase 3: Experience-frame fusion
        conscious_state = self.fuse_experience_and_frames(processed_experience, frame_selection)
        
        # Phase 4: Sanity checking
        sanity_result = self.perform_sanity_check(conscious_state)
        
        processing_time = time.time() - start_time
        
        return {
            "processed_experience": processed_experience,
            "frame_selection": frame_selection,
            "conscious_state": conscious_state,
            "sanity_check": sanity_result,
            "processing_time": processing_time,
            "bmd_efficiency": frame_selection.processing_efficiency,
            "final_status": sanity_result.sanity_status,
            "correspondence_achieved": sanity_result.correspondence_score
        }
    
    def get_selection_statistics(self) -> Dict[str, Any]:
        """Get statistics on BMD frame selection performance."""
        if not self.selection_history:
            return {}
        
        confidences = [selection.selection_confidence for selection in self.selection_history]
        efficiencies = [selection.processing_efficiency for selection in self.selection_history]
        frame_counts = [len(selection.selected_frames) for selection in self.selection_history]
        
        return {
            "total_selections": len(self.selection_history),
            "average_confidence": np.mean(confidences),
            "average_efficiency": np.mean(efficiencies),
            "average_frames_per_selection": np.mean(frame_counts),
            "max_confidence": max(confidences) if confidences else 0.0,
            "loaded_dice_effectiveness": np.mean(confidences) * np.mean(efficiencies)
        }


class FrameSelector:
    """
    Advanced frame selection engine with multiple selection strategies.
    
    Implements various frame selection algorithms for different processing contexts.
    """
    
    def __init__(self):
        """Initialize frame selector."""
        self.selection_strategies = {
            "relevance_based": self._relevance_based_selection,
            "recency_weighted": self._recency_weighted_selection,
            "emotion_guided": self._emotion_guided_selection,
            "belief_aligned": self._belief_aligned_selection,
            "counterfactual_rich": self._counterfactual_rich_selection
        }
    
    def select_optimal_frames(self, available_frames: Dict[str, MemoryFrame],
                            context: Dict[str, Any],
                            strategy: str = "relevance_based",
                            max_frames: int = 5) -> List[str]:
        """
        Select optimal frames using specified strategy.
        
        Args:
            available_frames: Available memory frames
            context: Selection context
            strategy: Selection strategy to use
            max_frames: Maximum frames to select
            
        Returns:
            List of selected frame IDs
        """
        if strategy not in self.selection_strategies:
            strategy = "relevance_based"
        
        selection_function = self.selection_strategies[strategy]
        selected_frames = selection_function(available_frames, context, max_frames)
        
        return selected_frames
    
    def _relevance_based_selection(self, frames: Dict[str, MemoryFrame],
                                 context: Dict[str, Any],
                                 max_frames: int) -> List[str]:
        """Select frames based on relevance scores."""
        frame_scores = [(frame_id, frame.relevance_score) for frame_id, frame in frames.items()]
        frame_scores.sort(key=lambda x: x[1], reverse=True)
        return [frame_id for frame_id, _ in frame_scores[:max_frames]]
    
    def _recency_weighted_selection(self, frames: Dict[str, MemoryFrame],
                                  context: Dict[str, Any],
                                  max_frames: int) -> List[str]:
        """Select frames weighted by recency and relevance."""
        frame_scores = []
        for frame_id, frame in frames.items():
            score = 0.7 * frame.relevance_score + 0.3 * frame.recency_score
            frame_scores.append((frame_id, score))
        
        frame_scores.sort(key=lambda x: x[1], reverse=True)
        return [frame_id for frame_id, _ in frame_scores[:max_frames]]
    
    def _emotion_guided_selection(self, frames: Dict[str, MemoryFrame],
                                context: Dict[str, Any],
                                max_frames: int) -> List[str]:
        """Select frames guided by emotional weighting."""
        frame_scores = []
        for frame_id, frame in frames.items():
            score = 0.5 * frame.relevance_score + 0.5 * frame.emotional_weight
            frame_scores.append((frame_id, score))
        
        frame_scores.sort(key=lambda x: x[1], reverse=True)
        return [frame_id for frame_id, _ in frame_scores[:max_frames]]
    
    def _belief_aligned_selection(self, frames: Dict[str, MemoryFrame],
                                context: Dict[str, Any],
                                max_frames: int) -> List[str]:
        """Select frames aligned with belief systems."""
        frame_scores = []
        for frame_id, frame in frames.items():
            score = 0.6 * frame.belief_compatibility + 0.4 * frame.relevance_score
            frame_scores.append((frame_id, score))
        
        frame_scores.sort(key=lambda x: x[1], reverse=True)
        return [frame_id for frame_id, _ in frame_scores[:max_frames]]
    
    def _counterfactual_rich_selection(self, frames: Dict[str, MemoryFrame],
                                     context: Dict[str, Any],
                                     max_frames: int) -> List[str]:
        """Select frames rich in counterfactual content."""
        frame_scores = []
        for frame_id, frame in frames.items():
            score = 0.6 * frame.counterfactual_richness + 0.4 * frame.relevance_score
            frame_scores.append((frame_id, score))
        
        frame_scores.sort(key=lambda x: x[1], reverse=True)
        return [frame_id for frame_id, _ in frame_scores[:max_frames]]
