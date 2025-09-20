"""
Core data types and structures for the Graffiti research framework.

This module defines the fundamental data structures used throughout the system,
implementing theoretical concepts from meaning impossibility theory, S-entropy
navigation, and environmental consciousness integration.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union, Tuple
from enum import Enum
import numpy as np
from datetime import datetime
import uuid


class ProcessingMode(Enum):
    """Processing modes for different theoretical frameworks."""
    TRADITIONAL = "traditional"
    S_ENTROPY_NAVIGATION = "s_entropy_navigation" 
    CHESS_WITH_MIRACLES = "chess_with_miracles"
    ENVIRONMENTAL_CONSCIOUSNESS = "environmental_consciousness"
    FULL_REVOLUTIONARY = "full_revolutionary"


class ComponentStatus(Enum):
    """Status of system components."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class ResponseType(Enum):
    """Expected response types for queries."""
    EXPLANATION = "explanation"
    ANALYSIS = "analysis"
    SYNTHESIS = "synthesis"
    NAVIGATION = "navigation"
    OPTIMIZATION = "optimization"


class Urgency(Enum):
    """Query urgency levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SEntropyCoordinates:
    """
    S-entropy coordinate representation in tri-dimensional space.
    
    Represents position in knowledge-time-entropy dimensional space
    for observer-process navigation.
    """
    knowledge: float = 0.0
    time: float = 0.0
    entropy: float = 0.0
    
    def __post_init__(self):
        """Validate coordinate bounds."""
        # Ensure coordinates are finite
        for coord in [self.knowledge, self.time, self.entropy]:
            if not np.isfinite(coord):
                raise ValueError(f"S-entropy coordinate must be finite, got {coord}")
    
    def distance_to(self, other: 'SEntropyCoordinates') -> float:
        """Calculate S-distance to another coordinate."""
        return np.sqrt(
            (self.knowledge - other.knowledge) ** 2 +
            (self.time - other.time) ** 2 +
            (self.entropy - other.entropy) ** 2
        )
    
    def as_vector(self) -> np.ndarray:
        """Return as numpy array."""
        return np.array([self.knowledge, self.time, self.entropy])


@dataclass
class BiometricDimension:
    """Biometric environmental dimension measurements."""
    physiological_arousal: float = 0.5
    cognitive_load: float = 0.5
    attention_state: float = 0.5
    stress_level: float = 0.5


@dataclass
class SpatialDimension:
    """Spatial environmental dimension measurements."""
    location_x: float = 0.0
    location_y: float = 0.0
    location_z: float = 0.0
    orientation: float = 0.0


@dataclass
class EnvironmentalState:
    """
    Twelve-dimensional environmental state representation.
    
    Captures complete environmental context for consciousness integration.
    """
    timestamp: datetime = field(default_factory=datetime.now)
    biometric: BiometricDimension = field(default_factory=BiometricDimension)
    spatial: SpatialDimension = field(default_factory=SpatialDimension)
    
    # Additional environmental dimensions
    atmospheric_pressure: float = 1013.25  # hPa
    temperature: float = 20.0  # Celsius
    humidity: float = 50.0  # Percentage
    light_level: float = 500.0  # Lux
    sound_level: float = 40.0  # dB
    electromagnetic_field: float = 0.0  # μT
    
    # Cosmic and quantum dimensions
    cosmic_background: float = 0.5
    quantum_coherence: float = 0.5
    temporal_flow: float = 1.0
    
    def calculate_uniqueness(self) -> float:
        """
        Calculate environmental uniqueness score.
        
        Returns a value representing how unique this environmental
        state is compared to baseline conditions.
        """
        baseline = EnvironmentalState()
        
        # Calculate differences from baseline
        differences = []
        differences.append(abs(self.biometric.physiological_arousal - baseline.biometric.physiological_arousal))
        differences.append(abs(self.biometric.cognitive_load - baseline.biometric.cognitive_load))
        differences.append(abs(self.atmospheric_pressure - baseline.atmospheric_pressure) / 1000.0)
        differences.append(abs(self.temperature - baseline.temperature) / 50.0)
        differences.append(abs(self.humidity - baseline.humidity) / 100.0)
        differences.append(abs(self.light_level - baseline.light_level) / 1000.0)
        differences.append(abs(self.sound_level - baseline.sound_level) / 100.0)
        differences.append(abs(self.cosmic_background - baseline.cosmic_background))
        differences.append(abs(self.quantum_coherence - baseline.quantum_coherence))
        differences.append(abs(self.temporal_flow - baseline.temporal_flow))
        
        return np.mean(differences)


@dataclass  
class UserContext:
    """User context for personalized processing."""
    expertise_level: str = "intermediate"
    preferred_style: str = "balanced"
    context_preferences: Dict[str, float] = field(default_factory=dict)
    historical_queries: List[str] = field(default_factory=list)
    
    @classmethod
    def default(cls) -> 'UserContext':
        """Create default user context."""
        return cls(
            context_preferences={
                "technical_depth": 0.6,
                "creative_exploration": 0.7,
                "precision_required": 0.8,
            }
        )


@dataclass
class QueryId:
    """Unique query identifier."""
    value: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    @classmethod
    def new(cls) -> 'QueryId':
        """Generate new query ID."""
        return cls()


@dataclass
class Query:
    """Query representation with environmental context."""
    id: QueryId = field(default_factory=QueryId.new)
    content: str = ""
    environmental_context: EnvironmentalState = field(default_factory=EnvironmentalState)
    user_context: UserContext = field(default_factory=UserContext.default)
    expected_response_type: ResponseType = ResponseType.EXPLANATION
    urgency: Urgency = Urgency.NORMAL
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_coordinates(self) -> SEntropyCoordinates:
        """Convert query to S-entropy coordinates."""
        # Simple coordinate mapping based on content characteristics
        knowledge_coord = len(self.content) / 1000.0  # Normalize by length
        time_coord = self.urgency.value == "high" and 0.8 or 0.5
        entropy_coord = self.environmental_context.calculate_uniqueness()
        
        return SEntropyCoordinates(
            knowledge=knowledge_coord,
            time=time_coord,
            entropy=entropy_coord
        )


@dataclass
class SemanticCoordinates:
    """Four-dimensional semantic coordinate representation."""
    technical_emotional: float = 0.0  # Technical(+) vs Emotional(-)
    action_descriptive: float = 0.0   # Action(+) vs Descriptive(-)
    abstract_concrete: float = 0.0    # Abstract(+) vs Concrete(-)
    positive_negative: float = 0.0    # Positive(+) vs Negative(-)
    
    def as_vector(self) -> np.ndarray:
        """Return as numpy array."""
        return np.array([
            self.technical_emotional,
            self.action_descriptive, 
            self.abstract_concrete,
            self.positive_negative
        ])
    
    def distance_to(self, other: 'SemanticCoordinates') -> float:
        """Calculate Euclidean distance to another coordinate."""
        return np.linalg.norm(self.as_vector() - other.as_vector())


@dataclass
class MolecularProcessingInfo:
    """Information about molecular processing results."""
    consensus_level: float = 0.0
    processing_time: float = 0.0
    molecular_count: int = 0
    equilibrium_reached: bool = False


@dataclass
class PerformanceMetrics:
    """Performance measurement results."""
    processing_time: float = 0.0
    memory_usage: float = 0.0  # MB
    accuracy_score: float = 0.0
    speedup_factor: float = 1.0
    compression_ratio: float = 1.0
    s_distance_minimization: float = 0.0
    
    def __str__(self) -> str:
        return f"Performance(time={self.processing_time:.3f}s, speedup={self.speedup_factor:.1f}×, accuracy={self.accuracy_score:.3f})"


@dataclass
class ValidationResult:
    """Validation test result."""
    test_name: str = ""
    passed: bool = False
    score: float = 0.0
    expected_improvement: float = 0.0
    actual_improvement: float = 0.0
    significance_level: float = 0.0
    effect_size: float = 0.0
    
    def improvement_ratio(self) -> float:
        """Calculate ratio of actual to expected improvement."""
        if self.expected_improvement == 0:
            return 1.0
        return self.actual_improvement / self.expected_improvement


@dataclass
class QueryResult:
    """Result from query processing."""
    query_id: QueryId = field(default_factory=QueryId.new)
    content: str = ""
    confidence: float = 0.0
    environmental_confidence: float = 0.0
    s_distance: float = 0.0
    processing_time: float = 0.0
    coordinates: SEntropyCoordinates = field(default_factory=SEntropyCoordinates)
    semantic_coordinates: SemanticCoordinates = field(default_factory=SemanticCoordinates)
    molecular_info: MolecularProcessingInfo = field(default_factory=MolecularProcessingInfo)
    performance_metrics: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        return f"QueryResult(confidence={self.confidence:.3f}, s_distance={self.s_distance:.6f})"


@dataclass
class CrossDomainTransferResult:
    """Result from cross-domain pattern transfer."""
    source_domain: str = ""
    target_domain: str = ""
    transfer_efficiency: float = 0.0
    improvement_factor: float = 1.0
    patterns_transferred: int = 0
    validation_score: float = 0.0


@dataclass  
class ChessMiracleResult:
    """Result from Chess with Miracles processing."""
    original_query: str = ""
    enhanced_query: str = ""
    miracle_score: float = 0.0
    weak_position_strength: float = 0.0
    victory_conditions: List[str] = field(default_factory=list)
    processing_pathway: List[str] = field(default_factory=list)


@dataclass
class BMDFrameSelection:
    """Biological Maxwell Demon frame selection result."""
    selected_frames: List[str] = field(default_factory=list)
    frame_relevance_scores: List[float] = field(default_factory=list)
    selection_confidence: float = 0.0
    processing_efficiency: float = 0.0


# Type aliases for complex types
CoordinateVector = Tuple[float, float, float]
SemanticVector = Tuple[float, float, float, float]
EnvironmentalMeasurement = Dict[str, float]
ProcessingPipeline = List[str]
