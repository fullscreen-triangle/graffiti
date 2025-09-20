"""
Core theoretical frameworks for the Graffiti research system.

This module implements the fundamental theoretical concepts:
- Meaning impossibility theory and 95%/5% information architecture  
- S-entropy framework for observer-process integration
- Universal problem-solving engine mathematics
- Biological Maxwell Demon operations
"""

from graffiti.core.types import (
    SEntropyCoordinates,
    EnvironmentalState,
    QueryResult,
    PerformanceMetrics,
    ValidationResult,
    ProcessingMode,
)

from graffiti.core.s_entropy import (
    SEntropyCalculator,
    SDistanceMetric,
)

from graffiti.core.meaning_impossibility import (
    MeaningImpossibilityAnalyzer,
    InformationArchitecture,
)

from graffiti.core.universal_solver import (
    UniversalProblemSolver,
    PredeterminedSolutionNavigator,
)

from graffiti.core.bmd_operations import (
    BiologicalMaxwellDemon,
    FrameSelector,
)

__all__ = [
    # Core types
    "SEntropyCoordinates",
    "EnvironmentalState", 
    "QueryResult",
    "PerformanceMetrics",
    "ValidationResult",
    "ProcessingMode",
    
    # S-entropy framework
    "SEntropyCalculator",
    "SDistanceMetric",
    
    # Meaning impossibility theory
    "MeaningImpossibilityAnalyzer",
    "InformationArchitecture",
    
    # Universal solver
    "UniversalProblemSolver", 
    "PredeterminedSolutionNavigator",
    
    # BMD operations
    "BiologicalMaxwellDemon",
    "FrameSelector",
]
