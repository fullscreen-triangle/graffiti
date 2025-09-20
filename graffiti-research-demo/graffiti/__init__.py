"""
Graffiti Research Demo Package

Revolutionary search engine architecture through environmental consciousness integration
and S-entropy coordinate navigation.

This package implements cutting-edge theoretical frameworks including:
- Meaning impossibility theory and 95%/5% information architecture
- Environmental consciousness integration with twelve-dimensional measurement
- S-entropy coordinate navigation with observer-process integration
- Chess with Miracles processing for weak query enhancement
- Empty dictionary gas molecular synthesis
"""

from graffiti.core.types import (
    # Core data structures
    SEntropyCoordinates,
    EnvironmentalState,
    QueryResult,
    ProcessingMode,
    
    # Performance metrics
    PerformanceMetrics,
    ValidationResult,
)

from graffiti.core.s_entropy import (
    SEntropyCalculator,
    SDistanceMetric,
)

from graffiti.integration.unified_engine import (
    GraffitiEngine,
)

# Package metadata
__version__ = "0.1.0"
__author__ = "Research Team"
__email__ = "research@graffiti-search.org"

# Core exports for easy access
__all__ = [
    # Core types
    "SEntropyCoordinates",
    "EnvironmentalState", 
    "QueryResult",
    "ProcessingMode",
    "PerformanceMetrics",
    "ValidationResult",
    
    # Core functionality
    "SEntropyCalculator",
    "SDistanceMetric",
    "GraffitiEngine",
    
    # Package metadata
    "__version__",
    "__author__",
    "__email__",
]

# Ensure core components are available
try:
    import numpy as np
    import scipy
    import pandas as pd
    _DEPENDENCIES_AVAILABLE = True
except ImportError:
    _DEPENDENCIES_AVAILABLE = False

if not _DEPENDENCIES_AVAILABLE:
    import warnings
    warnings.warn(
        "Core dependencies (numpy, scipy, pandas) not available. "
        "Some functionality may be limited. Install with: pip install graffiti-research-demo",
        ImportWarning
    )
