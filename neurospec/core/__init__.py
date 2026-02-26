"""NeuroSpec core — shared types, enums, and configuration.

Import the most commonly used types from here for convenience:

    from neurospec.core import Feature, FeatureCatalog, SteeringVector
"""

from neurospec.core.config import NeuroSpecConfig, get_config, set_config
from neurospec.core.types import (
    ActionKind,
    AlertConfig,
    Feature,
    FeatureCatalog,
    MonitorConfig,
    ProbeConfig,
    RLFRConfig,
    Severity,
    SpecCompilationResult,
    SpecMeta,
    SteeringDirection,
    SteeringVector,
    ValidationError,
)

__all__ = [
    "ActionKind",
    "AlertConfig",
    "Feature",
    "FeatureCatalog",
    "MonitorConfig",
    "NeuroSpecConfig",
    "ProbeConfig",
    "RLFRConfig",
    "Severity",
    "SpecCompilationResult",
    "SpecMeta",
    "SteeringDirection",
    "SteeringVector",
    "ValidationError",
    "get_config",
    "set_config",
]
