"""NeuroSpec marketplace — spec registry, versioning, and composition.

Usage:
    from neurospec.marketplace import SpecRegistry, SpecComposer

    registry = SpecRegistry()
    specs = registry.list_specs()

    composer = SpecComposer()
    merged, conflicts = composer.compose([spec_a, spec_b])
"""

from neurospec.marketplace.composer import ConflictInfo, SpecComposer
from neurospec.marketplace.registry import SpecRegistry
from neurospec.marketplace.versioning import SemVer, is_newer, latest_version

__all__ = [
    "ConflictInfo",
    "SemVer",
    "SpecComposer",
    "SpecRegistry",
    "is_newer",
    "latest_version",
]
