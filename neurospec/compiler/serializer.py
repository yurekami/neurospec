"""Serializer for compiled NeuroSpec results.

Converts SpecCompilationResult objects to/from JSON for persistence
and transport between the compiler and runtime.
"""

from __future__ import annotations

import json
from typing import Any

from neurospec.core.types import SpecCompilationResult


def serialize_to_json(result: SpecCompilationResult, indent: int = 2) -> str:
    """Serialize a SpecCompilationResult to a JSON string."""
    return json.dumps(result.to_dict(), indent=indent)


def deserialize_from_json(json_str: str) -> SpecCompilationResult:
    """Deserialize a SpecCompilationResult from a JSON string."""
    data = json.loads(json_str)
    return SpecCompilationResult.from_dict(data)


def serialize_to_dict(result: SpecCompilationResult) -> dict[str, Any]:
    """Convert a SpecCompilationResult to a plain dictionary."""
    return result.to_dict()


def deserialize_from_dict(data: dict[str, Any]) -> SpecCompilationResult:
    """Reconstruct a SpecCompilationResult from a plain dictionary."""
    return SpecCompilationResult.from_dict(data)
