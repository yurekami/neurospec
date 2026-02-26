"""NeuroSpec compiler — translates .ns specs into feature-level interventions.

Usage:
    from neurospec.compiler import NeuroSpecCompiler
    from neurospec.compiler.serializer import serialize_to_json

    compiler = NeuroSpecCompiler(catalog=catalog)
    result = compiler.compile(spec_decl)
    json_str = serialize_to_json(result)
"""

from neurospec.compiler.compiler import CompilationError, NeuroSpecCompiler
from neurospec.compiler.resolver import FeatureResolver, ResolutionResult
from neurospec.compiler.serializer import (
    deserialize_from_dict,
    deserialize_from_json,
    serialize_to_dict,
    serialize_to_json,
)

__all__ = [
    "CompilationError",
    "FeatureResolver",
    "NeuroSpecCompiler",
    "ResolutionResult",
    "deserialize_from_dict",
    "deserialize_from_json",
    "serialize_to_dict",
    "serialize_to_json",
]
