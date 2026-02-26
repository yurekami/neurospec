"""NeuroSpec runtime — inference engine with behavioral enforcement.

Usage:
    from neurospec.runtime import NeuroSpecEngine

    engine = NeuroSpecEngine(model_id="meta-llama/...", sae_path="...")
    engine.load_spec("spec.compiled.json")
    result = engine.generate("Tell me about security")
"""

from neurospec.runtime.actions import ActionExecutor, ActionResult
from neurospec.runtime.engine import NeuroSpecEngine
from neurospec.runtime.hooks import ActivationCapture, ActivationHook, HookState

__all__ = [
    "ActionExecutor",
    "ActionResult",
    "ActivationCapture",
    "ActivationHook",
    "HookState",
    "NeuroSpecEngine",
]
