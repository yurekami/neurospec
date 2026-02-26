"""PyTorch forward hooks for activation capture and steering.

Provides the core mechanism for intercepting model activations at a target
layer, decoding through the SAE, applying steering vectors, and re-encoding
the modified activations back into the model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


@dataclass
class SteeringInstruction:
    """A single steering instruction to apply during inference."""

    feature_ids: list[int]
    multiplier: float  # positive = amplify, negative = suppress
    label: str = ""


@dataclass
class HookState:
    """Mutable state shared between hooks and the engine."""

    steering_instructions: list[SteeringInstruction] = field(default_factory=list)
    last_features: Any = None  # Last SAE-decoded features (for monitoring)
    enabled: bool = True


class ActivationHook:
    """Forward hook that captures, steers, and monitors activations.

    Registers on a target layer's forward pass to:
    1. Capture raw activations
    2. Decode through SAE into feature space
    3. Apply steering vectors (amplify/suppress features)
    4. Re-encode and replace the activations
    5. Store feature values for monitoring
    """

    def __init__(
        self,
        sae: Any,
        state: HookState,
        callbacks: list[Callable[[Any], None]] | None = None,
    ) -> None:
        if torch is None:
            raise ImportError(
                "PyTorch is required for runtime hooks. Install with: pip install neurospec[runtime]"
            )

        self._sae = sae
        self._state = state
        self._callbacks = callbacks or []
        self._handle: Any = None

    def register(self, module: Any) -> None:
        """Register this hook on a PyTorch module."""
        self._handle = module.register_forward_hook(self._hook_fn)
        logger.debug("Registered activation hook on %s", type(module).__name__)

    def remove(self) -> None:
        """Remove the hook from the module."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def _hook_fn(self, module: Any, input: Any, output: Any) -> Any:
        """The actual forward hook function."""
        if not self._state.enabled:
            return output

        # Handle tuple outputs (common in transformer layers)
        if isinstance(output, tuple):
            hidden_states = output[0]
            rest = output[1:]
        else:
            hidden_states = output
            rest = None

        # Decode through SAE: dense -> sparse features
        features = self._sae.encode(hidden_states)

        # Store for monitoring
        self._state.last_features = features.detach().clone()

        # Apply steering
        if self._state.steering_instructions:
            features = self._apply_steering(features)

        # Re-encode: sparse features -> dense
        modified = self._sae.decode(features)

        # Fire monitoring callbacks
        for callback in self._callbacks:
            try:
                callback(self._state.last_features)
            except Exception as exc:
                logger.warning("Monitor callback failed: %s", exc)

        if rest is not None:
            return (modified,) + rest
        return modified

    def _apply_steering(self, features: Any) -> Any:
        """Apply all steering instructions to the feature tensor."""
        for instruction in self._state.steering_instructions:
            for feat_id in instruction.feature_ids:
                if feat_id < features.shape[-1]:
                    features[..., feat_id] *= instruction.multiplier

        return features


class ActivationCapture:
    """Simple hook that only captures activations without modification.

    Useful for monitoring and catalog building where no steering is needed.
    """

    def __init__(self) -> None:
        self._captured: list[Any] = []
        self._handle: Any = None

    @property
    def captured(self) -> list[Any]:
        return self._captured

    def register(self, module: Any) -> None:
        """Register a capture-only hook."""
        self._handle = module.register_forward_hook(self._hook_fn)

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def clear(self) -> None:
        self._captured.clear()

    def _hook_fn(self, module: Any, input: Any, output: Any) -> None:
        """Capture the output without modifying it."""
        if isinstance(output, tuple):
            self._captured.append(output[0].detach())
        else:
            self._captured.append(output.detach())
