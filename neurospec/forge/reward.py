"""Feature-based reward function for RLFR training.

Computes reward signals from linear probes running on a FROZEN copy
of the base model. The frozen model prevents the student from learning
to hack the probes (key insight from Goodfire's RLFR paper).
"""

from __future__ import annotations

import logging
from typing import Any

from neurospec.forge.probes import LinearProbe, ProbeResult

logger = logging.getLogger(__name__)

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]


class FeatureRewardFunction:
    """Compute rewards from probe outputs on a frozen model.

    CRITICAL: Probes must run on activations from a FROZEN copy of
    the base model, NOT the model being trained. This prevents the
    student from learning to game the probes.
    """

    def __init__(
        self,
        probes: list[LinearProbe],
        frozen_model: Any = None,
        sae: Any = None,
        reward_scale: float = 1.0,
        layer: int | None = None,
    ) -> None:
        if torch is None:
            raise ImportError("PyTorch is required. Install with: pip install neurospec[forge]")

        self._probes = probes
        self._frozen_model = frozen_model
        self._sae = sae
        self._reward_scale = reward_scale
        self._layer = layer

    def compute_reward(
        self,
        input_ids: Any,
        attention_mask: Any | None = None,
    ) -> tuple[float, list[ProbeResult]]:
        """Compute the reward for a generated sequence.

        Runs input through the FROZEN model, extracts SAE features,
        and evaluates all probes.

        Args:
            input_ids: Token IDs tensor.
            attention_mask: Optional attention mask.

        Returns:
            Tuple of (total_reward, list of individual probe results).
        """
        features = self._get_frozen_features(input_ids, attention_mask)
        return self._evaluate_probes(features)

    def _get_frozen_features(
        self,
        input_ids: Any,
        attention_mask: Any | None,
    ) -> Any:
        """Get SAE features from the frozen model."""
        captured: list[Any] = []

        def hook_fn(module: Any, input: Any, output: Any) -> None:
            if isinstance(output, tuple):
                captured.append(output[0])
            else:
                captured.append(output)

        # Find target layer
        target = self._get_target_layer()
        handle = target.register_forward_hook(hook_fn)

        try:
            kwargs: dict[str, Any] = {"input_ids": input_ids}
            if attention_mask is not None:
                kwargs["attention_mask"] = attention_mask

            with torch.no_grad():
                self._frozen_model(**kwargs)

            if not captured:
                raise RuntimeError("No activations captured from frozen model")

            hidden_states = captured[0]
            return self._sae.encode(hidden_states)

        finally:
            handle.remove()

    def _evaluate_probes(self, features: Any) -> tuple[float, list[ProbeResult]]:
        """Run all probes and compute total reward."""
        results: list[ProbeResult] = []
        total_reward = 0.0

        for probe in self._probes:
            result = probe.forward(features)
            results.append(result)

            # Reward = positive probe score (higher = better behavior)
            total_reward += result.score * self._reward_scale

        # Average across probes
        if self._probes:
            total_reward /= len(self._probes)

        return total_reward, results

    def _get_target_layer(self) -> Any:
        """Get target layer from frozen model."""
        layers = self._get_model_layers()
        if self._layer is not None and self._layer < len(layers):
            return layers[self._layer]
        return layers[len(layers) // 2]

    def _get_model_layers(self) -> list[Any]:
        """Extract transformer layers from the frozen model."""
        if hasattr(self._frozen_model, "model"):
            inner = self._frozen_model.model
            if hasattr(inner, "layers"):
                return list(inner.layers)
        if hasattr(self._frozen_model, "transformer"):
            if hasattr(self._frozen_model.transformer, "h"):
                return list(self._frozen_model.transformer.h)
        raise ValueError("Could not find transformer layers in frozen model")
