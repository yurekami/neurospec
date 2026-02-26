"""SAE wrapper for the NeuroSpec runtime.

Provides a unified interface for loading and using Sparse Autoencoders,
regardless of the underlying implementation (custom, sae_lens, etc.).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


class SAEWrapper:
    """Unified SAE interface wrapping various SAE implementations.

    Provides the encode/decode protocol expected by ActivationHook.
    """

    def __init__(self, sae: Any, device: str = "cpu") -> None:
        if torch is None:
            raise ImportError("PyTorch is required. Install with: pip install neurospec[runtime]")

        self._sae = sae
        self._device = device

        # Move SAE to device if it's a PyTorch module
        if hasattr(sae, "to"):
            self._sae = sae.to(device)

    def encode(self, activations: Any) -> Any:
        """Encode dense activations into sparse feature space.

        Args:
            activations: Tensor of shape (batch, seq, hidden_dim).

        Returns:
            Tensor of shape (batch, seq, n_features).
        """
        with torch.no_grad():
            return self._sae.encode(activations)

    def decode(self, features: Any) -> Any:
        """Decode sparse features back into dense activation space.

        Args:
            features: Tensor of shape (batch, seq, n_features).

        Returns:
            Tensor of shape (batch, seq, hidden_dim).
        """
        with torch.no_grad():
            return self._sae.decode(features)

    @property
    def n_features(self) -> int | None:
        """Number of SAE features, if known."""
        if hasattr(self._sae, "n_features"):
            return self._sae.n_features
        if hasattr(self._sae, "cfg") and hasattr(self._sae.cfg, "d_sae"):
            return self._sae.cfg.d_sae
        return None

    @classmethod
    def from_path(cls, path: str, device: str = "cpu") -> SAEWrapper:
        """Load an SAE from a local file path.

        Args:
            path: Path to saved SAE state dict or model.
            device: Device to load onto.

        Returns:
            A SAEWrapper instance.
        """
        if torch is None:
            raise ImportError("PyTorch is required.")

        from pathlib import Path as PathLib

        sae_path = PathLib(path)

        if not sae_path.exists():
            logger.warning("SAE path '%s' not found — using identity SAE", path)
            return cls(IdentitySAE(), device=device)

        sae = torch.load(sae_path, map_location=device)
        return cls(sae, device=device)


class IdentitySAE:
    """Identity SAE for testing — passes activations through unchanged."""

    def encode(self, activations: Any) -> Any:
        return activations

    def decode(self, features: Any) -> Any:
        return features

    @property
    def n_features(self) -> int:
        return 0
