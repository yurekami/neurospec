"""Linear probe training for NeuroSpec RLFR.

Trains lightweight linear classifiers on SAE features to detect
specific behavioral properties. These probes serve as reward signals
during RLFR training.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    optim = None  # type: ignore[assignment]


@dataclass
class ProbeResult:
    """Result of running a probe on activations."""

    name: str
    score: float
    activated: bool
    threshold: float


class LinearProbe:
    """A simple linear probe trained on SAE features.

    Maps a subset of SAE feature activations to a scalar score
    indicating the presence/absence of a behavioral property.
    """

    def __init__(
        self,
        name: str,
        feature_ids: list[int],
        threshold: float = 0.5,
    ) -> None:
        if torch is None:
            raise ImportError(
                "PyTorch is required for probes. Install with: pip install neurospec[forge]"
            )

        self._name = name
        self._feature_ids = feature_ids
        self._threshold = threshold
        self._weights = nn.Linear(len(feature_ids), 1)
        self._trained = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_trained(self) -> bool:
        return self._trained

    def forward(self, features: Any) -> ProbeResult:
        """Run the probe on SAE features.

        Args:
            features: SAE feature tensor of shape (..., n_features).

        Returns:
            ProbeResult with score and activation status.
        """
        # Extract relevant features
        selected = features[..., self._feature_ids]

        # Average across sequence dimension if present
        if selected.dim() > 2:
            selected = selected.mean(dim=-2)
        if selected.dim() > 1:
            selected = selected.mean(dim=0)

        with torch.no_grad():
            score = torch.sigmoid(self._weights(selected)).item()

        return ProbeResult(
            name=self._name,
            score=score,
            activated=score > self._threshold,
            threshold=self._threshold,
        )

    def train_probe(
        self,
        positive_features: list[Any],
        negative_features: list[Any],
        epochs: int = 100,
        lr: float = 1e-3,
    ) -> float:
        """Train the probe on labeled examples.

        Args:
            positive_features: Feature tensors where property IS present.
            negative_features: Feature tensors where property is NOT present.
            epochs: Training epochs.
            lr: Learning rate.

        Returns:
            Final training accuracy.
        """
        if not positive_features or not negative_features:
            logger.warning("Probe '%s': need both positive and negative examples", self._name)
            return 0.0

        # Build training data
        pos_data = [
            f[..., self._feature_ids].mean(dim=tuple(range(f.dim() - 1))) for f in positive_features
        ]
        neg_data = [
            f[..., self._feature_ids].mean(dim=tuple(range(f.dim() - 1))) for f in negative_features
        ]

        X = torch.stack(pos_data + neg_data)
        y = torch.cat([torch.ones(len(pos_data)), torch.zeros(len(neg_data))]).unsqueeze(1)

        optimizer = optim.Adam(self._weights.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()

        for epoch in range(epochs):
            optimizer.zero_grad()
            logits = self._weights(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        # Compute accuracy
        with torch.no_grad():
            preds = (torch.sigmoid(self._weights(X)) > 0.5).float()
            accuracy = (preds == y).float().mean().item()

        self._trained = True
        logger.info("Probe '%s' trained: accuracy=%.3f", self._name, accuracy)
        return accuracy

    def state_dict(self) -> dict[str, Any]:
        """Return probe state for serialization."""
        return {
            "name": self._name,
            "feature_ids": self._feature_ids,
            "threshold": self._threshold,
            "weights": self._weights.state_dict(),
            "trained": self._trained,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Load probe state from serialization."""
        self._weights.load_state_dict(state["weights"])
        self._trained = state.get("trained", True)
