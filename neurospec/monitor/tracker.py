"""Thread-safe feature activation tracker.

Maintains a sliding window of feature activation values for real-time
anomaly detection and dashboard display.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ActivationSample:
    """A single timestamped activation reading."""

    timestamp: float
    feature_id: int
    value: float
    token: str = ""


@dataclass
class FeatureStats:
    """Running statistics for a tracked feature."""

    feature_id: int
    label: str = ""
    count: int = 0
    mean: float = 0.0
    variance: float = 0.0
    min_value: float = float("inf")
    max_value: float = float("-inf")
    last_value: float = 0.0

    @property
    def std_dev(self) -> float:
        """Standard deviation of tracked values."""
        return self.variance**0.5 if self.variance > 0 else 0.0


class FeatureTracker:
    """Thread-safe tracker for feature activation history.

    Maintains per-feature sliding windows and running statistics
    for use by the anomaly detector and dashboard.
    """

    def __init__(self, window_size: int = 100) -> None:
        self._window_size = window_size
        self._lock = threading.Lock()
        self._history: dict[int, deque[ActivationSample]] = {}
        self._stats: dict[int, FeatureStats] = {}
        self._labels: dict[int, str] = {}

    def set_label(self, feature_id: int, label: str) -> None:
        """Set a human-readable label for a feature."""
        with self._lock:
            self._labels[feature_id] = label
            if feature_id in self._stats:
                self._stats[feature_id].label = label

    def record(self, feature_id: int, value: float, token: str = "") -> None:
        """Record a new activation value for a feature.

        Args:
            feature_id: SAE feature index.
            value: Activation magnitude.
            token: Optional token string that triggered this activation.
        """
        sample = ActivationSample(
            timestamp=time.time(),
            feature_id=feature_id,
            value=value,
            token=token,
        )

        with self._lock:
            # Initialize if new feature
            if feature_id not in self._history:
                self._history[feature_id] = deque(maxlen=self._window_size)
                self._stats[feature_id] = FeatureStats(
                    feature_id=feature_id,
                    label=self._labels.get(feature_id, f"feature_{feature_id}"),
                )

            self._history[feature_id].append(sample)
            self._update_stats(feature_id, value)

    def record_batch(self, activations: dict[int, float]) -> None:
        """Record multiple feature activations at once."""
        for feat_id, value in activations.items():
            self.record(feat_id, value)

    def get_stats(self, feature_id: int) -> FeatureStats | None:
        """Get current statistics for a feature."""
        with self._lock:
            return self._stats.get(feature_id)

    def get_all_stats(self) -> dict[int, FeatureStats]:
        """Get statistics for all tracked features."""
        with self._lock:
            return dict(self._stats)

    def get_history(self, feature_id: int) -> list[ActivationSample]:
        """Get activation history for a feature."""
        with self._lock:
            if feature_id not in self._history:
                return []
            return list(self._history[feature_id])

    def get_recent(self, feature_id: int, n: int = 10) -> list[ActivationSample]:
        """Get the N most recent samples for a feature."""
        with self._lock:
            if feature_id not in self._history:
                return []
            history = self._history[feature_id]
            return list(history)[-n:]

    def clear(self) -> None:
        """Clear all tracked data."""
        with self._lock:
            self._history.clear()
            self._stats.clear()

    def _update_stats(self, feature_id: int, value: float) -> None:
        """Update running statistics using Welford's online algorithm."""
        stats = self._stats[feature_id]
        stats.count += 1
        stats.last_value = value

        if value < stats.min_value:
            stats.min_value = value
        if value > stats.max_value:
            stats.max_value = value

        # Welford's algorithm for online mean and variance
        delta = value - stats.mean
        stats.mean += delta / stats.count
        delta2 = value - stats.mean
        stats.variance += (delta * delta2 - stats.variance) / stats.count
