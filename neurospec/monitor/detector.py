"""Anomaly detection for feature activations.

Uses z-score analysis and trend detection to identify when feature
activations deviate significantly from their baseline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from neurospec.monitor.tracker import FeatureTracker

logger = logging.getLogger(__name__)


@dataclass
class Anomaly:
    """A detected anomaly in feature activation."""

    feature_id: int
    label: str
    current_value: float
    mean: float
    std_dev: float
    z_score: float
    anomaly_type: str  # "spike", "drift", "sustained"


class AnomalyDetector:
    """Detect anomalous feature activations using statistical methods.

    Monitors feature tracker statistics and flags activations that
    deviate significantly from their running baseline.
    """

    def __init__(
        self,
        tracker: FeatureTracker,
        z_score_threshold: float = 3.0,
        min_samples: int = 10,
    ) -> None:
        self._tracker = tracker
        self._z_score_threshold = z_score_threshold
        self._min_samples = min_samples

    def check_all(self) -> list[Anomaly]:
        """Check all tracked features for anomalies.

        Returns a list of detected anomalies (may be empty).
        """
        anomalies: list[Anomaly] = []
        all_stats = self._tracker.get_all_stats()

        for feature_id, stats in all_stats.items():
            if stats.count < self._min_samples:
                continue

            anomaly = self._check_feature(feature_id, stats)
            if anomaly is not None:
                anomalies.append(anomaly)

        return anomalies

    def check_feature(self, feature_id: int) -> Anomaly | None:
        """Check a single feature for anomalies."""
        stats = self._tracker.get_stats(feature_id)
        if stats is None or stats.count < self._min_samples:
            return None
        return self._check_feature(feature_id, stats)

    def _check_feature(self, feature_id: int, stats: Any) -> Anomaly | None:
        """Run anomaly detection on a single feature's statistics."""
        if stats.std_dev == 0:
            return None

        z_score = abs(stats.last_value - stats.mean) / stats.std_dev

        if z_score < self._z_score_threshold:
            return None

        # Classify anomaly type
        anomaly_type = self._classify_anomaly(feature_id, stats, z_score)

        anomaly = Anomaly(
            feature_id=feature_id,
            label=stats.label,
            current_value=stats.last_value,
            mean=stats.mean,
            std_dev=stats.std_dev,
            z_score=z_score,
            anomaly_type=anomaly_type,
        )

        logger.warning(
            "Anomaly detected: feature %d (%s) z=%.2f type=%s value=%.3f mean=%.3f",
            feature_id,
            stats.label,
            z_score,
            anomaly_type,
            stats.last_value,
            stats.mean,
        )

        return anomaly

    def _classify_anomaly(self, feature_id: int, stats: Any, z_score: float) -> str:
        """Classify the type of anomaly based on recent history."""
        recent = self._tracker.get_recent(feature_id, n=5)

        if len(recent) < 3:
            return "spike"

        # Check if multiple recent values are elevated (sustained anomaly)
        high_count = sum(
            1
            for s in recent
            if stats.std_dev > 0
            and abs(s.value - stats.mean) / stats.std_dev > self._z_score_threshold * 0.7
        )

        if high_count >= 3:
            return "sustained"

        # Check for trend (consistently increasing)
        values = [s.value for s in recent]
        if len(values) >= 3 and all(values[i] < values[i + 1] for i in range(len(values) - 1)):
            return "drift"

        return "spike"
