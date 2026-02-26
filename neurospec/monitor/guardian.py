"""Guardian — the immune system coordinator.

Ties together the tracker, detector, and action executor to provide
continuous behavioral monitoring and automatic intervention.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from neurospec.core.types import ActionKind, MonitorConfig, SpecCompilationResult
from neurospec.monitor.detector import AnomalyDetector
from neurospec.monitor.tracker import FeatureTracker
from neurospec.runtime.actions import ActionExecutor

logger = logging.getLogger(__name__)


class Guardian:
    """Continuous monitoring and automatic intervention system.

    The Guardian watches feature activations in real-time and executes
    configured actions when thresholds or anomalies are detected.
    """

    def __init__(
        self,
        tracker: FeatureTracker,
        detector: AnomalyDetector,
        action_executor: ActionExecutor,
        monitors: list[MonitorConfig] | None = None,
    ) -> None:
        self._tracker = tracker
        self._detector = detector
        self._executor = action_executor
        self._monitors = monitors or []
        self._running = False
        self._thread: threading.Thread | None = None

    @classmethod
    def from_spec(
        cls,
        spec: SpecCompilationResult,
        tracker: FeatureTracker | None = None,
        window_size: int = 100,
        z_score_threshold: float = 3.0,
    ) -> Guardian:
        """Create a Guardian from a compiled spec.

        Args:
            spec: Compiled NeuroSpec result.
            tracker: Optional pre-existing tracker.
            window_size: Sliding window size for statistics.
            z_score_threshold: Z-score threshold for anomaly detection.

        Returns:
            A configured Guardian instance.
        """
        tracker = tracker or FeatureTracker(window_size=window_size)
        detector = AnomalyDetector(tracker, z_score_threshold=z_score_threshold)
        executor = ActionExecutor()

        return cls(
            tracker=tracker,
            detector=detector,
            action_executor=executor,
            monitors=spec.monitors,
        )

    def on_features(self, features: Any) -> None:
        """Callback for processing new feature activations.

        Called by the ActivationHook after each forward pass.

        Args:
            features: Tensor of SAE feature activations.
        """
        try:
            import torch
        except ImportError:
            return

        if features is None:
            return

        # Record activations for monitored features
        monitored_ids = set()
        for monitor in self._monitors:
            monitored_ids.update(monitor.feature_ids)

        for feat_id in monitored_ids:
            if feat_id < features.shape[-1]:
                value = float(features[..., feat_id].max())
                self._tracker.record(feat_id, value)

        # Check monitors
        self._check_monitors(features)

    def _check_monitors(self, features: Any) -> None:
        """Check all configured monitors against current feature values."""
        for monitor in self._monitors:
            for feat_id in monitor.feature_ids:
                if feat_id >= features.shape[-1]:
                    continue

                max_val = float(features[..., feat_id].max())
                if max_val > monitor.threshold:
                    logger.info(
                        "Guardian: monitor '%s' triggered (feature %d: %.3f > %.3f)",
                        monitor.name,
                        feat_id,
                        max_val,
                        monitor.threshold,
                    )
                    self._executor.execute(
                        monitor.action,
                        params=monitor.action_params,
                        context={
                            "monitor_name": monitor.name,
                            "feature_id": feat_id,
                            "value": max_val,
                            "threshold": monitor.threshold,
                        },
                    )

        # Also check for statistical anomalies
        anomalies = self._detector.check_all()
        for anomaly in anomalies:
            logger.warning(
                "Guardian: anomaly detected — %s on feature %d (%s) z=%.2f",
                anomaly.anomaly_type,
                anomaly.feature_id,
                anomaly.label,
                anomaly.z_score,
            )
            self._executor.execute(
                ActionKind.ALERT,
                params={"message": f"Anomaly: {anomaly.anomaly_type} on {anomaly.label}"},
                context={
                    "feature_id": anomaly.feature_id,
                    "z_score": anomaly.z_score,
                    "anomaly_type": anomaly.anomaly_type,
                },
            )

    @property
    def tracker(self) -> FeatureTracker:
        return self._tracker

    @property
    def detector(self) -> AnomalyDetector:
        return self._detector
