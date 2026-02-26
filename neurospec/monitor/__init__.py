"""NeuroSpec monitor — real-time feature tracking, anomaly detection, and dashboard.

Usage:
    from neurospec.monitor import Guardian, MonitorDashboard

    guardian = Guardian.from_spec(compiled_spec)
    dashboard = MonitorDashboard(guardian.tracker, guardian.detector)
    dashboard.run()
"""

from neurospec.monitor.dashboard import MonitorDashboard
from neurospec.monitor.detector import Anomaly, AnomalyDetector
from neurospec.monitor.guardian import Guardian
from neurospec.monitor.tracker import FeatureStats, FeatureTracker

__all__ = [
    "Anomaly",
    "AnomalyDetector",
    "FeatureStats",
    "FeatureTracker",
    "Guardian",
    "MonitorDashboard",
]
