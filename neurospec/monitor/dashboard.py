"""Rich terminal dashboard for real-time feature monitoring.

Displays live feature activation values, anomaly alerts, and
steering status in a terminal-based UI using the Rich library.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from neurospec.monitor.detector import AnomalyDetector
from neurospec.monitor.tracker import FeatureTracker

logger = logging.getLogger(__name__)

try:
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table

    HAS_RICH = True
except ImportError:
    HAS_RICH = False


class MonitorDashboard:
    """Terminal-based dashboard for real-time feature monitoring.

    Displays:
    - Feature activation table with sparklines
    - Anomaly alert feed
    - Steering status overview
    """

    def __init__(
        self,
        tracker: FeatureTracker,
        detector: AnomalyDetector,
        watched_features: list[int] | None = None,
        refresh_rate: float = 0.5,
    ) -> None:
        self._tracker = tracker
        self._detector = detector
        self._watched = watched_features
        self._refresh_rate = refresh_rate
        self._alert_log: list[str] = []
        self._running = False

    def run(self) -> None:
        """Start the live dashboard (blocks until stopped)."""
        if not HAS_RICH:
            logger.warning("Rich is not installed — falling back to plain text monitoring")
            self._run_plain()
            return

        self._running = True
        console = Console()

        with Live(
            self._render(), refresh_per_second=int(1 / self._refresh_rate), console=console
        ) as live:
            while self._running:
                # Check for anomalies
                anomalies = self._detector.check_all()
                for anomaly in anomalies:
                    alert_msg = (
                        f"[{time.strftime('%H:%M:%S')}] "
                        f"{anomaly.anomaly_type.upper()}: {anomaly.label} "
                        f"(z={anomaly.z_score:.1f}, val={anomaly.current_value:.3f})"
                    )
                    self._alert_log.append(alert_msg)
                    # Keep last 20 alerts
                    self._alert_log = self._alert_log[-20:]

                live.update(self._render())
                time.sleep(self._refresh_rate)

    def stop(self) -> None:
        """Signal the dashboard to stop."""
        self._running = False

    def _render(self) -> Any:
        """Render the full dashboard layout."""
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="alerts", size=10),
        )

        layout["header"].update(
            Panel(
                "[bold]NeuroSpec Monitor[/bold] | Features tracked: "
                f"{len(self._tracker.get_all_stats())}",
                style="bold white on blue",
            )
        )

        layout["body"].update(self._render_feature_table())
        layout["alerts"].update(self._render_alerts())

        return layout

    def _render_feature_table(self) -> Any:
        """Render the feature activation table."""
        table = Table(title="Feature Activations", expand=True)
        table.add_column("ID", style="cyan", width=8)
        table.add_column("Label", style="white", width=30)
        table.add_column("Current", style="green", width=10)
        table.add_column("Mean", style="yellow", width=10)
        table.add_column("Std", style="yellow", width=10)
        table.add_column("Z-Score", style="red", width=10)
        table.add_column("Trend", width=15)

        all_stats = self._tracker.get_all_stats()

        # Filter to watched features if specified
        if self._watched:
            stats_items = [(fid, all_stats[fid]) for fid in self._watched if fid in all_stats]
        else:
            stats_items = sorted(
                all_stats.items(), key=lambda x: abs(x[1].last_value), reverse=True
            )[:20]

        for feat_id, stats in stats_items:
            z_score = (
                abs(stats.last_value - stats.mean) / stats.std_dev if stats.std_dev > 0 else 0.0
            )
            trend = self._compute_sparkline(feat_id)

            # Color z-score based on severity
            z_style = "green"
            if z_score > 2.0:
                z_style = "yellow"
            if z_score > 3.0:
                z_style = "red bold"

            table.add_row(
                str(feat_id),
                stats.label,
                f"{stats.last_value:.3f}",
                f"{stats.mean:.3f}",
                f"{stats.std_dev:.3f}",
                f"[{z_style}]{z_score:.2f}[/{z_style}]",
                trend,
            )

        return Panel(table)

    def _render_alerts(self) -> Any:
        """Render the alert feed."""
        if not self._alert_log:
            content = "[dim]No anomalies detected[/dim]"
        else:
            content = "\n".join(self._alert_log[-8:])

        return Panel(content, title="Alerts", border_style="red")

    def _compute_sparkline(self, feature_id: int) -> str:
        """Generate a simple ASCII sparkline from recent values."""
        recent = self._tracker.get_recent(feature_id, n=10)
        if len(recent) < 2:
            return ""

        values = [s.value for s in recent]
        min_v = min(values)
        max_v = max(values)
        range_v = max_v - min_v

        if range_v == 0:
            return "." * len(values)

        chars = " .:-=+*#%@"
        sparkline = ""
        for v in values:
            idx = int((v - min_v) / range_v * (len(chars) - 1))
            sparkline += chars[idx]

        return sparkline

    def _run_plain(self) -> None:
        """Fallback plain-text monitoring when Rich is not available."""
        self._running = True
        while self._running:
            anomalies = self._detector.check_all()
            for anomaly in anomalies:
                print(
                    f"[ANOMALY] {anomaly.anomaly_type}: feature {anomaly.feature_id} "
                    f"({anomaly.label}) z={anomaly.z_score:.2f} value={anomaly.current_value:.3f}"
                )

            stats = self._tracker.get_all_stats()
            if stats:
                print(f"\n--- {len(stats)} features tracked ---")
                for feat_id, s in sorted(stats.items())[:10]:
                    print(f"  [{feat_id}] {s.label}: {s.last_value:.3f} (mean={s.mean:.3f})")
                print()

            time.sleep(self._refresh_rate * 2)
