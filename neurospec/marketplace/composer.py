"""Spec composition and conflict resolution.

Handles merging multiple compiled specs together, detecting conflicts
between steering vectors, and applying overrides from compose declarations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from neurospec.core.types import (
    AlertConfig,
    MonitorConfig,
    ProbeConfig,
    SpecCompilationResult,
    SteeringDirection,
    SteeringVector,
)

logger = logging.getLogger(__name__)


@dataclass
class ConflictInfo:
    """Information about a conflict between specs."""

    feature_ids: list[int]
    spec_a: str
    spec_b: str
    conflict_type: str  # "opposing_direction", "strength_mismatch"
    description: str


class SpecComposer:
    """Compose multiple compiled specs into a single spec.

    Handles:
    - Merging steering vectors (with conflict detection)
    - Merging monitors and alerts
    - Applying overrides from compose declarations
    """

    def compose(
        self,
        specs: list[SpecCompilationResult],
        overrides: dict[str, Any] | None = None,
        name: str = "composed",
    ) -> tuple[SpecCompilationResult, list[ConflictInfo]]:
        """Compose multiple specs into one.

        Args:
            specs: List of compiled specs to merge.
            overrides: Optional dict of dotted-path overrides.
            name: Name for the composed spec.

        Returns:
            Tuple of (merged spec, list of detected conflicts).
        """
        if not specs:
            return SpecCompilationResult(spec_name=name, model_id="", sae_id=""), []

        # Start with the first spec as base
        result = SpecCompilationResult(
            spec_name=name,
            model_id=specs[0].model_id,
            sae_id=specs[0].sae_id,
        )

        conflicts: list[ConflictInfo] = []

        # Merge all specs
        for spec in specs:
            new_conflicts = self._merge_steering(result, spec)
            conflicts.extend(new_conflicts)
            self._merge_monitors(result, spec)
            self._merge_alerts(result, spec)
            self._merge_probes(result, spec)

        # Apply overrides
        if overrides:
            self._apply_overrides(result, overrides)

        if conflicts:
            logger.warning("Composition detected %d conflicts", len(conflicts))
            for c in conflicts:
                logger.warning(
                    "  Conflict: %s (%s vs %s) — %s",
                    c.conflict_type,
                    c.spec_a,
                    c.spec_b,
                    c.description,
                )

        return result, conflicts

    def _merge_steering(
        self,
        result: SpecCompilationResult,
        source: SpecCompilationResult,
    ) -> list[ConflictInfo]:
        """Merge steering vectors, detecting conflicts."""
        conflicts: list[ConflictInfo] = []

        for new_sv in source.steering_vectors:
            # Check for conflicts with existing vectors
            conflict = self._find_steering_conflict(result, new_sv, source.spec_name)
            if conflict is not None:
                conflicts.append(conflict)
                # Resolve by keeping the later (source) spec's vector
                result.steering_vectors = [
                    sv
                    for sv in result.steering_vectors
                    if not set(sv.feature_ids) & set(new_sv.feature_ids)
                ]

            result.steering_vectors.append(new_sv)

        return conflicts

    def _find_steering_conflict(
        self,
        result: SpecCompilationResult,
        new_sv: SteeringVector,
        source_name: str,
    ) -> ConflictInfo | None:
        """Check if a new steering vector conflicts with existing ones."""
        new_ids = set(new_sv.feature_ids)

        for existing in result.steering_vectors:
            overlap = new_ids & set(existing.feature_ids)
            if not overlap:
                continue

            # Opposing directions = conflict
            if existing.direction != new_sv.direction:
                return ConflictInfo(
                    feature_ids=list(overlap),
                    spec_a=result.spec_name,
                    spec_b=source_name,
                    conflict_type="opposing_direction",
                    description=(
                        f"Features {list(overlap)} steered {existing.direction.value} "
                        f"in {result.spec_name} but {new_sv.direction.value} in {source_name}"
                    ),
                )

            # Same direction but different strength = warning (not hard conflict)
            if abs(existing.strength - new_sv.strength) > 0.3:
                return ConflictInfo(
                    feature_ids=list(overlap),
                    spec_a=result.spec_name,
                    spec_b=source_name,
                    conflict_type="strength_mismatch",
                    description=(
                        f"Features {list(overlap)} steered at {existing.strength:.2f} "
                        f"in {result.spec_name} but {new_sv.strength:.2f} in {source_name}"
                    ),
                )

        return None

    def _merge_monitors(self, result: SpecCompilationResult, source: SpecCompilationResult) -> None:
        """Merge monitors, keeping unique names."""
        existing_names = {m.name for m in result.monitors}
        for monitor in source.monitors:
            if monitor.name not in existing_names:
                result.monitors.append(monitor)

    def _merge_alerts(self, result: SpecCompilationResult, source: SpecCompilationResult) -> None:
        """Merge alerts, keeping unique names."""
        existing_names = {a.name for a in result.alerts}
        for alert in source.alerts:
            if alert.name not in existing_names:
                result.alerts.append(alert)

    def _merge_probes(self, result: SpecCompilationResult, source: SpecCompilationResult) -> None:
        """Merge probes, keeping unique names."""
        existing_names = {p.name for p in result.probes}
        for probe in source.probes:
            if probe.name not in existing_names:
                result.probes.append(probe)

    def _apply_overrides(self, result: SpecCompilationResult, overrides: dict[str, Any]) -> None:
        """Apply dotted-path overrides to the composed spec.

        Override format: {"spec.rule.property": value}
        Example: {"med.hallucination_risk.threshold": 0.2}
        """
        for path, value in overrides.items():
            parts = path.split(".")
            if len(parts) < 2:
                logger.warning("Invalid override path: %s", path)
                continue

            target_name = parts[-2] if len(parts) >= 2 else parts[0]
            property_name = parts[-1]

            # Try to apply to monitors
            for monitor in result.monitors:
                if monitor.name == target_name:
                    if property_name == "threshold":
                        monitor.threshold = float(value)
                        logger.info("Override applied: %s.threshold = %s", target_name, value)

            # Try to apply to steering vectors
            for sv in result.steering_vectors:
                if sv.label == target_name:
                    if property_name == "strength":
                        sv.strength = float(value)
                        logger.info("Override applied: %s.strength = %s", target_name, value)
