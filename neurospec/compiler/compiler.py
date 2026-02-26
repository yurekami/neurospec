"""NeuroSpec compiler — translates parsed AST into feature-level interventions.

Takes a SpecDecl (from the parser) and a FeatureCatalog, and produces a
SpecCompilationResult containing steering vectors, probes, monitors, and alerts.
"""

from __future__ import annotations

import logging
from typing import Any

from neurospec.compiler.resolver import FeatureResolver
from neurospec.core.types import (
    ActionKind,
    AlertConfig,
    FeatureCatalog,
    MonitorConfig,
    ProbeConfig,
    RLFRConfig,
    Severity,
    SpecCompilationResult,
    SteeringDirection,
    SteeringVector,
)
from neurospec.dsl.ast_nodes import (
    AlertIfRule,
    AmplifyRule,
    CompileToWeightsRule,
    MonitorRule,
    RequireRule,
    SpecDecl,
    SuppressRule,
)

logger = logging.getLogger(__name__)


# Map action name strings to ActionKind enum
_ACTION_MAP: dict[str, ActionKind] = {
    "steer_away": ActionKind.STEER_AWAY,
    "steer_toward": ActionKind.STEER_TOWARD,
    "pause_and_retry": ActionKind.PAUSE_AND_RETRY,
    "alert": ActionKind.ALERT,
    "kill": ActionKind.KILL,
    "log": ActionKind.LOG,
}


class CompilationError(Exception):
    """Raised when compilation fails due to unrecoverable issues."""


class NeuroSpecCompiler:
    """Compile SpecDecl AST nodes into SpecCompilationResult objects.

    Usage:
        compiler = NeuroSpecCompiler(catalog=my_catalog)
        result = compiler.compile(spec_decl)
    """

    def __init__(
        self,
        catalog: FeatureCatalog | None = None,
        model_id: str = "",
        sae_id: str = "",
    ) -> None:
        self._resolver = FeatureResolver(catalog)
        self._model_id = model_id
        self._sae_id = sae_id

    def compile(self, spec: SpecDecl) -> SpecCompilationResult:
        """Compile a single spec declaration into a SpecCompilationResult."""
        result = SpecCompilationResult(
            spec_name=spec.name,
            model_id=self._model_id,
            sae_id=self._sae_id,
        )

        rlfr_probes: list[ProbeConfig] = []
        rlfr_budget: int = 500
        rlfr_no_regression: list[str] = []

        for rule in spec.rules:
            if isinstance(rule, SuppressRule):
                self._compile_suppress(rule, result)
            elif isinstance(rule, AmplifyRule):
                self._compile_amplify(rule, result)
            elif isinstance(rule, RequireRule):
                self._compile_require(rule, result)
            elif isinstance(rule, MonitorRule):
                self._compile_monitor(rule, result)
            elif isinstance(rule, AlertIfRule):
                self._compile_alert_if(rule, result)
            elif isinstance(rule, CompileToWeightsRule):
                probes, budget, no_reg = self._compile_to_weights(rule, result)
                rlfr_probes.extend(probes)
                rlfr_budget = budget
                rlfr_no_regression.extend(no_reg)

        # Build RLFR config if any compile_to_weights rules exist
        if rlfr_probes:
            result.rlfr = RLFRConfig(
                probe_configs=rlfr_probes,
                training_budget=rlfr_budget,
                verify_no_regression=rlfr_no_regression,
            )

        return result

    # ------------------------------------------------------------------
    # Rule compilers
    # ------------------------------------------------------------------

    def _compile_suppress(self, rule: SuppressRule, result: SpecCompilationResult) -> None:
        """Compile a suppress rule into a negative steering vector."""
        feature_ids = self._resolve_features(rule.features)

        result.steering_vectors.append(
            SteeringVector(
                feature_ids=feature_ids,
                direction=SteeringDirection.SUPPRESS,
                strength=rule.strength,
                label=rule.name,
            )
        )

    def _compile_amplify(self, rule: AmplifyRule, result: SpecCompilationResult) -> None:
        """Compile an amplify rule into a positive steering vector."""
        feature_ids = self._resolve_features(rule.features)

        result.steering_vectors.append(
            SteeringVector(
                feature_ids=feature_ids,
                direction=SteeringDirection.AMPLIFY,
                strength=rule.strength,
                label=rule.name,
            )
        )

    def _compile_require(self, rule: RequireRule, result: SpecCompilationResult) -> None:
        """Compile a require rule into a conditional steering vector + probe."""
        feature_ids = self._resolve_features(rule.amplify_features)

        # Create a conditional steering vector
        result.steering_vectors.append(
            SteeringVector(
                feature_ids=feature_ids,
                direction=SteeringDirection.AMPLIFY,
                strength=rule.strength,
                label=rule.name,
                conditional=rule.when if rule.when else None,
            )
        )

        # Also create a probe for the condition
        if rule.when:
            result.probes.append(
                ProbeConfig(
                    name=f"{rule.name}_condition",
                    feature_ids=feature_ids,
                    threshold=0.5,
                )
            )

    def _compile_monitor(self, rule: MonitorRule, result: SpecCompilationResult) -> None:
        """Compile a monitor rule into a MonitorConfig."""
        feature_ids = self._resolve_features(rule.features)
        action = _ACTION_MAP.get(rule.action, ActionKind.LOG)

        result.monitors.append(
            MonitorConfig(
                name=rule.name,
                feature_ids=feature_ids,
                threshold=rule.threshold,
                action=action,
                action_params=rule.action_params,
            )
        )

    def _compile_alert_if(self, rule: AlertIfRule, result: SpecCompilationResult) -> None:
        """Compile an alert_if rule into an AlertConfig."""
        feature_ids = self._resolve_features(rule.features)
        severity = _SEVERITY_MAP.get(rule.severity, Severity.WARNING)

        result.alerts.append(
            AlertConfig(
                name=rule.name,
                feature_ids=feature_ids,
                threshold=rule.threshold,
                severity=severity,
                message=rule.message,
            )
        )

    def _compile_to_weights(
        self,
        rule: CompileToWeightsRule,
        result: SpecCompilationResult,
    ) -> tuple[list[ProbeConfig], int, list[str]]:
        """Compile a compile_to_weights rule into RLFR probe configs.

        Returns (probes, training_budget, verify_no_regression).
        """
        # Collect all feature IDs from all other rules in this spec for probes
        all_feature_ids: list[int] = []
        for sv in result.steering_vectors:
            all_feature_ids.extend(sv.feature_ids)
        for mon in result.monitors:
            all_feature_ids.extend(mon.feature_ids)

        # Deduplicate
        unique_ids = list(dict.fromkeys(all_feature_ids))

        probes = [
            ProbeConfig(
                name=f"{rule.name}_rlfr_probe",
                feature_ids=unique_ids,
                probe_source=rule.probe_source,
            )
        ]

        return probes, rule.training_budget, rule.verify_no_regression

    # ------------------------------------------------------------------
    # Feature resolution
    # ------------------------------------------------------------------

    def _resolve_features(self, names: list[str]) -> list[int]:
        """Resolve a list of feature names to SAE feature IDs.

        If no catalog is loaded, uses hash-based placeholder IDs.
        """
        if not names:
            return []

        if not self._resolver.has_catalog:
            # Generate deterministic placeholder IDs from names
            return [abs(hash(name)) % 100000 for name in names]

        ids: list[int] = []
        for name in names:
            resolution = self._resolver.resolve(name)
            if resolution.resolved:
                ids.extend(resolution.feature_ids[:1])  # Take best match
            else:
                logger.warning("Could not resolve feature '%s' — using placeholder", name)
                ids.append(abs(hash(name)) % 100000)

        return ids


# Severity string -> enum mapping
_SEVERITY_MAP: dict[str, Severity] = {
    "info": Severity.INFO,
    "warning": Severity.WARNING,
    "error": Severity.ERROR,
    "critical": Severity.CRITICAL,
}
