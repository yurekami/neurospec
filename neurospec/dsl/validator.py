"""Semantic validator for the NeuroSpec DSL AST.

Checks the parsed AST for semantic issues:
- Duplicate spec names
- Empty feature lists
- Invalid strength / threshold values
- Missing model or SAE declarations
- Compose references to unknown specs
"""

from __future__ import annotations

from neurospec.core.types import ValidationError
from neurospec.dsl.ast_nodes import (
    AlertIfRule,
    AmplifyRule,
    CompileToWeightsRule,
    ComposeDecl,
    MonitorRule,
    RequireRule,
    SpecFile,
    SuppressRule,
)


def validate_spec(spec_file: SpecFile) -> list[ValidationError]:
    """Run all validation passes on a parsed SpecFile.

    Returns a list of ValidationError objects (may be empty if valid).
    """
    errors: list[ValidationError] = []

    _validate_declarations(spec_file, errors)
    _validate_specs(spec_file, errors)
    _validate_composes(spec_file, errors)

    return errors


def _validate_declarations(spec_file: SpecFile, errors: list[ValidationError]) -> None:
    """Validate top-level model/sae declarations."""
    if spec_file.model is None:
        errors.append(
            ValidationError(
                message="Missing 'model' declaration",
                severity="warning",
            )
        )
    if spec_file.sae is None:
        errors.append(
            ValidationError(
                message="Missing 'sae' declaration",
                severity="warning",
            )
        )


def _validate_specs(spec_file: SpecFile, errors: list[ValidationError]) -> None:
    """Validate spec declarations and their rules."""
    seen_names: set[str] = set()

    for spec in spec_file.specs:
        # Duplicate name check
        if spec.name in seen_names:
            errors.append(
                ValidationError(
                    message=f"Duplicate spec name: '{spec.name}'",
                    line=spec.line,
                    severity="error",
                )
            )
        seen_names.add(spec.name)

        if not spec.rules:
            errors.append(
                ValidationError(
                    message=f"Spec '{spec.name}' has no rules",
                    line=spec.line,
                    severity="warning",
                )
            )

        for rule in spec.rules:
            _validate_rule(rule, spec.name, errors)


def _validate_rule(rule: object, spec_name: str, errors: list[ValidationError]) -> None:
    """Validate a single rule node."""
    if isinstance(rule, SuppressRule):
        _check_features(rule.features, "suppress", rule.name, rule.line, errors)
        _check_strength(rule.strength, "suppress", rule.name, rule.line, errors)

    elif isinstance(rule, AmplifyRule):
        _check_features(rule.features, "amplify", rule.name, rule.line, errors)
        _check_strength(rule.strength, "amplify", rule.name, rule.line, errors)

    elif isinstance(rule, RequireRule):
        if not rule.when and not rule.amplify_features:
            errors.append(
                ValidationError(
                    message=f"require '{rule.name}' needs at least 'when' or 'amplify' clause",
                    line=rule.line,
                    severity="error",
                )
            )
        if rule.amplify_features:
            _check_features(rule.amplify_features, "require", rule.name, rule.line, errors)
        _check_strength(rule.strength, "require", rule.name, rule.line, errors)

    elif isinstance(rule, MonitorRule):
        _check_features(rule.features, "monitor", rule.name, rule.line, errors)
        _check_threshold(rule.threshold, "monitor", rule.name, rule.line, errors)

    elif isinstance(rule, AlertIfRule):
        _check_features(rule.features, "alert_if", rule.name, rule.line, errors)
        _check_threshold(rule.threshold, "alert_if", rule.name, rule.line, errors)
        valid_severities = {"info", "warning", "error", "critical"}
        if rule.severity not in valid_severities:
            errors.append(
                ValidationError(
                    message=f"alert_if '{rule.name}' has invalid severity: '{rule.severity}' (expected one of {valid_severities})",
                    line=rule.line,
                    severity="error",
                )
            )

    elif isinstance(rule, CompileToWeightsRule):
        if rule.training_budget <= 0:
            errors.append(
                ValidationError(
                    message=f"compile_to_weights '{rule.name}' has invalid training_budget: {rule.training_budget}",
                    line=rule.line,
                    severity="error",
                )
            )
        valid_methods = {"rlfr", "dpo", "sft"}
        if rule.method not in valid_methods:
            errors.append(
                ValidationError(
                    message=f"compile_to_weights '{rule.name}' has unknown method: '{rule.method}' (expected one of {valid_methods})",
                    line=rule.line,
                    severity="warning",
                )
            )


def _validate_composes(spec_file: SpecFile, errors: list[ValidationError]) -> None:
    """Validate compose declarations."""
    known_specs = {s.name for s in spec_file.specs}
    known_aliases = {imp.alias for imp in spec_file.imports}
    available = known_specs | known_aliases

    for compose in spec_file.composes:
        for source in compose.sources:
            if source not in available:
                errors.append(
                    ValidationError(
                        message=f"compose '{compose.name}' references unknown spec: '{source}'",
                        line=compose.line,
                        severity="error",
                    )
                )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_features(
    features: list[str],
    rule_kind: str,
    rule_name: str,
    line: int,
    errors: list[ValidationError],
) -> None:
    if not features:
        errors.append(
            ValidationError(
                message=f"{rule_kind} '{rule_name}' has empty features list",
                line=line,
                severity="error",
            )
        )


def _check_strength(
    strength: float,
    rule_kind: str,
    rule_name: str,
    line: int,
    errors: list[ValidationError],
) -> None:
    if not 0.0 <= strength <= 2.0:
        errors.append(
            ValidationError(
                message=f"{rule_kind} '{rule_name}' has out-of-range strength: {strength} (expected 0.0-2.0)",
                line=line,
                severity="warning",
            )
        )


def _check_threshold(
    threshold: float,
    rule_kind: str,
    rule_name: str,
    line: int,
    errors: list[ValidationError],
) -> None:
    if not 0.0 <= threshold <= 1.0:
        errors.append(
            ValidationError(
                message=f"{rule_kind} '{rule_name}' has out-of-range threshold: {threshold} (expected 0.0-1.0)",
                line=line,
                severity="warning",
            )
        )
