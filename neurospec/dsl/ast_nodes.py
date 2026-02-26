"""AST node definitions for the NeuroSpec DSL.

These dataclasses form the abstract syntax tree produced by the parser.
The tree structure mirrors the .ns file layout:

    SpecFile
      -> model, sae declarations
      -> import statements
      -> spec declarations
          -> rules (suppress, amplify, require, monitor, alert_if, compile_to_weights)
      -> compose declarations
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Top-level declarations
# ---------------------------------------------------------------------------


@dataclass
class ModelDecl:
    """A `model "model_id"` declaration."""

    model_id: str
    line: int = 0


@dataclass
class SAEDecl:
    """A `sae "sae_id"` declaration."""

    sae_id: str
    line: int = 0


@dataclass
class ImportDecl:
    """An `import "path" as alias` declaration."""

    path: str
    alias: str
    line: int = 0


# ---------------------------------------------------------------------------
# Rule nodes (inside spec blocks)
# ---------------------------------------------------------------------------


@dataclass
class SuppressRule:
    """A `suppress name { features: [...], action: ... }` rule."""

    name: str
    features: list[str] = field(default_factory=list)
    action: str = "steer_away"
    action_params: dict[str, Any] = field(default_factory=dict)
    strength: float = 0.5
    line: int = 0


@dataclass
class AmplifyRule:
    """An `amplify name { features: [...], strength: ... }` rule."""

    name: str
    features: list[str] = field(default_factory=list)
    strength: float = 0.5
    line: int = 0


@dataclass
class RequireRule:
    """A `require name { when: ..., amplify: [...], strength: ... }` rule."""

    name: str
    when: str = ""
    amplify_features: list[str] = field(default_factory=list)
    strength: float = 0.5
    line: int = 0


@dataclass
class MonitorRule:
    """A `monitor name { features: [...], threshold: ..., action: ... }` rule."""

    name: str
    features: list[str] = field(default_factory=list)
    threshold: float = 0.5
    action: str = "log"
    action_params: dict[str, Any] = field(default_factory=dict)
    line: int = 0


@dataclass
class AlertIfRule:
    """An `alert_if name { features: [...], threshold: ..., severity: ... }` rule."""

    name: str
    features: list[str] = field(default_factory=list)
    threshold: float = 0.5
    severity: str = "warning"
    message: str = ""
    line: int = 0


@dataclass
class CompileToWeightsRule:
    """A `compile_to_weights name { ... }` rule for RLFR training config."""

    name: str
    method: str = "rlfr"
    probe_source: str = "catalog"
    training_budget: int = 500
    verify_no_regression: list[str] = field(default_factory=list)
    line: int = 0


# Union of all rule types
Rule = SuppressRule | AmplifyRule | RequireRule | MonitorRule | AlertIfRule | CompileToWeightsRule


# ---------------------------------------------------------------------------
# Spec declarations
# ---------------------------------------------------------------------------


@dataclass
class SpecDecl:
    """A `spec name { ... }` declaration containing rules."""

    name: str
    rules: list[Rule] = field(default_factory=list)
    line: int = 0


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


@dataclass
class OverrideDecl:
    """An `override spec.rule.property: value` inside a compose block."""

    target: str  # e.g., "med.hallucination_risk.threshold"
    value: Any
    line: int = 0


@dataclass
class ComposeDecl:
    """A `spec name = compose(a, b) { override ... }` declaration."""

    name: str
    sources: list[str] = field(default_factory=list)
    overrides: list[OverrideDecl] = field(default_factory=list)
    line: int = 0


# ---------------------------------------------------------------------------
# Root node
# ---------------------------------------------------------------------------


@dataclass
class SpecFile:
    """Root of the AST — represents an entire .ns file."""

    model: ModelDecl | None = None
    sae: SAEDecl | None = None
    imports: list[ImportDecl] = field(default_factory=list)
    specs: list[SpecDecl] = field(default_factory=list)
    composes: list[ComposeDecl] = field(default_factory=list)
