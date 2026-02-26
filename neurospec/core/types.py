"""Core data types for NeuroSpec.

All shared dataclasses used across the compiler, runtime, monitor, and forge layers.
Every type is JSON-serializable via its to_dict/from_dict methods.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SteeringDirection(str, Enum):
    """Direction of feature steering."""

    AMPLIFY = "amplify"
    SUPPRESS = "suppress"


class ActionKind(str, Enum):
    """Kind of action to take when a monitor triggers."""

    STEER_AWAY = "steer_away"
    STEER_TOWARD = "steer_toward"
    PAUSE_AND_RETRY = "pause_and_retry"
    ALERT = "alert"
    KILL = "kill"
    LOG = "log"


class Severity(str, Enum):
    """Alert / validation severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# Feature Catalog types
# ---------------------------------------------------------------------------


@dataclass
class Feature:
    """A single interpretable feature extracted from a SAE."""

    id: int
    label: str
    description: str
    tags: list[str] = field(default_factory=list)
    confidence: float = 1.0
    top_activations: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "description": self.description,
            "tags": self.tags,
            "confidence": self.confidence,
            "top_activations": self.top_activations,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Feature:
        return cls(
            id=data["id"],
            label=data["label"],
            description=data["description"],
            tags=data.get("tags", []),
            confidence=data.get("confidence", 1.0),
            top_activations=data.get("top_activations", []),
        )


@dataclass
class FeatureCatalog:
    """Searchable catalog of SAE features for a specific model + SAE pair."""

    model_id: str
    sae_id: str
    features: list[Feature] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # --- Persistence ---

    def save(self, path: str) -> None:
        """Save catalog to JSON file."""
        data = {
            "model_id": self.model_id,
            "sae_id": self.sae_id,
            "metadata": self.metadata,
            "features": [f.to_dict() for f in self.features],
        }
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> FeatureCatalog:
        """Load catalog from JSON file."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            model_id=data["model_id"],
            sae_id=data["sae_id"],
            metadata=data.get("metadata", {}),
            features=[Feature.from_dict(f) for f in data.get("features", [])],
        )

    # --- Search ---

    def search(self, query: str, top_k: int = 10) -> list[Feature]:
        """Simple keyword search over feature labels, descriptions, and tags."""
        query_lower = query.lower()
        scored: list[tuple[float, Feature]] = []

        for feat in self.features:
            score = 0.0
            if query_lower in feat.label.lower():
                score += 3.0
            if query_lower in feat.description.lower():
                score += 1.0
            for tag in feat.tags:
                if query_lower in tag.lower():
                    score += 2.0
            if score > 0:
                scored.append((score, feat))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [feat for _, feat in scored[:top_k]]

    def get_by_id(self, feature_id: int) -> Feature | None:
        """Look up a feature by its integer ID."""
        for feat in self.features:
            if feat.id == feature_id:
                return feat
        return None

    def get_by_label(self, label: str) -> Feature | None:
        """Look up a feature by exact label match."""
        label_lower = label.lower()
        for feat in self.features:
            if feat.label.lower() == label_lower:
                return feat
        return None


# ---------------------------------------------------------------------------
# Steering / Compilation types
# ---------------------------------------------------------------------------


@dataclass
class SteeringVector:
    """A compiled steering intervention targeting specific SAE features."""

    feature_ids: list[int]
    direction: SteeringDirection
    strength: float
    label: str = ""
    layer: int | None = None
    conditional: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_ids": self.feature_ids,
            "direction": self.direction.value,
            "strength": self.strength,
            "label": self.label,
            "layer": self.layer,
            "conditional": self.conditional,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SteeringVector:
        return cls(
            feature_ids=data["feature_ids"],
            direction=SteeringDirection(data["direction"]),
            strength=data["strength"],
            label=data.get("label", ""),
            layer=data.get("layer"),
            conditional=data.get("conditional"),
        )


@dataclass
class ProbeConfig:
    """Configuration for a linear probe that monitors a behavioral property."""

    name: str
    feature_ids: list[int]
    threshold: float = 0.5
    layer: int | None = None
    probe_source: str = "catalog"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "feature_ids": self.feature_ids,
            "threshold": self.threshold,
            "layer": self.layer,
            "probe_source": self.probe_source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProbeConfig:
        return cls(
            name=data["name"],
            feature_ids=data["feature_ids"],
            threshold=data.get("threshold", 0.5),
            layer=data.get("layer"),
            probe_source=data.get("probe_source", "catalog"),
        )


@dataclass
class MonitorConfig:
    """Configuration for a runtime feature monitor."""

    name: str
    feature_ids: list[int]
    threshold: float
    action: ActionKind
    action_params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "feature_ids": self.feature_ids,
            "threshold": self.threshold,
            "action": self.action.value,
            "action_params": self.action_params,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MonitorConfig:
        return cls(
            name=data["name"],
            feature_ids=data["feature_ids"],
            threshold=data["threshold"],
            action=ActionKind(data["action"]),
            action_params=data.get("action_params", {}),
        )


@dataclass
class AlertConfig:
    """Configuration for a feature alert."""

    name: str
    feature_ids: list[int]
    threshold: float
    severity: Severity = Severity.WARNING
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "feature_ids": self.feature_ids,
            "threshold": self.threshold,
            "severity": self.severity.value,
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AlertConfig:
        return cls(
            name=data["name"],
            feature_ids=data["feature_ids"],
            threshold=data["threshold"],
            severity=Severity(data.get("severity", "warning")),
            message=data.get("message", ""),
        )


@dataclass
class RLFRConfig:
    """Configuration for RLFR (Reinforcement Learning from Feature Rewards) training."""

    probe_configs: list[ProbeConfig]
    training_budget: int = 500
    learning_rate: float = 1e-5
    reward_scale: float = 1.0
    frozen_model: bool = True
    verify_no_regression: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "probe_configs": [p.to_dict() for p in self.probe_configs],
            "training_budget": self.training_budget,
            "learning_rate": self.learning_rate,
            "reward_scale": self.reward_scale,
            "frozen_model": self.frozen_model,
            "verify_no_regression": self.verify_no_regression,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RLFRConfig:
        return cls(
            probe_configs=[ProbeConfig.from_dict(p) for p in data.get("probe_configs", [])],
            training_budget=data.get("training_budget", 500),
            learning_rate=data.get("learning_rate", 1e-5),
            reward_scale=data.get("reward_scale", 1.0),
            frozen_model=data.get("frozen_model", True),
            verify_no_regression=data.get("verify_no_regression", []),
        )


# ---------------------------------------------------------------------------
# Compilation result
# ---------------------------------------------------------------------------


@dataclass
class SpecCompilationResult:
    """The output of compiling a single spec declaration."""

    spec_name: str
    model_id: str
    sae_id: str
    steering_vectors: list[SteeringVector] = field(default_factory=list)
    probes: list[ProbeConfig] = field(default_factory=list)
    monitors: list[MonitorConfig] = field(default_factory=list)
    alerts: list[AlertConfig] = field(default_factory=list)
    rlfr: RLFRConfig | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "spec_name": self.spec_name,
            "model_id": self.model_id,
            "sae_id": self.sae_id,
            "steering_vectors": [sv.to_dict() for sv in self.steering_vectors],
            "probes": [p.to_dict() for p in self.probes],
            "monitors": [m.to_dict() for m in self.monitors],
            "alerts": [a.to_dict() for a in self.alerts],
            "metadata": self.metadata,
        }
        if self.rlfr is not None:
            result["rlfr"] = self.rlfr.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SpecCompilationResult:
        rlfr = None
        if "rlfr" in data and data["rlfr"] is not None:
            rlfr = RLFRConfig.from_dict(data["rlfr"])
        return cls(
            spec_name=data["spec_name"],
            model_id=data["model_id"],
            sae_id=data["sae_id"],
            steering_vectors=[
                SteeringVector.from_dict(sv) for sv in data.get("steering_vectors", [])
            ],
            probes=[ProbeConfig.from_dict(p) for p in data.get("probes", [])],
            monitors=[MonitorConfig.from_dict(m) for m in data.get("monitors", [])],
            alerts=[AlertConfig.from_dict(a) for a in data.get("alerts", [])],
            rlfr=rlfr,
            metadata=data.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# Marketplace types
# ---------------------------------------------------------------------------


@dataclass
class SpecMeta:
    """Metadata for a published spec in the marketplace."""

    name: str
    version: str
    author: str
    description: str
    model_id: str
    sae_id: str = ""
    tags: list[str] = field(default_factory=list)
    downloads: int = 0
    spec_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "author": self.author,
            "description": self.description,
            "model_id": self.model_id,
            "sae_id": self.sae_id,
            "tags": self.tags,
            "downloads": self.downloads,
            "spec_hash": self.spec_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SpecMeta:
        return cls(
            name=data["name"],
            version=data["version"],
            author=data.get("author", ""),
            description=data.get("description", ""),
            model_id=data.get("model_id", ""),
            sae_id=data.get("sae_id", ""),
            tags=data.get("tags", []),
            downloads=data.get("downloads", 0),
            spec_hash=data.get("spec_hash", ""),
        )


# ---------------------------------------------------------------------------
# Protocols (structural typing interfaces)
# ---------------------------------------------------------------------------


class SAEInterface(Protocol):
    """Protocol for Sparse Autoencoder implementations.

    Any SAE that provides encode/decode methods with the right signature
    can be used with NeuroSpec — no inheritance needed.
    """

    def encode(self, activations: Any) -> Any:
        """Dense activations -> sparse feature activations."""
        ...

    def decode(self, features: Any) -> Any:
        """Sparse feature activations -> dense activations."""
        ...


class ModelInterface(Protocol):
    """Protocol for model wrappers used by the runtime."""

    def forward(self, input_ids: Any, **kwargs: Any) -> Any:
        """Run forward pass and return outputs."""
        ...


# ---------------------------------------------------------------------------
# Validation types
# ---------------------------------------------------------------------------


@dataclass
class ValidationError:
    """A validation error or warning from the spec validator."""

    message: str
    line: int = 0
    column: int = 0
    severity: str = "error"

    def __str__(self) -> str:
        loc = f"line {self.line}" if self.line else "unknown"
        return f"[{self.severity}] {loc}: {self.message}"
