"""Feature name resolver for the NeuroSpec compiler.

Maps natural-language feature names from .ns specs to SAE feature IDs
by searching the FeatureCatalog.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from neurospec.core.types import Feature, FeatureCatalog

logger = logging.getLogger(__name__)


@dataclass
class ResolutionResult:
    """Result of resolving a feature name to SAE feature IDs."""

    query: str
    feature_ids: list[int] = field(default_factory=list)
    matched_features: list[Feature] = field(default_factory=list)
    confidence: float = 0.0
    resolved: bool = False

    @property
    def best_match(self) -> Feature | None:
        """Return the best-matching feature, if any."""
        return self.matched_features[0] if self.matched_features else None


class FeatureResolver:
    """Resolve natural-language feature names to SAE feature IDs.

    Uses keyword matching against the catalog's labels, descriptions, and tags.
    When no catalog is available, feature names are passed through as
    unresolved placeholders (with a warning).
    """

    def __init__(self, catalog: FeatureCatalog | None = None) -> None:
        self._catalog = catalog

    @property
    def has_catalog(self) -> bool:
        return self._catalog is not None

    def resolve(self, name: str, top_k: int = 3) -> ResolutionResult:
        """Resolve a feature name to a list of matching SAE feature IDs.

        Args:
            name: Natural-language feature name from the spec (e.g., "sql_injection_pattern").
            top_k: Maximum number of matching features to return.

        Returns:
            A ResolutionResult with matched IDs and confidence.
        """
        if self._catalog is None:
            logger.warning("No catalog loaded — feature '%s' will be unresolved", name)
            return ResolutionResult(query=name, resolved=False)

        # Try exact label match first
        exact = self._catalog.get_by_label(name)
        if exact is not None:
            return ResolutionResult(
                query=name,
                feature_ids=[exact.id],
                matched_features=[exact],
                confidence=exact.confidence,
                resolved=True,
            )

        # Fall back to keyword search
        results = self._catalog.search(name, top_k=top_k)
        if not results:
            logger.warning("No features matched '%s' in catalog", name)
            return ResolutionResult(query=name, resolved=False)

        return ResolutionResult(
            query=name,
            feature_ids=[f.id for f in results],
            matched_features=results,
            confidence=results[0].confidence,
            resolved=True,
        )

    def resolve_many(self, names: list[str], top_k: int = 3) -> dict[str, ResolutionResult]:
        """Resolve multiple feature names at once.

        Returns a dict mapping each name to its ResolutionResult.
        """
        return {name: self.resolve(name, top_k=top_k) for name in names}
