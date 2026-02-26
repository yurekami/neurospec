"""LLM-based feature auto-labeling for the NeuroSpec catalog.

Takes SAE feature activations (top-activating text examples) and uses an
LLM to generate human-readable labels, descriptions, and tags for each feature.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from neurospec.catalog.extractor import FeatureActivations
from neurospec.core.config import get_config

logger = logging.getLogger(__name__)


@dataclass
class FeatureLabel:
    """LLM-generated label for a SAE feature."""

    feature_id: int
    label: str
    description: str
    tags: list[str]
    confidence: float


class FeatureLabeler:
    """Label SAE features using an LLM provider.

    Supports OpenAI and Anthropic APIs via optional dependencies.
    Falls back to heuristic labeling if no API key is available.
    """

    def __init__(
        self,
        provider: str = "openai",
        model: str | None = None,
    ) -> None:
        config = get_config()
        self._provider = provider
        self._model = model or config.labeler_model

    def label_features(
        self,
        activations: list[FeatureActivations],
        batch_size: int = 10,
    ) -> list[FeatureLabel]:
        """Label a list of features based on their top-activating examples.

        Args:
            activations: Feature activation data from the extractor.
            batch_size: Number of features to label per LLM call.

        Returns:
            List of FeatureLabel objects.
        """
        labels: list[FeatureLabel] = []

        for batch_start in range(0, len(activations), batch_size):
            batch = activations[batch_start : batch_start + batch_size]
            batch_labels = self._label_batch(batch)
            labels.extend(batch_labels)
            logger.info(
                "Labeled features %d-%d of %d",
                batch_start,
                min(batch_start + batch_size, len(activations)),
                len(activations),
            )

        return labels

    def _label_batch(self, batch: list[FeatureActivations]) -> list[FeatureLabel]:
        """Label a batch of features via LLM or heuristic fallback."""
        if self._provider == "openai":
            return self._label_with_openai(batch)
        elif self._provider == "anthropic":
            return self._label_with_anthropic(batch)
        else:
            return self._label_heuristic(batch)

    def _label_with_openai(self, batch: list[FeatureActivations]) -> list[FeatureLabel]:
        """Label features using OpenAI API."""
        try:
            import openai
        except ImportError:
            logger.warning("openai not installed — falling back to heuristic labeling")
            return self._label_heuristic(batch)

        prompt = self._build_labeling_prompt(batch)
        try:
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at interpreting neural network features. Given top-activating text examples for SAE features, provide concise labels, descriptions, and tags.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
            )
            return self._parse_llm_response(response.choices[0].message.content or "{}", batch)
        except Exception as exc:
            logger.warning("OpenAI labeling failed: %s — falling back to heuristic", exc)
            return self._label_heuristic(batch)

    def _label_with_anthropic(self, batch: list[FeatureActivations]) -> list[FeatureLabel]:
        """Label features using Anthropic API."""
        try:
            import anthropic
        except ImportError:
            logger.warning("anthropic not installed — falling back to heuristic labeling")
            return self._label_heuristic(batch)

        prompt = self._build_labeling_prompt(batch)
        try:
            client = anthropic.Anthropic()
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
                system="You are an expert at interpreting neural network features. Given top-activating text examples for SAE features, provide concise labels, descriptions, and tags. Respond in JSON format.",
            )
            content = response.content[0].text if response.content else "{}"
            return self._parse_llm_response(content, batch)
        except Exception as exc:
            logger.warning("Anthropic labeling failed: %s — falling back to heuristic", exc)
            return self._label_heuristic(batch)

    def _label_heuristic(self, batch: list[FeatureActivations]) -> list[FeatureLabel]:
        """Simple heuristic labeling when no LLM is available.

        Uses common tokens across top examples to infer a label.
        """
        labels: list[FeatureLabel] = []

        for feat in batch:
            tokens = [ex.token.strip() for ex in feat.top_examples if ex.token.strip()]
            if tokens:
                # Use most common token as label basis
                token_counts: dict[str, int] = {}
                for t in tokens:
                    token_counts[t] = token_counts.get(t, 0) + 1
                common_token = max(token_counts, key=lambda k: token_counts[k])
                label = f"feature_{feat.feature_id}_{common_token}"
            else:
                label = f"feature_{feat.feature_id}"

            labels.append(
                FeatureLabel(
                    feature_id=feat.feature_id,
                    label=label,
                    description=f"SAE feature {feat.feature_id} (max activation: {feat.max_activation:.3f})",
                    tags=["auto-labeled", "heuristic"],
                    confidence=0.3,
                )
            )

        return labels

    def _build_labeling_prompt(self, batch: list[FeatureActivations]) -> str:
        """Build the LLM prompt for labeling a batch of features."""
        parts = [
            "Label the following SAE features based on their top-activating text examples.",
            "For each feature, provide:",
            "- label: short snake_case name (e.g., 'sql_injection_pattern')",
            "- description: 1-2 sentence explanation",
            "- tags: list of relevant tags",
            "- confidence: 0.0-1.0 how confident you are in the label",
            "",
            'Respond in JSON format: {"features": [{"feature_id": ..., "label": ..., "description": ..., "tags": [...], "confidence": ...}]}',
            "",
        ]

        for feat in batch:
            parts.append(
                f"=== Feature {feat.feature_id} (max activation: {feat.max_activation:.3f}) ==="
            )
            for i, ex in enumerate(feat.top_examples[:5]):  # Show top 5
                parts.append(
                    f"  Example {i + 1} (act={ex.activation:.3f}, token='{ex.token}'): {ex.text[:200]}"
                )
            parts.append("")

        return "\n".join(parts)

    def _parse_llm_response(
        self,
        response_text: str,
        batch: list[FeatureActivations],
    ) -> list[FeatureLabel]:
        """Parse the LLM JSON response into FeatureLabel objects."""
        try:
            data = json.loads(response_text)
            features_data = data.get("features", [])

            labels: list[FeatureLabel] = []
            for item in features_data:
                labels.append(
                    FeatureLabel(
                        feature_id=item["feature_id"],
                        label=item.get("label", f"feature_{item['feature_id']}"),
                        description=item.get("description", ""),
                        tags=item.get("tags", []),
                        confidence=float(item.get("confidence", 0.5)),
                    )
                )

            # Fill in any missing features with heuristic labels
            labeled_ids = {l.feature_id for l in labels}
            missing = [f for f in batch if f.feature_id not in labeled_ids]
            if missing:
                labels.extend(self._label_heuristic(missing))

            return labels

        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("Failed to parse LLM response: %s", exc)
            return self._label_heuristic(batch)
