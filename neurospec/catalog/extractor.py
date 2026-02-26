"""SAE feature extraction pipeline.

Runs sample texts through a model, collects activations at a target layer,
passes them through an SAE encoder, and identifies the top-activating
examples for each feature.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]


@dataclass
class ActivationRecord:
    """A single activation sample for a feature."""

    text: str
    activation: float
    token_index: int = 0
    token: str = ""


@dataclass
class FeatureActivations:
    """Top-activating examples for a single SAE feature."""

    feature_id: int
    top_examples: list[ActivationRecord] = field(default_factory=list)
    mean_activation: float = 0.0
    max_activation: float = 0.0


class FeatureExtractor:
    """Extract SAE feature activations from a model.

    Runs texts through the model, hooks a target layer to capture
    activations, and encodes them through the SAE.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        sae: Any,
        layer: int | None = None,
        device: str = "auto",
    ) -> None:
        if torch is None:
            raise ImportError(
                "PyTorch is required for feature extraction. Install with: pip install neurospec[runtime]"
            )

        self._model = model
        self._tokenizer = tokenizer
        self._sae = sae
        self._layer = layer
        self._device = self._resolve_device(device)

    def extract(
        self,
        texts: list[str],
        top_k: int = 20,
        batch_size: int = 8,
    ) -> list[FeatureActivations]:
        """Extract feature activations for all SAE features across the given texts.

        Args:
            texts: Input texts to process.
            top_k: Number of top-activating examples to keep per feature.
            batch_size: Batch size for model inference.

        Returns:
            List of FeatureActivations, one per SAE feature.
        """
        all_activations: dict[int, list[ActivationRecord]] = {}

        for batch_start in range(0, len(texts), batch_size):
            batch_texts = texts[batch_start : batch_start + batch_size]
            batch_records = self._process_batch(batch_texts)

            for feature_id, records in batch_records.items():
                if feature_id not in all_activations:
                    all_activations[feature_id] = []
                all_activations[feature_id].extend(records)

        # Sort and trim to top_k per feature
        results: list[FeatureActivations] = []
        for feature_id in sorted(all_activations.keys()):
            records = all_activations[feature_id]
            records.sort(key=lambda r: r.activation, reverse=True)
            top = records[:top_k]

            mean_act = sum(r.activation for r in top) / len(top) if top else 0.0
            max_act = top[0].activation if top else 0.0

            results.append(
                FeatureActivations(
                    feature_id=feature_id,
                    top_examples=top,
                    mean_activation=mean_act,
                    max_activation=max_act,
                )
            )

        logger.info("Extracted %d features from %d texts", len(results), len(texts))
        return results

    def _process_batch(self, texts: list[str]) -> dict[int, list[ActivationRecord]]:
        """Process a batch of texts and return per-feature activation records."""
        captured: list[Any] = []

        def hook_fn(module: Any, input: Any, output: Any) -> None:
            captured.append(output)

        # Register hook on target layer
        target_layer = self._get_target_layer()
        handle = target_layer.register_forward_hook(hook_fn)

        try:
            inputs = self._tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                self._model(**inputs)

            # Process captured activations through SAE
            if not captured:
                return {}

            hidden_states = captured[0]
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]

            # Encode through SAE: (batch, seq, hidden) -> (batch, seq, n_features)
            features = self._sae.encode(hidden_states)

            return self._collect_records(features, texts, inputs)

        finally:
            handle.remove()
            captured.clear()

    def _collect_records(
        self,
        features: Any,
        texts: list[str],
        inputs: dict[str, Any],
    ) -> dict[int, list[ActivationRecord]]:
        """Collect activation records from SAE feature outputs."""
        records: dict[int, list[ActivationRecord]] = {}

        # features shape: (batch, seq, n_features)
        batch_size, seq_len, n_features = features.shape

        for batch_idx in range(batch_size):
            for feat_idx in range(n_features):
                # Get max activation across sequence for this feature
                activations = features[batch_idx, :, feat_idx]
                max_val = float(activations.max())

                if max_val > 0.0:
                    max_pos = int(activations.argmax())
                    token_ids = inputs["input_ids"][batch_idx]
                    token_str = self._tokenizer.decode([token_ids[max_pos]])

                    record = ActivationRecord(
                        text=texts[batch_idx],
                        activation=max_val,
                        token_index=max_pos,
                        token=token_str,
                    )

                    if feat_idx not in records:
                        records[feat_idx] = []
                    records[feat_idx].append(record)

        return records

    def _get_target_layer(self) -> Any:
        """Get the model layer to hook for activation capture."""
        if self._layer is not None:
            layers = self._get_model_layers()
            if self._layer < len(layers):
                return layers[self._layer]

        # Default: use the middle layer
        layers = self._get_model_layers()
        mid = len(layers) // 2
        logger.info("Using default layer %d (middle of %d layers)", mid, len(layers))
        return layers[mid]

    def _get_model_layers(self) -> list[Any]:
        """Extract the list of transformer layers from the model."""
        # Support common HuggingFace model architectures
        if hasattr(self._model, "model"):
            inner = self._model.model
            if hasattr(inner, "layers"):
                return list(inner.layers)
            if hasattr(inner, "decoder") and hasattr(inner.decoder, "layers"):
                return list(inner.decoder.layers)
        if hasattr(self._model, "transformer"):
            transformer = self._model.transformer
            if hasattr(transformer, "h"):
                return list(transformer.h)
        raise ValueError(
            "Could not find transformer layers in model. Supported: LlamaForCausalLM, GPT2LMHeadModel"
        )

    @staticmethod
    def _resolve_device(device: str) -> str:
        """Resolve 'auto' device to the best available."""
        if device != "auto":
            return device
        if torch is not None and torch.cuda.is_available():
            return "cuda"
        return "cpu"
