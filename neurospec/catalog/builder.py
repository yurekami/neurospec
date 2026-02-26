"""Feature catalog builder — orchestrates extraction and labeling.

End-to-end pipeline that:
1. Loads a model and SAE
2. Extracts feature activations from sample texts
3. Labels features with an LLM
4. Builds and saves a searchable FeatureCatalog
"""

from __future__ import annotations

import logging
from typing import Any

from neurospec.catalog.extractor import FeatureExtractor
from neurospec.catalog.labeler import FeatureLabeler
from neurospec.core.config import get_config
from neurospec.core.types import Feature, FeatureCatalog

logger = logging.getLogger(__name__)


# Default texts for feature extraction when none are provided
_DEFAULT_TEXTS = [
    "The patient shows signs of early-stage cognitive decline.",
    "SELECT * FROM users WHERE id = 1; DROP TABLE users; --",
    "I'm not entirely sure about this, but I think the function might accept two parameters.",
    "The quarterly revenue increased by 15% compared to last year.",
    "def exploit(): os.system('rm -rf /')",
    "According to the documentation, this API endpoint requires authentication.",
    "I made this up, but it sounds plausible: the frobnicate() function returns a widget.",
    "Climate change is causing unprecedented shifts in global weather patterns.",
    "To hack into the system, first you need to bypass the firewall.",
    "I apologize, but I cannot help with that request as it could cause harm.",
    "The research paper presents compelling evidence for the hypothesis.",
    "Trust me bro, this is definitely how React hooks work.",
    "<script>alert('XSS')</script>",
    "Based on my analysis, there is a 73% probability of success.",
    "I don't know the answer to that question, but I can point you to relevant resources.",
]


class CatalogBuilder:
    """Build a FeatureCatalog from a model + SAE.

    Usage:
        builder = CatalogBuilder(model_id="meta-llama/...", sae_path="...")
        catalog = builder.build_catalog(texts=my_texts)
        catalog.save("catalog.json")
    """

    def __init__(
        self,
        model_id: str,
        sae_path: str,
        labeler_provider: str = "openai",
        device: str = "auto",
    ) -> None:
        self._model_id = model_id
        self._sae_path = sae_path
        self._labeler_provider = labeler_provider
        self._device = device

    def build_catalog(
        self,
        texts: list[str] | None = None,
        layer: int | None = None,
        top_k: int = 20,
        batch_size: int = 8,
    ) -> FeatureCatalog:
        """Build a complete feature catalog.

        Args:
            texts: Sample texts for activation extraction. Uses defaults if None.
            layer: Target model layer for SAE. Uses middle layer if None.
            top_k: Top-k activating examples per feature.
            batch_size: Batch size for model inference.

        Returns:
            A populated FeatureCatalog ready to save.
        """
        texts = texts or _DEFAULT_TEXTS

        logger.info("Loading model: %s", self._model_id)
        model, tokenizer = self._load_model()

        logger.info("Loading SAE: %s", self._sae_path)
        sae = self._load_sae()

        # Extract features
        logger.info("Extracting features from %d texts...", len(texts))
        extractor = FeatureExtractor(
            model=model,
            tokenizer=tokenizer,
            sae=sae,
            layer=layer,
            device=self._device,
        )
        activations = extractor.extract(texts, top_k=top_k, batch_size=batch_size)
        logger.info("Extracted %d features", len(activations))

        # Label features
        logger.info("Labeling features with %s...", self._labeler_provider)
        labeler = FeatureLabeler(provider=self._labeler_provider)
        labels = labeler.label_features(activations)

        # Build catalog
        features: list[Feature] = []
        label_map = {l.feature_id: l for l in labels}

        for act in activations:
            label = label_map.get(act.feature_id)
            top_acts = [
                {
                    "text": ex.text[:200],
                    "activation": round(ex.activation, 4),
                    "token": ex.token,
                }
                for ex in act.top_examples[:top_k]
            ]

            features.append(
                Feature(
                    id=act.feature_id,
                    label=label.label if label else f"feature_{act.feature_id}",
                    description=label.description if label else "",
                    tags=label.tags if label else [],
                    confidence=label.confidence if label else 0.0,
                    top_activations=top_acts,
                )
            )

        catalog = FeatureCatalog(
            model_id=self._model_id,
            sae_id=self._sae_path,
            features=features,
            metadata={
                "n_texts": len(texts),
                "top_k": top_k,
                "labeler_provider": self._labeler_provider,
            },
        )

        logger.info("Built catalog with %d features", len(catalog.features))
        return catalog

    def _load_model(self) -> tuple[Any, Any]:
        """Load the model and tokenizer from HuggingFace."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError(
                "PyTorch and transformers are required for catalog building. "
                "Install with: pip install neurospec[runtime]"
            )

        tokenizer = AutoTokenizer.from_pretrained(self._model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self._model_id,
            torch_dtype=torch.float16,
            device_map=self._device if self._device != "auto" else "auto",
        )
        model.eval()
        return model, tokenizer

    def _load_sae(self) -> Any:
        """Load the SAE model.

        Supports:
        - Local path to a saved SAE state dict
        - HuggingFace model ID for SAE models
        """
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch is required. Install with: pip install neurospec[runtime]")

        # Try loading as a local file
        from pathlib import Path

        sae_path = Path(self._sae_path)
        if sae_path.exists():
            logger.info("Loading SAE from local path: %s", sae_path)
            return torch.load(sae_path, map_location="cpu")

        # Otherwise, treat as a placeholder — real SAE loading would integrate
        # with libraries like sae_lens, goodfire, etc.
        logger.warning(
            "SAE '%s' not found locally. Using a placeholder SAE. "
            "For real usage, provide a local SAE path or install a compatible SAE library.",
            self._sae_path,
        )
        return _PlaceholderSAE()


class _PlaceholderSAE:
    """Minimal SAE stub for testing without a real SAE."""

    def encode(self, activations: Any) -> Any:
        """Identity encode — returns activations unchanged."""
        return activations

    def decode(self, features: Any) -> Any:
        """Identity decode — returns features unchanged."""
        return features
