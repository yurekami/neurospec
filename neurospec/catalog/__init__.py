"""NeuroSpec catalog — SAE feature extraction, labeling, and search.

Usage:
    from neurospec.catalog import CatalogBuilder

    builder = CatalogBuilder(model_id="meta-llama/...", sae_path="...")
    catalog = builder.build_catalog()
    catalog.save("catalog.json")
"""

from neurospec.catalog.builder import CatalogBuilder
from neurospec.catalog.extractor import FeatureActivations, FeatureExtractor
from neurospec.catalog.labeler import FeatureLabel, FeatureLabeler

__all__ = [
    "CatalogBuilder",
    "FeatureActivations",
    "FeatureExtractor",
    "FeatureLabel",
    "FeatureLabeler",
]
