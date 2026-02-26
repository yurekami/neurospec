"""Global configuration for NeuroSpec.

Manages default settings for the compiler, runtime, and other components.
Settings can be overridden via environment variables or explicit configuration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


_DEFAULT_REGISTRY_DIR = Path.home() / ".neurospec" / "registry"
_DEFAULT_CACHE_DIR = Path.home() / ".neurospec" / "cache"


@dataclass
class NeuroSpecConfig:
    """Top-level configuration for NeuroSpec."""

    # Paths
    registry_dir: Path = field(default_factory=lambda: _DEFAULT_REGISTRY_DIR)
    cache_dir: Path = field(default_factory=lambda: _DEFAULT_CACHE_DIR)

    # Runtime defaults
    default_device: str = "auto"
    default_layer: int | None = None
    default_sae_id: str = ""

    # Catalog defaults
    catalog_top_k: int = 20
    labeler_provider: str = "openai"
    labeler_model: str = "gpt-4o-mini"

    # Runtime steering
    steering_max_strength: float = 2.0
    steering_default_strength: float = 0.5

    # Monitor defaults
    monitor_window_size: int = 100
    monitor_z_score_threshold: float = 3.0

    # RLFR defaults
    rlfr_learning_rate: float = 1e-5
    rlfr_default_budget: int = 500

    # Server
    server_host: str = "0.0.0.0"
    server_port: int = 8420

    def ensure_dirs(self) -> None:
        """Create required directories if they do not exist."""
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls) -> NeuroSpecConfig:
        """Build config from environment variables, falling back to defaults."""
        config = cls()

        if val := os.environ.get("NEUROSPEC_REGISTRY_DIR"):
            config.registry_dir = Path(val)
        if val := os.environ.get("NEUROSPEC_CACHE_DIR"):
            config.cache_dir = Path(val)
        if val := os.environ.get("NEUROSPEC_DEVICE"):
            config.default_device = val
        if val := os.environ.get("NEUROSPEC_LABELER_PROVIDER"):
            config.labeler_provider = val
        if val := os.environ.get("NEUROSPEC_LABELER_MODEL"):
            config.labeler_model = val
        if val := os.environ.get("NEUROSPEC_SERVER_HOST"):
            config.server_host = val
        if val := os.environ.get("NEUROSPEC_SERVER_PORT"):
            config.server_port = int(val)

        return config


# Module-level singleton
_config: NeuroSpecConfig | None = None


def get_config() -> NeuroSpecConfig:
    """Return the global NeuroSpec config, lazily initialized from env."""
    global _config
    if _config is None:
        _config = NeuroSpecConfig.from_env()
    return _config


def set_config(config: NeuroSpecConfig) -> None:
    """Override the global config (useful in tests)."""
    global _config
    _config = config
