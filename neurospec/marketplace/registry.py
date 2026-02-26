"""Local spec registry for the NeuroSpec marketplace.

Manages a local directory of published specs that can be listed,
searched, installed, and published.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

from neurospec.core.config import get_config
from neurospec.core.types import SpecMeta

logger = logging.getLogger(__name__)


class SpecRegistry:
    """Local filesystem-based spec registry.

    Stores published specs in ~/.neurospec/registry/ with metadata.
    """

    def __init__(self, registry_dir: Path | None = None) -> None:
        config = get_config()
        self._dir = registry_dir or config.registry_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._dir / "index.json"

    def list_specs(self) -> list[SpecMeta]:
        """List all specs in the registry."""
        index = self._load_index()
        return [SpecMeta.from_dict(entry) for entry in index.get("specs", [])]

    def search(self, query: str) -> list[SpecMeta]:
        """Search specs by name, description, or tags."""
        query_lower = query.lower()
        results: list[SpecMeta] = []

        for spec in self.list_specs():
            if (
                query_lower in spec.name.lower()
                or query_lower in spec.description.lower()
                or any(query_lower in tag.lower() for tag in spec.tags)
            ):
                results.append(spec)

        return results

    def publish(self, spec_path: str, meta: SpecMeta) -> None:
        """Publish a spec to the registry.

        Args:
            spec_path: Path to the .ns spec file.
            meta: Metadata for the published spec.
        """
        src = Path(spec_path)
        if not src.exists():
            raise FileNotFoundError(f"Spec file not found: {spec_path}")

        # Create spec directory
        spec_dir = self._dir / meta.name / meta.version
        spec_dir.mkdir(parents=True, exist_ok=True)

        # Copy spec file
        shutil.copy2(src, spec_dir / src.name)

        # Write metadata
        meta_path = spec_dir / "meta.json"
        meta_path.write_text(json.dumps(meta.to_dict(), indent=2), encoding="utf-8")

        # Update index
        index = self._load_index()
        specs = index.get("specs", [])

        # Remove existing version if present
        specs = [s for s in specs if not (s["name"] == meta.name and s["version"] == meta.version)]
        specs.append(meta.to_dict())
        index["specs"] = specs

        self._save_index(index)
        logger.info("Published %s v%s to registry", meta.name, meta.version)

    def install(self, spec_name: str, version: str | None = None, target_dir: str = ".") -> Path:
        """Install a spec from the registry.

        Args:
            spec_name: Name of the spec to install.
            version: Specific version (default: latest).
            target_dir: Directory to install the spec into.

        Returns:
            Path to the installed spec file.
        """
        # Find the spec
        specs = [s for s in self.list_specs() if s.name == spec_name]
        if not specs:
            raise ValueError(f"Spec '{spec_name}' not found in registry")

        if version:
            matching = [s for s in specs if s.version == version]
            if not matching:
                raise ValueError(f"Version '{version}' of '{spec_name}' not found")
            meta = matching[0]
        else:
            # Latest version (simple string sort)
            meta = sorted(specs, key=lambda s: s.version, reverse=True)[0]

        # Copy spec file to target
        spec_dir = self._dir / meta.name / meta.version
        spec_files = list(spec_dir.glob("*.ns"))
        if not spec_files:
            raise FileNotFoundError(f"No .ns file found for {meta.name} v{meta.version}")

        target = Path(target_dir) / spec_files[0].name
        shutil.copy2(spec_files[0], target)

        # Update download count
        self._increment_downloads(meta.name, meta.version)

        logger.info("Installed %s v%s to %s", meta.name, meta.version, target)
        return target

    def get_meta(self, spec_name: str, version: str | None = None) -> SpecMeta | None:
        """Get metadata for a spec."""
        specs = [s for s in self.list_specs() if s.name == spec_name]
        if not specs:
            return None
        if version:
            matching = [s for s in specs if s.version == version]
            return matching[0] if matching else None
        return sorted(specs, key=lambda s: s.version, reverse=True)[0]

    def _load_index(self) -> dict[str, Any]:
        """Load the registry index."""
        if self._index_path.exists():
            return json.loads(self._index_path.read_text(encoding="utf-8"))
        return {"specs": []}

    def _save_index(self, index: dict[str, Any]) -> None:
        """Save the registry index."""
        self._index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")

    def _increment_downloads(self, name: str, version: str) -> None:
        """Increment download count for a spec."""
        index = self._load_index()
        for spec in index.get("specs", []):
            if spec["name"] == name and spec["version"] == version:
                spec["downloads"] = spec.get("downloads", 0) + 1
        self._save_index(index)
