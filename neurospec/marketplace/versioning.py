"""Semantic versioning utilities for NeuroSpec specs.

Provides version comparison, bumping, and compatibility checking
for published specs in the marketplace.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, order=True)
class SemVer:
    """A semantic version (major.minor.patch)."""

    major: int
    minor: int
    patch: int
    prerelease: str = ""

    def __str__(self) -> str:
        base = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            return f"{base}-{self.prerelease}"
        return base

    @classmethod
    def parse(cls, version_str: str) -> SemVer:
        """Parse a version string like '1.2.3' or '1.2.3-beta.1'."""
        match = re.match(
            r"^(\d+)\.(\d+)\.(\d+)(?:-(.+))?$",
            version_str.strip(),
        )
        if not match:
            raise ValueError(f"Invalid semver: {version_str!r}")

        return cls(
            major=int(match.group(1)),
            minor=int(match.group(2)),
            patch=int(match.group(3)),
            prerelease=match.group(4) or "",
        )

    def bump_major(self) -> SemVer:
        return SemVer(self.major + 1, 0, 0)

    def bump_minor(self) -> SemVer:
        return SemVer(self.major, self.minor + 1, 0)

    def bump_patch(self) -> SemVer:
        return SemVer(self.major, self.minor, self.patch + 1)

    def is_compatible_with(self, other: SemVer) -> bool:
        """Check if this version is backwards-compatible with another.

        Within the same major version, newer minor/patch versions are
        considered compatible (following semver conventions).
        """
        if self.major != other.major:
            return False
        if self.major == 0:
            # Pre-1.0: minor version changes may break
            return self.minor == other.minor
        return True


def is_newer(a: str, b: str) -> bool:
    """Check if version string `a` is newer than `b`."""
    return SemVer.parse(a) > SemVer.parse(b)


def latest_version(versions: list[str]) -> str:
    """Return the latest version from a list of version strings."""
    if not versions:
        raise ValueError("Empty version list")
    parsed = [(SemVer.parse(v), v) for v in versions]
    parsed.sort(key=lambda x: x[0], reverse=True)
    return parsed[0][1]
