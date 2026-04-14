"""Filesystem locations inside the installed `release_benchmark` package."""

from pathlib import Path


def package_root() -> Path:
    """Top-level package directory (contains `configs/`, `methods/`, etc.)."""
    return Path(__file__).resolve().parent


def sweep_config_dir() -> Path:
    return package_root() / "configs" / "sweep"
