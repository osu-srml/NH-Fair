"""NH-Fair: unified fairness benchmark for vision models and vision-language models."""

try:
    from importlib.metadata import version

    __version__ = version("nh-fair")
except Exception:  # pragma: no cover - local checkout before install
    __version__ = "1.0.0"

__all__ = ["__version__"]
