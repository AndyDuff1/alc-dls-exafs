"""Larch CLI Wrapper - A lightweight wrapper around larch for EXAFS processing."""

from pathlib import Path

__version__ = "0.1.0"
__all__ = ["LarchWrapper", "DEFAULT_CACHE_DIR"]

# Default cache directory for FEFF calculations and results
DEFAULT_CACHE_DIR = Path.home() / ".larch_cache"
