"""Official-split, multi-profile experiment framework."""

from .profiles import DEFAULT_CONFIG_PATH, PROJECT_ROOT, load_registry
from .schema import PreparedProfile, SplitData

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "PROJECT_ROOT",
    "PreparedProfile",
    "SplitData",
    "load_registry",
]
