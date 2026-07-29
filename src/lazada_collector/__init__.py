"""Lazada review collection package."""

__version__ = "0.3.0"

from .collector import CollectorConfig, LazadaCollector
from .schema import ProductRecord, ReviewRecord

__all__ = [
    "CollectorConfig",
    "LazadaCollector",
    "ProductRecord",
    "ReviewRecord",
]
