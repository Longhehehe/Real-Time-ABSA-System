"""Active Vietnamese multi-polarity ABSA modeling package."""

from .schema import ASPECTS, POLARITIES, ABSAExample, AnnotationSchemaError

__all__ = [
    "ASPECTS",
    "POLARITIES",
    "ABSAExample",
    "AnnotationSchemaError",
]

__version__ = "0.1.0"
