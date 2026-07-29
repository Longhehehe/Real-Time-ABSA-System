"""Typed collector failures that can be recorded in crawl manifests."""


class CollectorError(RuntimeError):
    code = "COLLECTOR_ERROR"


class BlockedError(CollectorError):
    code = "BLOCKED"


class RateLimitedError(CollectorError):
    code = "RATE_LIMITED"


class ProductUnavailableError(CollectorError):
    code = "PRODUCT_UNAVAILABLE"


class AuthenticationError(CollectorError):
    code = "AUTH_EXPIRED"


class ResponseSchemaError(CollectorError):
    code = "SCHEMA_CHANGED"


class TransportError(CollectorError):
    code = "TRANSPORT_ERROR"


class DependencyError(CollectorError):
    code = "MISSING_DEPENDENCY"
