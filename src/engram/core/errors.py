"""Engram error hierarchy."""


class EngramError(Exception):
    """Base error for all Engram errors."""


class StoreError(EngramError):
    """Error in storage operations."""


class BufferError(StoreError):
    """Error in event buffer operations."""


class RetrievalError(EngramError):
    """Error in retrieval operations."""


class ConsolidationError(EngramError):
    """Error in consolidation operations."""


class SecurityError(EngramError):
    """Security violation (injection, ACL, etc.)."""


class ConfigError(EngramError):
    """Invalid configuration."""
