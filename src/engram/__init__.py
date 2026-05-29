"""Engram — Neuroscience-inspired memory for AI agents."""

__version__ = "0.1.0"

from engram.core.types import (
    ConsolidationReport,
    DataClassification,
    Event,
    EventType,
    Memory,
    MemoryState,
    MemoryType,
    RecallContext,
    ScoredMemory,
)
from engram.core.config import Config
from engram.engine import Engram
from engram.security.acl import AccessPolicy, Permission
from engram.security.encryption import ContentEncryptor, EncryptionError

__all__ = [
    "Engram",
    "Config",
    "Event",
    "EventType",
    "Memory",
    "MemoryType",
    "MemoryState",
    "DataClassification",
    "ScoredMemory",
    "RecallContext",
    "ConsolidationReport",
    "AccessPolicy",
    "Permission",
    "ContentEncryptor",
    "EncryptionError",
    "__version__",
]
