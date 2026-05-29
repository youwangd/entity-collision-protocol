"""Encryption at rest — AES content encryption for JSONL + SQLite (Design §5.5).

Uses Fernet (AES-128-CBC with HMAC-SHA256) from the `cryptography` package.
Key source: environment variable, keyfile, or passed directly.

Graceful degradation: if `cryptography` not installed, encryption is disabled
and a warning is logged. This maintains the zero-infra principle.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from cryptography.fernet import Fernet, InvalidToken

    _HAS_CRYPTO = True
except ImportError:
    _HAS_CRYPTO = False
    InvalidToken = Exception  # type: ignore[misc,assignment]


class EncryptionError(Exception):
    """Raised on encryption/decryption failures."""


class ContentEncryptor:
    """Encrypts/decrypts content strings.

    Key sources (in priority order):
    1. Direct key parameter
    2. Environment variable (ENGRAM_ENCRYPTION_KEY)
    3. Keyfile (~/.engram/key or custom path)

    The key can be:
    - A Fernet key (44 url-safe base64 chars)
    - A passphrase (hashed to derive Fernet key via SHA-256 + base64)
    """

    def __init__(
        self,
        enabled: bool = False,
        key: str | None = None,
        key_source: str = "env",  # "env" | "keyfile" | "direct"
        key_path: str | None = None,
    ) -> None:
        self._enabled = enabled
        self._fernet: Fernet | None = None

        if not enabled:
            return

        if not _HAS_CRYPTO:
            logger.warning(
                "encryption requested but `cryptography` package not installed — "
                "install with: pip install engram[encryption] or pip install cryptography"
            )
            self._enabled = False
            return

        raw_key = self._resolve_key(key, key_source, key_path)
        if raw_key is None:
            logger.warning("encryption enabled but no key found — disabling encryption")
            self._enabled = False
            return

        fernet_key = self._to_fernet_key(raw_key)
        self._fernet = Fernet(fernet_key)
        logger.info("encryption initialized (source=%s)", key_source)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def encrypt(self, plaintext: str) -> str:
        """Encrypt a string. Returns base64-encoded ciphertext prefixed with 'enc:'."""
        if not self._enabled or self._fernet is None:
            return plaintext
        try:
            token = self._fernet.encrypt(plaintext.encode("utf-8"))
            return "enc:" + token.decode("ascii")
        except Exception as e:
            raise EncryptionError(f"encryption failed: {e}") from e

    def decrypt(self, ciphertext: str) -> str:
        """Decrypt a string. If not prefixed with 'enc:', returns as-is (plaintext passthrough)."""
        if not ciphertext.startswith("enc:"):
            return ciphertext  # plaintext passthrough — supports mixed encrypted/plain stores
        if not self._enabled or self._fernet is None:
            raise EncryptionError(
                "encrypted content found but encryption is not enabled — "
                "provide the encryption key to decrypt"
            )
        try:
            token = ciphertext[4:].encode("ascii")
            return self._fernet.decrypt(token).decode("utf-8")
        except InvalidToken as e:
            raise EncryptionError("decryption failed — wrong key or corrupted data") from e
        except Exception as e:
            raise EncryptionError(f"decryption failed: {e}") from e

    def _resolve_key(
        self, key: str | None, key_source: str, key_path: str | None
    ) -> str | None:
        """Resolve encryption key from the configured source."""
        if key_source == "direct" and key:
            return key

        # Try env var first (works for all sources as fallback)
        env_key = os.environ.get("ENGRAM_ENCRYPTION_KEY")
        if env_key:
            return env_key

        if key_source == "env":
            return env_key

        if key_source == "keyfile":
            path = Path(key_path) if key_path else Path.home() / ".engram" / "key"
            if path.exists():
                return path.read_text(encoding="utf-8").strip()
            logger.warning("keyfile not found: %s", path)
            return None

        return key or env_key

    @staticmethod
    def _to_fernet_key(raw: str) -> bytes:
        """Convert a raw key/passphrase to a valid Fernet key.

        If the key is already a valid Fernet key (44 url-safe base64 chars), use it directly.
        Otherwise, derive one via SHA-256 hash → base64.
        """
        raw_bytes = raw.encode("utf-8")

        # Check if already a valid Fernet key
        if len(raw) == 44:
            try:
                decoded = base64.urlsafe_b64decode(raw_bytes)
                if len(decoded) == 32:
                    return raw_bytes
            except Exception:
                pass

        # Derive from passphrase: SHA-256 → 32 bytes → base64
        derived = hashlib.sha256(raw_bytes).digest()
        return base64.urlsafe_b64encode(derived)

    @staticmethod
    def generate_key() -> str:
        """Generate a new random Fernet key."""
        if not _HAS_CRYPTO:
            raise EncryptionError(
                "cannot generate key — `cryptography` package not installed"
            )
        return Fernet.generate_key().decode("ascii")
