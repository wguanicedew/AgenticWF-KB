"""
SecretBox: encryption at rest for secrets the KB keeps in its database (model API keys).

The key never lives in the database: it comes from KB_SECRETS_KEY (the key itself) or
KB_SECRETS_KEY_FILE (a file holding it). The container's entrypoint creates the file on first boot
(/var/lib/agenticwf-kb/secrets/secrets_key). Back it up with the volume: without it, stored
secrets cannot be decrypted.

Fernet (AES-128-CBC + HMAC-SHA256, from `cryptography`): a stolen database dump alone reveals
nothing, and a tampered value fails to decrypt instead of yielding garbage.
"""

from __future__ import annotations

import os
from pathlib import Path


class SecretsUnavailable(Exception):
    """No key configured, or the stored value cannot be decrypted with it."""


class SecretBox:
    def __init__(self, key: bytes):
        from cryptography.fernet import Fernet

        self._fernet = Fernet(key)

    @staticmethod
    def generate_key() -> bytes:
        from cryptography.fernet import Fernet

        return Fernet.generate_key()

    @classmethod
    def from_environment(cls) -> "SecretBox | None":
        key = os.environ.get("KB_SECRETS_KEY")
        path = os.environ.get("KB_SECRETS_KEY_FILE")
        if not key and path and Path(path).is_file():
            key = Path(path).read_text().strip()
        return cls(key.encode()) if key else None

    def encrypt(self, plaintext: str) -> str:
        return self._fernet.encrypt(plaintext.encode()).decode()

    def decrypt(self, token: str) -> str:
        from cryptography.fernet import InvalidToken

        try:
            return self._fernet.decrypt(token.encode()).decode()
        except InvalidToken as exc:
            raise SecretsUnavailable("stored secret cannot be decrypted with the configured key") from exc


def hint(secret: str) -> str:
    """What may be shown of a secret: its last four characters."""
    return "…" + secret[-4:] if len(secret) >= 8 else "…"
