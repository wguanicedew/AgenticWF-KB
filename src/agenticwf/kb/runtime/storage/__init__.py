"""
File storage for originals, parsed markdown and extracted images (docs/ingestion.md).

Local only in phase 1 (`[storage] backend = local`, `root`, default $KB_DATA_ROOT/files); S3 is
later. Keys look like `<owner-scope>/<kb id>/<knowledge id>/<file>`; a key can never leave the
root.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import tempfile
from pathlib import Path

from agenticwf.kb.core.common.config import config_get, config_has_option


class StorageError(Exception):
    pass


def owner_scope(subject: str) -> str:
    """A file-system-safe directory name for an owner: readable part + short hash (no collisions)."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", subject)[:48].strip("._") or "owner"
    return f"{safe}-{hashlib.sha256(subject.encode()).hexdigest()[:8]}"


def safe_name(name: str) -> str:
    name = os.path.basename(name.replace("\\", "/")) or "file"
    name = re.sub(r"[\x00-\x1f/]+", "_", name).strip(". ") or "file"
    return name[:200]


class LocalStorage:
    def __init__(self, root: str | Path | None = None):
        if root is None:
            if config_has_option("storage", "root"):
                root = config_get("storage", "root")
            else:
                root = Path(os.environ.get("KB_DATA_ROOT", "/var/lib/agenticwf-kb")) / "files"
        self.root = Path(root).resolve()

    def path(self, key: str) -> Path:
        target = (self.root / key).resolve()
        if target != self.root and self.root not in target.parents:
            raise StorageError(f"key escapes the storage root: {key!r}")
        return target

    def save(self, key: str, data: bytes) -> int:
        """Write atomically (temporary file + rename); returns the size."""
        target = self.path(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=".tmp-")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
            os.replace(tmp, target)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        return len(data)

    def read(self, key: str) -> bytes:
        try:
            return self.path(key).read_bytes()
        except FileNotFoundError as exc:
            raise StorageError(f"missing file {key!r}") from exc

    def read_text(self, key: str) -> str:
        return self.read(key).decode("utf-8")

    def exists(self, key: str) -> bool:
        return self.path(key).is_file()

    def delete(self, key: str) -> None:
        self.path(key).unlink(missing_ok=True)

    def keys(self, prefix: str) -> list[str]:
        """The keys of the files under `prefix`."""
        target = self.path(prefix)
        if not target.is_dir():
            return []
        return sorted(str(p.relative_to(self.root)) for p in target.rglob("*") if p.is_file())

    def delete_prefix(self, prefix: str) -> None:
        target = self.path(prefix)
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
