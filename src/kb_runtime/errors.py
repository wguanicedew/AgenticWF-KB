"""
Errors the runtime raises to its host (the API, the CLI). They carry the API's error code, so
kb_api.errors maps them to the envelope without the runtime knowing about HTTP.
"""

from __future__ import annotations

from typing import Any


class RuntimeError_(Exception):
    code = 1000

    def __init__(self, message: str, details: Any = None):
        super().__init__(message)
        self.message, self.details = message, details


class Invalid(RuntimeError_):
    """A malformed or unacceptable request (1000)."""
    code = 1000


class Forbidden(RuntimeError_):
    """A readable object the principal may not change, or a missing permission (1002)."""
    code = 1002


class NotFound(RuntimeError_):
    """Missing, or not readable by the principal: never say which (1003)."""
    code = 1003


class Conflict(RuntimeError_):
    """A duplicate (1005)."""
    code = 1005


class Busy(RuntimeError_):
    """Too much running for this owner at once (the per-owner thread limit)."""
    code = 1000
