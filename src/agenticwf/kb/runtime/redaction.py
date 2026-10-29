"""
Redaction rules (docs/ingestion.md, Redaction).

    Redactor.from_body({"terms": [...], "patterns": [...], "replacement": "[REDACTED]",
                        "case_sensitive": false})

`terms` are literal strings (whitespace in a term matches any run of whitespace, so
"DE12 3456" also finds "DE12  3456"); `patterns` are regular expressions. A rule that can match
the empty string is refused. `scrub` applies the rules to strings anywhere inside JSON-like
values (citations, chunk metadata, OCR text). Previews show where a match is with the match
itself replaced, so a dry run never echoes the secret.
"""

from __future__ import annotations

import re

from agenticwf.kb.runtime.errors import Invalid

MAX_RULES = 50
MAX_RULE_LENGTH = 500
DEFAULT_REPLACEMENT = "[REDACTED]"
PREVIEW_CONTEXT = 40


def _strings(value) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        raise Invalid("terms and patterns must be lists of strings")
    return [str(v) for v in value if str(v).strip()]


class Redactor:
    def __init__(self, terms: list[str], patterns: list[str], replacement: str = DEFAULT_REPLACEMENT,
                 case_sensitive: bool = False):
        rules = [r"\s+".join(re.escape(part) for part in term.split()) for term in terms] + list(patterns)
        if not rules:
            raise Invalid("give at least one of terms or patterns")
        if len(rules) > MAX_RULES or any(len(r) > MAX_RULE_LENGTH for r in terms + patterns):
            raise Invalid(f"at most {MAX_RULES} rules of at most {MAX_RULE_LENGTH} characters")
        try:
            self.regex = re.compile("|".join(f"(?:{r})" for r in rules), 0 if case_sensitive else re.IGNORECASE)
        except re.error as exc:
            raise Invalid(f"invalid pattern: {exc}") from exc
        if self.regex.fullmatch(""):
            raise Invalid("a pattern must not match the empty string")
        if replacement and self.regex.search(replacement):
            raise Invalid("the replacement must not match the rules")
        self.replacement = replacement

    @classmethod
    def from_body(cls, body: dict) -> "Redactor":
        replacement = body.get("replacement")
        return cls(_strings(body.get("terms")), _strings(body.get("patterns")),
                   DEFAULT_REPLACEMENT if replacement is None else str(replacement), bool(body.get("case_sensitive")))

    def count(self, text: str | None) -> int:
        return sum(1 for _ in self.regex.finditer(text)) if text else 0

    def apply(self, text: str | None) -> tuple[str, int]:
        if not text:
            return text or "", 0
        return self.regex.subn(lambda _: self.replacement, text)

    def scrub(self, value):
        """`value` with every string inside it redacted (dicts and lists are copied)."""
        if isinstance(value, str):
            return self.apply(value)[0]
        if isinstance(value, dict):
            return {k: self.scrub(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self.scrub(v) for v in value]
        return value

    def previews(self, text: str | None, limit: int = 5) -> list[str]:
        out = []
        for match in self.regex.finditer(text or ""):
            start, end = max(0, match.start() - PREVIEW_CONTEXT), min(len(text), match.end() + PREVIEW_CONTEXT)
            before, after = self.apply(text[start:match.start()])[0], self.apply(text[match.end():end])[0]
            out.append(("…" if start else "") + " ".join(f"{before}{self.replacement or '▮'}{after}".split())
                       + ("…" if end < len(text) else ""))
            if len(out) >= limit:
                break
        return out
