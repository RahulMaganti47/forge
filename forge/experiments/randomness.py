"""Stable keyed random streams shared by the matched production comparisons."""

from __future__ import annotations

import hashlib


def production_seed(base: int, *parts: object) -> int:
    """Preserve the frozen pipe-delimited 63-bit production seed contract."""

    digest = hashlib.sha256("|".join((str(base), *(str(value) for value in parts))).encode())
    return int.from_bytes(digest.digest()[:8], "little") % (2**63 - 1)


__all__ = ["production_seed"]
