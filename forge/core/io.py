"""Deterministic serialization and atomic writes.

CSV uses LF line endings and gzip uses a zero timestamp to preserve recorded
artifact hashes. JSON uses sorted keys and rejects nonfinite values.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import os
import tempfile
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------------- JSON


def stable_json(value: Any) -> str:
    """Serialize JSON with sorted keys, compact formatting, and no nonfinite values."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def stable_json_bytes(value: Any) -> bytes:
    return stable_json(value).encode()


def pretty_json_bytes(value: Any) -> bytes:
    """Encode sorted, indented JSON with a trailing newline for result files."""
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


# --------------------------------------------------------------------------------- CSV


def csv_bytes(rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> bytes:
    """Encode CSV rows with LF line endings to preserve ledger hashes."""
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode()


def gzip_bytes(payload: bytes) -> bytes:
    """Gzip with the timestamp zeroed, so identical input always gives identical output."""
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", mtime=0) as compressed:
        compressed.write(payload)
    return output.getvalue()


def csv_gz_bytes(rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> bytes:
    return gzip_bytes(csv_bytes(rows, fieldnames))


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read CSV rows as string dictionaries, handling gzip by extension.

    Callers parse numeric fields.
    """
    if path.suffix == ".gz":
        with gzip.open(path, "rt", newline="") as handle:
            return list(csv.DictReader(handle))
    with path.open("rt", newline="") as handle:
        return list(csv.DictReader(handle))


def iter_csv(path: Path) -> Iterator[dict[str, str]]:
    """Stream a CSV row by row, for ledgers too large to hold in memory."""
    if path.suffix == ".gz":
        with gzip.open(path, "rt", newline="") as handle:
            yield from csv.DictReader(handle)
    else:
        with path.open("rt", newline="") as handle:
            yield from csv.DictReader(handle)


# --------------------------------------------------------------------------------- JSON lines


def read_json(path: Path) -> Any:
    """Read a JSON document, transparently handling gzip by extension."""
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as handle:
            return json.load(handle)
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[Any]:
    return list(iter_jsonl(path))


def iter_jsonl(path: Path) -> Iterator[Any]:
    """Stream JSON lines, skipping blank lines. Handles gzip by extension."""
    opener = gzip.open(path, "rt") if path.suffix == ".gz" else path.open("rt")
    with opener as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def jsonl_bytes(records: Iterable[Any]) -> bytes:
    """Encode records as newline-delimited JSON, each line deterministically serialized."""
    return "".join(f"{stable_json(record)}\n" for record in records).encode()


# ------------------------------------------------------------------- reading with a domain error
#
# The 182 local `_load_json` and `_read_csv` copies cannot be replaced by the plain readers above,
# because each raises its module's own exception type and callers -- including tests -- catch that
# type. So the error class is a parameter. A survey of all 182 shaped the rest:
#
#   120 distinct raised classes         -> `error` must be supplied by the caller
#   110/122 and 24/60 take a `label`    -> `label` names the input in the message, defaulting to
#                                          the path, which is what the label-less copies effectively do
#   121/122 `_load_json` copies assert  -> requiring a JSON object is the contract, not an extra;
#     the parsed value is an object        that is why this is `read_json_object`, not `read_json`
#   10/60 `_read_csv` copies check      -> `required_fields`, off by default
#     for required columns
#
# The caught sets vary but nest: `FileNotFoundError` is an `OSError`, and both `JSONDecodeError`
# and `UnicodeDecodeError` are `ValueError`s. Each function catches the union of what the copies
# caught, listed explicitly rather than as bare `ValueError` so an unrelated failure still escapes.
#
# Messages are core's, not each module's. Reproducing ~20 message templates across 120 error classes
# is not something one function can do, so migrating a consumer changes its message text even though
# the exception type is preserved. That is why migration is a separate reviewed step.


def read_json_object(
    path: Path,
    *,
    error: type[Exception] = ValueError,
    label: str | None = None,
) -> dict[str, Any]:
    """Read a JSON object, raising `error` if it is missing, malformed, or not an object.

    The original exception is preserved as `__cause__`; losing it would turn "the config is
    invalid" into a dead end when the real cause was a truncated file two directories away.
    """
    name = label or str(path)

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key: {key}")
            value[key] = item
        return value

    try:
        if path.suffix == ".gz":
            with gzip.open(path, "rt") as handle:
                value = json.load(handle, object_pairs_hook=reject_duplicates)
        else:
            value = json.loads(path.read_text(), object_pairs_hook=reject_duplicates)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise error(f"{name} could not be read: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise error(f"{name} must contain a JSON object: {path}")
    return value


def read_csv_rows(
    path: Path,
    *,
    error: type[Exception] = ValueError,
    label: str | None = None,
    required_fields: Sequence[str] | None = None,
) -> list[dict[str, str]]:
    """Read CSV rows, raising `error` if the file is unreadable or lacks a required column.

    Missing columns are reported together rather than one at a time, so a caller fixing a schema
    mismatch sees the whole gap in one message.
    """
    name = label or str(path)
    try:
        rows = read_csv(path)
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise error(f"{name} could not be read: {path}: {exc}") from exc
    if required_fields:
        present = set(rows[0]) if rows else set()
        missing = [field for field in required_fields if field not in present]
        if missing:
            raise error(f"{name} is missing fields {sorted(missing)}: {path}")
    return rows


# --------------------------------------------------------------------------------- writing


def atomic_write(path: Path, payload: bytes) -> None:
    """Write bytes so the destination is never observed partially written.

    Writes to a temporary file in the same directory, fsyncs it, then renames over the target --
    rename within a directory is atomic. The repository's engineering contract requires this:
    "calculate first, then write complete artifacts. Do not leave a plausible-looking result after
    an exception." A half-written ledger that still parses is the failure mode this prevents.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_json(path: Path, value: Any, *, pretty: bool = True) -> None:
    atomic_write(path, pretty_json_bytes(value) if pretty else stable_json_bytes(value))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    """Write a CSV, gzipping when the path ends in `.gz`."""
    payload = csv_bytes(rows, fieldnames)
    atomic_write(path, gzip_bytes(payload) if path.suffix == ".gz" else payload)


def write_csv_iter(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    """Stream a deterministic CSV artifact to an atomic destination.

    Large atom-level ledgers should not first materialize millions of row dictionaries or a second
    complete uncompressed byte copy.  Gzip metadata is deterministic and CSV line endings match
    :func:`csv_gz_bytes`; the compressed deflate stream is intentionally produced incrementally.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        with open(temporary, "wb") as raw_handle:
            compressed = (
                gzip.GzipFile(filename="", fileobj=raw_handle, mode="wb", mtime=0)
                if path.suffix == ".gz"
                else None
            )
            binary_handle = compressed if compressed is not None else raw_handle
            text_handle = io.TextIOWrapper(binary_handle, encoding="utf-8", newline="")
            writer = csv.DictWriter(
                text_handle,
                fieldnames=fieldnames,
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
            text_handle.flush()
            text_handle.detach()
            if compressed is not None:
                compressed.close()
            raw_handle.flush()
            os.fsync(raw_handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_jsonl(path: Path, records: Iterable[Any]) -> None:
    payload = jsonl_bytes(records)
    atomic_write(path, gzip_bytes(payload) if path.suffix == ".gz" else payload)


__all__ = [
    "atomic_write",
    "csv_bytes",
    "csv_gz_bytes",
    "gzip_bytes",
    "iter_csv",
    "iter_jsonl",
    "jsonl_bytes",
    "pretty_json_bytes",
    "read_csv",
    "read_csv_rows",
    "read_json",
    "read_json_object",
    "read_jsonl",
    "stable_json",
    "stable_json_bytes",
    "write_csv",
    "write_csv_iter",
    "write_json",
    "write_jsonl",
]
