import json
from pathlib import Path

import pytest

from forge.commands.artifacts import verify
from forge.commands.reproduce import reproduce, summary, verify_table_rows
from forge.core.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[1]


def test_numerical_reference_identity() -> None:
    assert sha256_file(ROOT / "data/table_reference.json") == (
        "aac0776f827eb99b5d07441e51dd8558187591d38fb89069426f4a5edcb860f0"
    )


def test_seed_summary_uses_sample_sd_and_preserves_undefined() -> None:
    assert summary([1, 2, 3]) == r"$2.0\pm1.0$"
    assert summary([None, None, None]) == "N/E"
    with pytest.raises(ValueError):
        summary([1, None, 3])


@pytest.mark.artifacts
def test_every_numerical_table_row_matches_saved_reference(tmp_path: Path) -> None:
    for name in ("paper-model-v1", "evidence-v1"):
        report = verify(ROOT, ROOT / f"manifests/{name}.json")
        missing = [row for row in report["files"] if row["status"] == "missing"]
        mismatched = [row for row in report["files"] if row["status"] == "mismatch"]
        assert not mismatched, mismatched
        if missing:
            pytest.skip(f"fetch {name} to run frozen-table replay")
    output = tmp_path / "tables"
    reproduce(ROOT, output)
    assert verify_table_rows(ROOT, output) == 89
    assert json.loads((output / "receipt.json").read_text())["status"] == "pass"
