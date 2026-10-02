import json
from pathlib import Path

import pytest

from forge.commands.artifacts import verify
from forge.commands.reproduce import reproduce, summary, verify_manuscript_rows
from forge.core.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[1]


def test_reference_submission_and_source_identity() -> None:
    assert (
        sha256_file(ROOT / "paper/submission.pdf")
        == "7ea7cf2e51922e5e58cd6659dc183409e465ede2eb1586fc8d0ec39935c4fd53"
    )
    assert (
        sha256_file(ROOT / "paper/source/main.tex")
        == "36aba6ea556cc6b62a436d3b2df48d79e39813773f549592df053a163f78fafa"
    )


def test_seed_summary_uses_sample_sd_and_preserves_undefined() -> None:
    assert summary([1, 2, 3]) == r"$2.0\pm1.0$"
    assert summary([None, None, None]) == "N/E"
    with pytest.raises(ValueError):
        summary([1, None, 3])


@pytest.mark.artifacts
def test_every_numerical_table_row_matches_exact_manuscript(tmp_path: Path) -> None:
    for name in ("paper-model-v1", "submission19337-evidence-v1"):
        report = verify(ROOT, ROOT / f"manifests/{name}.json")
        missing = [row for row in report["files"] if row["status"] == "missing"]
        mismatched = [row for row in report["files"] if row["status"] == "mismatch"]
        assert not mismatched, mismatched
        if missing:
            pytest.skip(f"fetch {name} to run frozen-table replay")
    output = tmp_path / "tables"
    reproduce(ROOT, output)
    assert verify_manuscript_rows(ROOT, output) == 89
    assert json.loads((output / "receipt.json").read_text())["status"] == "pass"
