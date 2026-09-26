"""Training corpora, split contracts, and source-balanced sampling policies.

The public API is intentionally small.  Corpus construction owns data identity and split policy;
model code consumes its artifacts and must not reach into corpus implementation helpers.
"""

from forge.corpus.phase1 import Phase1DataError, freeze_phase1_data_contract

__all__ = ["Phase1DataError", "freeze_phase1_data_contract"]
