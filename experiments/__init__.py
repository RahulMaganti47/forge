"""Runnable, hash-pinned FORGE experiment applications.

This package owns orchestration and may import :mod:`forge`; the scientific library never imports
from here. Loading the catalog is explicit so importing either package has no scientific-stage side
effects.
"""

from experiments.catalog import load_catalog, resolve_specification, specification_paths

__all__ = ["load_catalog", "resolve_specification", "specification_paths"]
