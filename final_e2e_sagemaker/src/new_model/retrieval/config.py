"""Retrieval configuration for the new entity-resolution system.

Every knob that the Approach2 baseline hard-codes is a field here, so a
comparison can change exactly one variable and leave the rest provably fixed.

Baseline values reproduce Approach2 (common.py:12-16, blocking.py:16-17).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class RetrievalConfig:
    name: str = "baseline"

    # --- DF gates (shares of the country corpus, NOT absolute thresholds) ---
    # Approach2 computes max_df = share * n_all per country (blocking.py:72),
    # so a share is the portable parameter; an absolute value is country-specific.
    name_generic_share: float = 0.05
    addr_generic_share: float = 0.03
    # Optional absolute override for the address gate. None => use the share.
    # 400000 was the experimental value measured on the US corpus.
    addr_max_df_abs: float | None = None

    # --- key families ---
    name_core_top: int = 6      # rarest name tokens paired into 'b' keys
    addr_core_top: int = 8      # rarest addr tokens paired into 'c' keys

    # --- index ---
    cap_frac: float = 0.0002    # posting-list cap as a fraction of |B|
    min_cap: int = 20

    # --- search ---
    pool: int = 600             # candidates rescored by the reranker
    top_k: int = 15             # kept per S1 per source
    rerank: str = "cos"         # 'cos' or 'jac'

    def addr_max_df(self, n_all: int) -> float:
        """Resolve the address gate for a country of size n_all."""
        if self.addr_max_df_abs is not None:
            return float(self.addr_max_df_abs)
        return self.addr_generic_share * n_all

    def name_max_df(self, n_all: int) -> float:
        return self.name_generic_share * n_all

    def as_dict(self) -> dict:
        return asdict(self)


BASELINE = RetrievalConfig(name="approach2_baseline")
EXP_ADDR_400K = RetrievalConfig(name="addr_max_df_400k", addr_max_df_abs=400_000)
