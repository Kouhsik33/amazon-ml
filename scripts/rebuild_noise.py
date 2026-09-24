#!/usr/bin/env python3
"""Re-derive the name/address noise vocabularies from the persisted token stats.

Why not ``drop_rate``: the generator deletes tokens roughly uniformly at ~47%,
so a high drop rate identifies nothing -- every content word has one. Two
signals actually separate filler from identity:

* ``add_rate = added / (kept + added)`` -- does the corruption *inject* this
  token? Only structural words get injected (``the`` 0.91, ``center`` 0.73,
  ``services`` 0.69, ``inc`` 0.16).
* **document frequency** -- legal forms saturate the corpus (``limited`` 23.6%
  of S1 names, ``private`` 19.5%, ``llc`` 16.1%) while content words do not
  (``foods`` 0.51%, ``black`` 0.22%). ``private`` is almost never injected, so
  DF is what catches it.

A token is filler if it is injected, or if it is corpus-saturated *and*
meaningfully unstable. Both thresholds are properties of the measured
distribution, not of any particular country or language.
"""
import json
import sys
from pathlib import Path

CONFIG = Path(__file__).resolve().parents[1] / "configs"
ADD_RATE_MIN = 0.05
DF_FRAC_MIN = 0.02
DROP_RATE_MIN = 0.25
MIN_N = 300
INJECT_ONLY_MIN = 1_000


def rebuild(field: str, n_pairs: int) -> list[str]:
    st = json.loads((CONFIG / f"{field}_token_stats.json").read_text())
    alias = json.loads((CONFIG / f"{field}_alias.json").read_text())
    keep_t, injected, saturated = [], [], []
    for t, (k, d, a, df) in st.items():
        n = k + d
        # Numbers are never filler: a street number, PIN or unit number is the
        # most identity-bearing part of an address. They look "injected" only
        # because the generator adds unit/PMB numbers.
        if t.isdigit() or any(ch.isdigit() for ch in t):
            continue
        if len(t) <= 1:          # initials can identify a short business name
            continue
        # A token the corruption only ever injects (never present in clean S1)
        # has no keep/drop support at all, so the support gate must not hide it.
        if n < MIN_N and a < INJECT_ONLY_MIN:
            continue
        add_rate = a / (k + a) if (k + a) else 1.0
        drop_rate = d / n if n else 0.0
        df_frac = df / n_pairs
        if add_rate >= ADD_RATE_MIN:
            injected.append(t)
        elif df_frac >= DF_FRAC_MIN and drop_rate >= DROP_RATE_MIN:
            saturated.append(t)
        else:
            keep_t.append((t, drop_rate, add_rate, df_frac))
    noise = sorted(set(injected) | set(saturated) - set(alias))
    print(f"[{field}] injected={len(injected)} saturated={len(saturated)} "
          f"-> {len(noise)} noise tokens")
    print(f"  injected : {sorted(injected)[:40]}")
    print(f"  saturated: {sorted(saturated)}")
    # sanity: content words that must survive
    return noise


def rebuild_addr(n_pairs: int) -> list[str]:
    """Addresses use post-alias instability, not injection rate.

    The two fields need different criteria for a concrete reason. Names have no
    alias table, so ``drop_rate`` is swamped by the generator's uniform ~47%
    token deletion and only ``add_rate`` separates filler from content. Address
    tokens *are* aliased first (``mh``/``maharashtra``/``महाराष्ट्र`` collapse
    to one token), so a token that still looks unstable afterwards is genuinely
    filler. Applying the injection rule here instead would flag every city
    name, because the generator *substitutes* cities (``Phoenix``/
    ``Sunnyslope``) rather than injecting filler -- and cities, while
    unreliable, are real signal worth keeping in the comparison views.
    """
    st = json.loads((CONFIG / "addr_token_stats.json").read_text())
    alias = json.loads((CONFIG / "addr_alias.json").read_text())
    noise = set()
    for t, (k, d, a, df) in st.items():
        if any(ch.isdigit() for ch in t) or len(t) <= 1:
            continue
        n = k + d
        if n >= 300 and df >= 1_500 and d / n >= 0.45:
            noise.add(t)
        elif a >= 1_500 and df < a * 0.20:
            noise.add(t)
    noise -= set(alias)
    print(f"[addr] {len(noise)} noise tokens (post-alias instability)")
    print(f"  {sorted(noise)}")
    return sorted(noise)


if __name__ == "__main__":
    n_pairs = 1_500_000
    noise = rebuild("name", n_pairs)
    (CONFIG / "name_noise.json").write_text(json.dumps(noise, ensure_ascii=False, indent=0))
    noise = rebuild_addr(n_pairs)
    (CONFIG / "addr_noise.json").write_text(json.dumps(noise, ensure_ascii=False, indent=0))
    nn = set(json.loads((CONFIG / "name_noise.json").read_text()))
    must_keep = ["black", "foods", "good", "sky", "investment", "unique", "galaxy",
                 "lotus", "laxmi", "bombay", "sunrise", "infotech", "power"]
    must_drop = ["limited", "private", "inc", "llc", "llp", "the", "center",
                 "services", "com", "dba", "formerly", "corp", "co"]
    bad_keep = [t for t in must_keep if t in nn]
    bad_drop = [t for t in must_drop if t not in nn]
    print(f"\ncontent words wrongly marked noise : {bad_keep or 'none'}")
    print(f"legal/filler words NOT marked noise: {bad_drop or 'none'}")
    sys.exit(1 if (bad_keep or bad_drop) else 0)
