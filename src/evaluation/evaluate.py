#!/usr/bin/env python3
"""CLI evaluator: score a matching_results-style TSV against a ground-truth TSV.

    python -m evaluation.evaluate --gt val_ground_truth.tsv \
                                  --pred output/matching_results.tsv [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.metrics import evaluate_predictions, load_id_list_tsv  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Score predictions with macro F_0.5.")
    ap.add_argument("--gt", required=True, help="ground-truth TSV (source1_entity_id, matched_entity_ids)")
    ap.add_argument("--pred", required=True, help="predictions TSV (source1_entity_id, matched_entity_ids)")
    ap.add_argument("--json", default=None, help="also write metrics as JSON here")
    args = ap.parse_args()

    gt = load_id_list_tsv(args.gt)
    pred = load_id_list_tsv(args.pred)

    missing = len(set(gt) - set(pred))
    extra = len(set(pred) - set(gt))
    if missing:
        print(f"WARNING: {missing} ground-truth entities absent from predictions "
              f"(scored as empty predictions).", file=sys.stderr)
    if extra:
        print(f"WARNING: {extra} predicted entities not in the ground truth (ignored).",
              file=sys.stderr)

    res = evaluate_predictions(gt, pred)
    print(res)
    if args.json:
        Path(args.json).write_text(json.dumps(res.as_dict(), indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
