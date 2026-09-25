"""Step 1: learn the non-Latin -> Latin token lexicon from the TRAIN labels (no external resources)."""
import json
import sys

import pandas as pd

from common import DATA_DIR, WORK_DIR, read_split
from normalize import learn_lexicon, norm_name


def main():
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    d = read_split("train")
    gt = pd.read_csv(DATA_DIR / "train" / "train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
    gt = gt[gt.matched_entity_ids != ""]
    pairs = gt.assign(m=gt.matched_entity_ids.str.split(",")).explode("m")[["source1_entity_id", "m"]]
    b_names = pd.concat([d["s2"].set_index("entity_id").business_name, d["s3"].set_index("entity_id").business_name])
    s1_names = d["s1"].set_index("entity_id").business_name
    pairs["nb"] = pairs.m.map(b_names)
    pairs = pairs[~pairs.nb.fillna("").map(str.isascii)]
    n1 = [norm_name(x) for x in pairs.source1_entity_id.map(s1_names)]
    nb = [norm_name(x) for x in pairs.nb]
    print("non-Latin training pairs:", len(pairs))
    lex = learn_lexicon(n1, nb, min_j=0.0)
    print("lexicon size:", len(lex))
    with open(WORK_DIR / "lexicon.json", "w", encoding="utf-8") as f:
        json.dump(lex, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
