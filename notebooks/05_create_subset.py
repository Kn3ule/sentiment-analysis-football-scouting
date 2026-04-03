"""
05_create_subset.py
===================
Creates a per-language class-balanced evaluation subset from the fully labeled
annotation pool (to_label_v2_human_claude.csv).

Sampling strategy
-----------------
1. Keep only rows with a valid label (POSITIVE / NEUTRAL / NEGATIVE).
2. Per language, cap each class at the size of the smallest class in that
   language (ensures perfect balance within each language).
3. Within each (language, label) group, select comments using player-stratified
   round-robin sampling to maximise player diversity.

Targets (derived from actual data minimums)
-------------------------------------------
  DE: 97 per class  →  291 total
  EN: 65 per class  →  195 total
  ES: 60 per class  →  180 total
  FR: 28 per class  →   84 total
  ────────────────────────────────
  Grand total: 750  |  250 per class

Output
------
  data/labeled/subset_balanced_750.csv
    columns: comment_id, language, club, player, comment, comment_de, label

Usage:
  uv run python notebooks/05_create_subset.py
"""

import random
import pathlib
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE       = pathlib.Path(__file__).parent.parent
INPUT_CSV  = BASE / "data" / "labeled" / "to_label_v2_human_claude.csv"
OUTPUT_CSV = BASE / "data" / "labeled" / "subset_balanced_750.csv"

RANDOM_SEED  = 42
VALID_LABELS = ["POSITIVE", "NEUTRAL", "NEGATIVE"]


# ── Helpers ────────────────────────────────────────────────────────────────────

def round_robin_sample(group: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """
    Player-stratified round-robin: cycle through players, pick 1 comment per
    player per round until n comments are selected.
    """
    buckets = {
        player: pdf.sample(frac=1, random_state=seed).index.tolist()
        for player, pdf in group.groupby("player")
    }
    player_keys = list(buckets.keys())
    random.shuffle(player_keys)

    selected = []
    pos = {p: 0 for p in player_keys}

    while len(selected) < n:
        made_progress = False
        for p in player_keys:
            if len(selected) >= n:
                break
            if pos[p] < len(buckets[p]):
                selected.append(buckets[p][pos[p]])
                pos[p] += 1
                made_progress = True
        if not made_progress:
            break

    return group.loc[selected]


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    random.seed(RANDOM_SEED)

    df = pd.read_csv(INPUT_CSV)
    df = df[df["label"].isin(VALID_LABELS)].copy()

    print(f"Labeled comments available: {len(df)}")
    print(df.groupby(["language", "label"]).size().unstack(fill_value=0)
          .reindex(["DE", "EN", "ES", "FR"]))

    # Per-language minimum class size → target per class
    targets = {
        lang: int(df[df["language"] == lang].groupby("label").size().min())
        for lang in ["DE", "EN", "ES", "FR"]
    }

    print("\nPer-language targets (smallest class):")
    for lang, n in targets.items():
        print(f"  {lang}: {n} per class → {n * 3} total")
    print(f"  Grand total : {sum(n * 3 for n in targets.values())}")
    print(f"  Per label   : {sum(targets.values())}")

    # ── Sampling ──────────────────────────────────────────────────────────────
    parts = []
    for lang, target in targets.items():
        for label in VALID_LABELS:
            group = df[(df["language"] == lang) & (df["label"] == label)]
            sampled = round_robin_sample(group, target, seed=RANDOM_SEED)
            parts.append(sampled)

    result = pd.concat(parts, ignore_index=True)

    # ── comment_id ────────────────────────────────────────────────────────────
    counters = {lang: 1 for lang in ["DE", "EN", "ES", "FR"]}
    comment_ids = []
    for _, row in result.iterrows():
        lang = row["language"]
        comment_ids.append(f"{lang}_{counters[lang]:04d}")
        counters[lang] += 1
    result.insert(0, "comment_id", comment_ids)

    result = result[["comment_id", "language", "club", "player", "comment", "comment_de", "label"]]

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\nFinal distribution:")
    print(result.groupby(["language", "label"]).size().unstack(fill_value=0)
          .reindex(["DE", "EN", "ES", "FR"]))
    print(f"\nPer label:\n{result['label'].value_counts().sort_index()}")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✓ Saved {len(result)} comments → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
