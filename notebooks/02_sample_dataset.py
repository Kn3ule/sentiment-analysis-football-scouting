"""
02_sample_dataset.py
====================
Stratified sampling of the evaluation dataset from player_corpus.csv.

Sampling strategy
-----------------
The natural class distribution in the corpus is approximately:
  65 % POSITIVE / 14 % NEGATIVE / 21 % NEUTRAL

Uniform random sampling would yield ~35 NEGATIVE instances per language
(249 × 0.14), which is insufficient for reliable per-class evaluation.
To construct a balanced evaluation benchmark, the following strategy is used:

  NEGATIVE pool  — keyword-guided pre-filtering
      Language-specific negative-sentiment keywords are applied to the
      player-filtered corpus. Keywords were derived from the vocabulary
      of the 200-comment pilot dataset (labeled_data_50.csv), retaining
      terms with empirical precision ≥ 0.75 on NEGATIVE vs. POSITIVE.
      The filtered pool is capped at NEG_CANDIDATES per language
      (= NEG_TARGET × OVERSAMPLE_FACTOR) to account for false positives.

  POSITIVE / NEUTRAL pool  — uniform random sampling
      All comments NOT matching any negative keyword are pooled and randomly
      sampled. A larger oversample (RANDOM_CANDIDATES) is drawn because the
      NEUTRAL class (~21 %) requires a wide random pool to reach the target
      of ~83 instances.

Targets
-------
  NEG_TARGET        = 83 per language
  POS_TARGET        = 83 per language
  NEU_TARGET        = 83 per language
  ─────────────────────────────────────
  Total target      = 249 per language × 4 languages = 996

Candidate counts (pre-labeling)
  NEG_CANDIDATES    = 125 per language  (83 × 1.5, accounts for ~75% keyword precision)
  RANDOM_CANDIDATES = 350 per language  (for ~83 POS + ~83 NEU; NEU requires ~395 at 21%
                                         natural rate → FR may require all available)

  Total candidates  ≈ 475 per language × 4 = ~1,900 comments to label
  (FR may be smaller if pool is exhausted)

Outputs
-------
  data/labeled/to_label_v2.csv
    columns: language, club, player, comment, sampling_pool

  sampling_pool values:
    "neg_candidates"    — keyword-filtered, likely NEGATIVE
    "random_pos_neu"    — random sample, likely POSITIVE or NEUTRAL

Usage:
  python notebooks/02_sample_dataset.py
  uv run python notebooks/02_sample_dataset.py
"""

import re
import pathlib
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = pathlib.Path(__file__).parent.parent
CORPUS_CSV = BASE / "data" / "preprocessing" / "player_corpus.csv"
OUTPUT_CSV = BASE / "data" / "labeled" / "to_label_v2.csv"

# ── Sampling parameters ────────────────────────────────────────────────────────
RANDOM_SEED      = 42       # fixed for reproducibility
NEG_TARGET       = 83       # target NEGATIVE labels per language
POS_TARGET       = 83       # target POSITIVE labels per language
NEU_TARGET       = 83       # target NEUTRAL labels per language
OVERSAMPLE_NEG   = 1.5      # neg_candidates = NEG_TARGET × OVERSAMPLE_NEG
RANDOM_CANDIDATES = 350     # random pool size per language (POS + NEU)
                             # FR: capped at available pool size

NEG_CANDIDATES = int(NEG_TARGET * OVERSAMPLE_NEG)  # = 125

# ── Negative-sentiment keyword lists ──────────────────────────────────────────
# Derived from the vocabulary of the 200-comment pilot dataset (labeled_data_50.csv).
# Precision ≥ 0.75 on NEGATIVE vs. POSITIVE class (empirically validated).
# See Appendix A of the paper for the full keyword lists and validation procedure.

NEG_KEYWORDS: dict[str, list[str]] = {
    "DE": [
        # Performance
        "schlecht", "schwach", "katastrophal", "katastrophe", "schrecklich",
        "peinlich", "grauenhaft", "miserabel", "erbärmlich", "unterirdisch",
        "blamage", "blamiert", "blamiere", "enttäuschend", "enttäuscht",
        "versagt", "versagen", "patzer", "fehler",
        # Demand to remove/sell
        "feuern", "rauswerfen", "raus damit", "weg damit", "muss weg",
        "verkaufen", "abgeben", "nicht tragbar", "nicht spielen lassen",
        "raus", "absteiger", "abstieg",
        # Character / quality
        "overrated", "inkompetent", "kotzt mich an", "schäm dich",
        "scheiße", "mist", "nie mehr", "aufhören",
    ],
    "EN": [
        # Direct insults / quality
        "useless", "terrible", "awful", "pathetic", "hopeless", "dreadful",
        "shocking", "embarrassing", "shameful", "disgrace", "disgraceful",
        "garbage", "rubbish", "trash", "disaster", "liability", "fraud",
        "overrated", "flop", "fraud", "bottler", "donkey",
        # Demand to drop / sell
        "sell him", "get rid", "ship him out", "bench him", "drop him",
        "should be dropped", "should leave", "should go", "needs to go",
        "not good enough", "not worthy", "not worth",
        # Failure / performance
        "disappointing", "disappoints", "wasted", "cost us", "let us down",
        "poor performance", "worst", "failure", "finished", "washed up",
        "can't defend", "can't score", "mistake", "error", "blunder",
        "clown", "waste of", "never should",
    ],
    "ES": [
        # Direct quality
        "pésimo", "malísimo", "horrible", "terrible", "desastre",
        "fracaso", "vergüenza", "vergonzoso", "inútil", "incompetente",
        "nefasto", "fatal", "lamentable", "espantoso", "penoso",
        "ridículo", "ridícula", "flojo", "floja",
        # Demand to remove/sell
        "fuera", "vender", "se vaya", "que se vaya", "no sirve",
        "debería salir", "hay que venderlo", "tiene que irse",
        "a la grada", "al banquillo", "no merece",
        # Failure / performance
        "decepcionante", "decepción", "decepciona", "culpa", "culpable",
        "falla", "fallo", "error", "equivoca", "mal partido",
        "muy malo", "muy mala", "no está al nivel",
    ],
    "FR": [
        # Direct quality
        "nul", "nulle", "nuls", "nulles", "honteux", "honteuse",
        "lamentable", "inutile", "incompétent", "incompétente",
        "médiocre", "catastrophe", "catastrophique", "désastreux",
        "déplorable", "scandaleux", "minable", "pitoyable", "pathétique",
        "mauvais", "mauvaise", "horrible", "terrible",
        # Demand to remove/sell
        "vendre", "partir", "doit partir", "qu'il parte", "viré",
        "limoger", "se casser", "pas le niveau", "n'a pas le niveau",
        "à vendre", "dehors",
        # Failure / performance
        "décevant", "décevante", "déception", "raté", "ratée",
        "honte", "ridicule", "zéro", "nul à",
    ],
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def build_neg_pattern(keywords: list[str]) -> re.Pattern:
    """Case-insensitive OR-pattern for all negative keywords."""
    escaped = [re.escape(kw) for kw in keywords]
    return re.compile("|".join(escaped), re.I | re.UNICODE)


def sample_language(
    df_lang: pd.DataFrame,
    language: str,
    rng,
) -> pd.DataFrame:
    """
    Sample NEG_CANDIDATES + RANDOM_CANDIDATES from one language's corpus.
    Returns a DataFrame with a 'sampling_pool' column.
    """
    pattern = build_neg_pattern(NEG_KEYWORDS[language])

    neg_mask = df_lang["comment"].str.contains(pattern, regex=True, na=False)
    neg_pool    = df_lang[neg_mask].copy()
    random_pool = df_lang[~neg_mask].copy()

    # ── NEGATIVE candidates ───────────────────────────────────────────────────
    n_neg = min(NEG_CANDIDATES, len(neg_pool))
    sampled_neg = neg_pool.sample(n=n_neg, random_state=rng).copy()
    sampled_neg["sampling_pool"] = "neg_candidates"

    # ── POSITIVE / NEUTRAL candidates ─────────────────────────────────────────
    # FR pool is often small — take everything available up to RANDOM_CANDIDATES.
    n_random = min(RANDOM_CANDIDATES, len(random_pool))
    sampled_random = random_pool.sample(n=n_random, random_state=rng).copy()
    sampled_random["sampling_pool"] = "random_pos_neu"

    combined = pd.concat([sampled_neg, sampled_random], ignore_index=True)

    print(
        f"  {language}  neg_pool={len(neg_pool):4d} → sampled {n_neg:3d}  |  "
        f"random_pool={len(random_pool):4d} → sampled {n_random:3d}  |  "
        f"total={len(combined):3d}"
    )
    return combined


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    if not CORPUS_CSV.exists():
        raise FileNotFoundError(
            f"{CORPUS_CSV} not found.\n"
            "Run 01_build_corpus.py first."
        )

    df = pd.read_csv(CORPUS_CSV)
    print(f"Loaded corpus: {len(df):,} comments\n")

    parts = []
    print("Sampling per language:")
    print(f"  Targets: {NEG_TARGET} NEG + {POS_TARGET} POS + {NEU_TARGET} NEU = "
          f"{NEG_TARGET + POS_TARGET + NEU_TARGET} per language")
    print(f"  Candidates: {NEG_CANDIDATES} neg + {RANDOM_CANDIDATES} random\n")

    for language in ["DE", "EN", "ES", "FR"]:
        df_lang = df[df["language"] == language].copy()
        if len(df_lang) == 0:
            print(f"  {language}  WARNING: no comments in corpus — skipping")
            continue
        sampled = sample_language(df_lang, language, rng=RANDOM_SEED)
        parts.append(sampled)

    result = pd.concat(parts, ignore_index=True)

    # Shuffle final output (don't reveal sampling structure to annotator)
    result = result.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    # Keep only the columns needed for labeling
    result = result[["language", "club", "player", "comment", "sampling_pool"]]

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n── Sampling summary ──────────────────────────────────────────")
    print(f"Total candidates to label: {len(result):,}")
    print("\nPer language:")
    print(result.groupby(["language", "sampling_pool"]).size().unstack(fill_value=0).to_string())
    print("\nNote:")
    print(f"  After human labeling, select up to {NEG_TARGET} NEGATIVE + "
          f"{POS_TARGET} POSITIVE + {NEU_TARGET} NEUTRAL per language")
    print(f"  for the final evaluation benchmark of ~"
          f"{(NEG_TARGET + POS_TARGET + NEU_TARGET) * 4} comments.")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✓ Saved to {OUTPUT_CSV}  ({len(result):,} rows)")
    print(  "  Column 'sampling_pool' is metadata only — do NOT use it as a label.")


if __name__ == "__main__":
    main()
