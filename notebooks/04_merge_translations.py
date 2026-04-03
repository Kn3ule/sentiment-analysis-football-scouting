"""
04_merge_translations.py
========================
Merges translation files (translations_EN/ES/FR.csv) back into to_label_v2.csv,
adding a 'comment_de' column.

German comments keep their original text in comment_de.

Usage:
    uv run python notebooks/04_merge_translations.py

Output:
    data/labeled/to_label_v2.csv   (updated in-place with comment_de column)
"""

import pathlib
import pandas as pd

BASE      = pathlib.Path(__file__).parent.parent
LABEL_CSV = BASE / "data" / "labeled" / "to_label_v2.csv"
OUT_DIR   = BASE / "data" / "labeled"


def main() -> None:
    df = pd.read_csv(LABEL_CSV)

    # German comments: comment_de = original comment
    df["comment_de"] = df.apply(
        lambda r: r["comment"] if r["language"] == "DE" else None, axis=1
    )

    # Merge translations for EN, ES, FR
    missing = []
    for lang in ["EN", "ES", "FR"]:
        trans_path = OUT_DIR / f"translations_{lang}.csv"
        if not trans_path.exists():
            print(f"[WARN] {trans_path.name} not found — {lang} left untranslated")
            missing.append(lang)
            continue
        trans = pd.read_csv(trans_path).set_index("original_index")
        mask = df["language"] == lang
        df.loc[mask, "comment_de"] = df.loc[mask].index.map(
            trans["comment_de"]
        )
        print(f"  {lang}: {mask.sum()} translations merged")

    # Fallback: untranslated comments keep original text
    df["comment_de"] = df["comment_de"].fillna(df["comment"])

    df.to_csv(LABEL_CSV, index=False)
    print(f"\n✓ Updated {LABEL_CSV}  ({len(df)} rows, comment_de column added)")

    if missing:
        print(f"[NOTE] Missing translations for: {missing} — originals kept")


if __name__ == "__main__":
    main()
