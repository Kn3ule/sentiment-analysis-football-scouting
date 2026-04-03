"""
03_translate.py
===============
Translates non-German comments in to_label_v2.csv to German.

Usage:
    uv run python notebooks/03_translate.py --lang EN
    uv run python notebooks/03_translate.py --lang ES
    uv run python notebooks/03_translate.py --lang FR

Each run produces:
    data/labeled/translations_<LANG>.csv
    columns: index (original row index), comment_de

Merge all three outputs with 01_build_corpus results using:
    uv run python notebooks/04_merge_translations.py
"""

import argparse
import os
import pathlib
import time
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

BASE      = pathlib.Path(__file__).parent.parent
INPUT_CSV = BASE / "data" / "labeled" / "to_label_v2.csv"
OUT_DIR   = BASE / "data" / "labeled"
BATCH_LOG = 50  # print progress every N comments

SYSTEM_PROMPT = (
    "Du bist ein professioneller Übersetzer. "
    "Übersetze den folgenden Kommentar eines Fußball-Fans ins Deutsche. "
    "Gib NUR die Übersetzung zurück — keinen Zusatztext, keine Erklärung. "
    "Behalte Eigennamen, Spielernamen und Instagram-Handles unverändert. "
    "Behalte Emojis."
)


def translate_comment(client: OpenAI, comment: str) -> str:
    """Translate a single comment to German via OpenAI."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=256,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": comment},
        ],
    )
    return response.choices[0].message.content.strip()


def main(lang: str) -> None:
    df = pd.read_csv(INPUT_CSV)
    subset = df[df["language"] == lang].copy()
    print(f"Translating {len(subset)} {lang} comments to German...")

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    results = []
    for i, (idx, row) in enumerate(subset.iterrows()):
        comment = str(row["comment"])

        # Skip very short comments — not worth translating
        if len(comment.strip()) < 5:
            translation = comment
        else:
            try:
                translation = translate_comment(client, comment)
            except Exception as e:
                print(f"  [WARN] row {idx}: {e} — keeping original")
                translation = comment

        results.append({"original_index": idx, "comment_de": translation})

        if (i + 1) % BATCH_LOG == 0:
            print(f"  {i + 1}/{len(subset)} done")
        # Small delay to avoid rate limits
        time.sleep(0.1)

    out_df = pd.DataFrame(results)
    out_path = OUT_DIR / f"translations_{lang}.csv"
    out_df.to_csv(out_path, index=False)
    print(f"✓ Saved {len(out_df)} translations → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", required=True, choices=["EN", "ES", "FR"])
    args = parser.parse_args()
    main(args.lang)
