"""
01_build_corpus.py
==================
Builds the player-filtered corpus from all raw Instagram JSON files.

Reproduces the exact corpus boundaries stated in the paper:
  - 17 clubs across 4 major European leagues
  - 10 players per club (sourced from Transfermarkt, season 2024/25)
  - Language assigned by club channel (channel-language principle)

Supplementary crawls:
  vfb2_instacomments.json   → treated as additional VfB Stuttgart data
  rclens1_instacomments.json → treated as additional RC Lens data

Preprocessing steps (in order):
  1. Load all JSON files for the 17 clubs
  2. Filter Instagram artefacts ("Original-Audio", "•", Reels metadata)
  3. Global deduplication on (comment, club, player)
  4. Player name matching: full name + last name, case-insensitive substring
  5. Minimum length filter: >= 10 characters after stripping
  6. Exclude pilot dataset (labeled_data_50.csv, 200 comments)

Output:
  data/preprocessing/player_corpus.csv
    columns: language, club, player, comment

Usage:
  python notebooks/01_build_corpus.py
  uv run python notebooks/01_build_corpus.py
"""

import json
import re
import pathlib
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = pathlib.Path(__file__).parent.parent
INSTAGRAM_DIR = BASE / "data" / "instagram"
PILOT_CSV = BASE / "data" / "labeled" / "v1" / "labeled_data_50.csv"
OUTPUT_CSV = BASE / "data" / "preprocessing" / "player_corpus.csv"

# ── Club configuration ─────────────────────────────────────────────────────────
# Maps each club to its JSON file(s) and language.
# Multiple files = supplementary crawls of the same club (merged, deduplicated).
# Language = channel language, not comment language (see paper Section 3.1).
#
# 17 clubs as listed in paper:
#   Germany  (Bundesliga, 5): VfB Stuttgart, Bayer Leverkusen, FC Bayern,
#                              1. FC Heidenheim, Borussia Dortmund
#   England  (Premier League, 6): Chelsea, Liverpool, Arsenal,
#                                  Aston Villa, Man City, Ipswich
#   Spain    (La Liga, 3): Sevilla, Real Sociedad, Villarreal
#   France   (Ligue 1, 3): Lille, Stade Rennais, RC Lens

CLUB_CONFIG = {
    # club_key: (language, [json_stems])
    "vfb":          ("DE", ["vfb_instacomments",  "vfb2_instacomments"]),
    "bayer04":      ("DE", ["bayer04_instacomments"]),
    "fcbayern":     ("DE", ["fcbayern_instacomments"]),
    "fch":          ("DE", ["fch_instacomments"]),
    "bvb":          ("DE", ["bvb_instacomments"]),
    "fcchelsea":    ("EN", ["fcchelsea_instacomments"]),
    "liverpoolfc":  ("EN", ["liverpoolfc_instacomments"]),
    "arsenal":      ("EN", ["arsenal_instacomments"]),
    "avfc":         ("EN", ["avfc_instacomments"]),
    "mancity":      ("EN", ["mancity_instacomments"]),
    "ipswich":      ("EN", ["ipswich_instacomments"]),
    "sevillafc":    ("ES", ["sevillafc_instacomments"]),
    "realsociedad": ("ES", ["realsociedad_instacomments"]),
    "villarealcf":  ("ES", ["villarealcf_instacomments"]),
    "lille":        ("FR", ["lille_instacomments"]),
    "staderennesfc":("FR", ["staderennesfc_instacomments"]),
    "rclens":       ("FR", ["rclens1_instacomments"]),
}

# ── Player list ────────────────────────────────────────────────────────────────
# 10 players per club (consistent with paper: "randomly selecting ten players
# per club from their squad lists, sourced from Transfermarkt.de").
# VfB selection: top 10 by comment volume in original crawl (Transfermarkt 2024/25).
# Ipswich selection: Transfermarkt 2024/25 Premier League squad, top-10 by
# squad prominence (Ipswich was missing from original player.json — added here).

PLAYERS = {
    "vfb": [
        "Deniz Undav", "Anrie Chase", "Fabian Rieder", "Enzo Millot",
        "Jamie Leweling", "Pascal Stenzel", "Alexander Nübel",
        "Ermedin Demirovic", "Angelo Stiller", "Jeff Chabot",
    ],
    "bayer04": [
        "Matej Kovar", "Lukas Hradecky", "Edmond Tabsoba", "Piero Hincapie",
        "Jonathan Tah", "Florian Wirtz", "Granit Xhaka",
        "Robert Andrich", "Victor Boniface", "Alejandro Grimaldo",
    ],
    "fcbayern": [
        "Manuel Neuer", "Min-Jae Kim", "Dayot Upamecano", "Alphonso Davies",
        "Joshua Kimmich", "Aleksandar Pavlovic", "Jamal Musiala",
        "Serge Gnabry", "Leroy Sane", "Michael Olise",
    ],
    "fch": [
        "Kevin Müller", "Patrick Mainka", "Benedikt Gimber",
        "Jonas Föhrenbach", "Lennard Maloney", "Niklas Dorsch",
        "Jan Schöppner", "Paul Wanner", "Leo Scienza", "Marvin Pieringer",
    ],
    "bvb": [
        "Gregor Kobel", "Nico Schlotterbeck", "Waldemar Anton", "Emre Can",
        "Niklas Süle", "Julian Brandt", "Karim Adeyemi",
        "Donyell Malen", "Serhou Guirassy", "Maximilian Beier",
    ],
    "fcchelsea": [
        "Robert Sanchez", "Marc Cucurella", "Reece James", "Enzo Fernandez",
        "Cole Palmer", "Christopher Nkunku", "Jadon Sancho",
        "Joao Felix", "Wesley Fofana", "Ben Chilwell",
    ],
    "liverpoolfc": [
        "Alisson", "Ibrahima Konate", "Joe Gomez", "Trent Alexander-Arnold",
        "Alexis Mac Allister", "Ryan Gravenberch", "Curtis Jones",
        "Dominik Szoboszlai", "Luis Diaz", "Mohamed Salah",
    ],
    "arsenal": [
        "David Raya", "William Saliba", "Ben White", "Jurrien Timber",
        "Oleksandr Zinchenko", "Thomas Partey", "Martin Odegaard",
        "Declan Rice", "Kai Havertz", "Gabriel Martinelli",
    ],
    "avfc": [
        "Emiliano Martinez", "Pau Torres", "Diego Carlos", "Lucas Digne",
        "Ian Maatsen", "Leon Bailey", "Ollie Watkins",
        "Morgan Rogers", "Jacob Ramsey", "Youri Tielemans",
    ],
    "mancity": [
        "Ederson", "Josko Gvardiol", "Kyle Walker", "Rodri",
        "Mateo Kovacic", "Ilkay Gündogan", "Bernardo Silva",
        "Kevin De Bruyne", "Phil Foden", "Erling Haaland",
    ],
    "ipswich": [
        "Arijanet Muric", "Leif Davis", "Luke Woolfenden", "Axel Tuanzebe",
        "Ben Johnson", "Kalvin Phillips", "Omari Hutchinson",
        "Sammie Szmodics", "Liam Delap", "Wes Burns",
    ],
    "sevillafc": [
        "Alvaro Fernandez", "Loic Bade", "Valentin Barco", "Juanlu Sanchez",
        "Jesus Navas", "Djibril Sow", "Dodi Lukebakio",
        "Isaac Romero", "Kelechi Iheanacho", "Albert Sambi Lokonga",
    ],
    "realsociedad": [
        "Alex Remiro", "Nayef Aguerd", "Igor Zubeldia", "Jon Pacheco",
        "Sheraldo Becker", "Mikel Oyarzabal", "Takefusa Kubo",
        "Martin Zubimendi", "Brais Mendez", "Luka Sucic",
    ],
    "villarealcf": [
        "Diego Conde", "Logan Costa", "Raul Albiol", "Sergi Cardona",
        "Juan Foyth", "Dani Parejo", "Pape Gueye",
        "Ilias Akhomach", "Nicolas Pepe", "Ayoze Perez",
    ],
    "lille": [
        "Lucas Chevalier", "Bafode Diakite", "Alexsandro", "Mitchel Bakker",
        "Benjamin Andre", "Ayyoub Bouaddi", "Jonathan David",
        "Matias Fernandez-Pardo", "Edon Zhegrova", "Angel Gomes",
    ],
    "staderennesfc": [
        "Steve Mandanda", "Christopher Wooh", "Alidu Seidu",
        "Lorenz Assignon", "Adrien Truffert", "Glen Kamara",
        "Baptiste Santamaria", "Ludovic Blas", "Amine Gouiri", "Arnaud Kalimuendo",
    ],
    "rclens": [
        "Brice Samba", "Jonathan Gradit", "Kevin Danso", "Facundo Medina",
        "Neil El Aynaoui", "Deiver Machado", "Przemyslaw Frankowski",
        "David Pereira Da Costa", "Andy Diouf", "Remy Labeau Lascary",
    ],
}

# ── Artefact patterns ──────────────────────────────────────────────────────────
ARTEFACT_PATTERNS = [
    re.compile(r"original.?audio", re.I),
    re.compile(r"^[•·]+$"),
    re.compile(r"^original audio$", re.I),
]

MIN_COMMENT_LEN = 10  # characters after stripping whitespace

# ── Helpers ────────────────────────────────────────────────────────────────────

def load_json_file(path: pathlib.Path) -> list[dict]:
    """Load an Instagram JSON file; returns list of post dicts."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


def is_artefact(comment: str) -> bool:
    """True if comment is an Instagram-specific artefact to be filtered."""
    return any(p.search(comment) for p in ARTEFACT_PATTERNS)


def build_name_variants(player_name: str) -> list[str]:
    """
    Returns matching variants for a player name:
      - full name  (e.g. "Alexander Nübel")
      - last name  (e.g. "Nübel")
    Both are used as case-insensitive substrings.
    """
    parts = player_name.strip().split()
    variants = [player_name.lower()]
    if len(parts) > 1:
        variants.append(parts[-1].lower())
    return variants


def comment_mentions_player(comment: str, variants: list[str]) -> bool:
    comment_lower = comment.lower()
    return any(v in comment_lower for v in variants)


# ── Main ───────────────────────────────────────────────────────────────────────

def build_corpus() -> pd.DataFrame:
    rows = []
    seen = set()  # (comment_lower, club, player) for deduplication

    for club, (language, json_stems) in CLUB_CONFIG.items():
        players = PLAYERS[club]

        # Load all JSON files for this club
        club_comments: list[str] = []
        for stem in json_stems:
            json_path = INSTAGRAM_DIR / f"{stem}.json"
            if not json_path.exists():
                print(f"  [WARN] Not found: {json_path.name} — skipping")
                continue
            posts = load_json_file(json_path)
            for post in posts:
                for comment in post.get("comments", []):
                    if isinstance(comment, str):
                        club_comments.append(comment)

        print(f"{club:15s} ({language})  raw comments: {len(club_comments):6,d}")

        for player in players:
            variants = build_name_variants(player)

            for comment in club_comments:
                # Artefact filter
                if is_artefact(comment):
                    continue
                # Minimum length
                if len(comment.strip()) < MIN_COMMENT_LEN:
                    continue
                # Player name match
                if not comment_mentions_player(comment, variants):
                    continue
                # Deduplication
                key = (comment.lower().strip(), club, player)
                if key in seen:
                    continue
                seen.add(key)

                rows.append({
                    "language": language,
                    "club":     club,
                    "player":   player,
                    "comment":  comment.strip(),
                })

    return pd.DataFrame(rows)


def exclude_pilot(df: pd.DataFrame) -> pd.DataFrame:
    """Remove the 200 pilot comments (labeled_data_50.csv) from the corpus."""
    if not PILOT_CSV.exists():
        print("[WARN] labeled_data_50.csv not found — no pilot exclusion applied")
        return df
    pilot = pd.read_csv(PILOT_CSV)
    pilot_comments = set(pilot["comment"].str.strip().str.lower())
    before = len(df)
    df = df[~df["comment"].str.strip().str.lower().isin(pilot_comments)].copy()
    print(f"\nPilot exclusion: removed {before - len(df)} comments ({before} → {len(df)})")
    return df


def print_stats(df: pd.DataFrame) -> None:
    print("\n── Corpus statistics ─────────────────────────────────────────")
    print(f"Total player-filtered comments: {len(df):,}")
    print("\nPer language:")
    lang_stats = df.groupby("language").agg(
        comments=("comment", "count"),
        clubs=("club", "nunique"),
        players=("player", "nunique"),
    )
    print(lang_stats.to_string())
    print("\nPer club (top 5 by comment count):")
    print(df.groupby(["language", "club"])["comment"].count()
            .sort_values(ascending=False).head(20).to_string())


def main():
    print("Building player corpus from Instagram JSONs...\n")

    df = build_corpus()
    df = exclude_pilot(df)
    print_stats(df)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✓ Saved to {OUTPUT_CSV}  ({len(df):,} rows)")


if __name__ == "__main__":
    main()
