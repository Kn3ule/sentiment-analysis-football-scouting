"""
Labeling Tool — Streamlit Frontend
===================================
Manuelles Labeling von Fan-Kommentaren für das Evaluierungs-Dataset.

Start:
    uv run streamlit run src/labeling_tool.py

Workflow:
    - Zeigt Originalkommentar + deutsche Übersetzung (falls nicht DE)
    - Zeigt Claude-Label als Vorschlag zum Abgleich
    - Drei Buttons: POSITIVE / NEGATIVE / NEUTRAL
    - Auto-Save nach jeder Entscheidung
"""

import streamlit as st
import pandas as pd
import pathlib

st.set_page_config(
    page_title="Comment Labeler",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE      = pathlib.Path(__file__).parent.parent
DATA_PATH = BASE / "data" / "labeled" / "to_label_v2_human_claude.csv"

# ── Load / Save ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=0)
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    if "label" not in df.columns:
        df["label"] = ""
    df["label"] = df["label"].fillna("").astype(str)
    if "claude_label" not in df.columns:
        df["claude_label"] = ""
    df["claude_label"] = df["claude_label"].fillna("").astype(str)
    if "comment_de" not in df.columns:
        df["comment_de"] = df["comment"]
    return df

def save_data(df: pd.DataFrame) -> None:
    df.to_csv(DATA_PATH, index=False)

# ── Session init ───────────────────────────────────────────────────────────────
if "df" not in st.session_state:
    st.session_state.df = load_data()

df = st.session_state.df

# ── Filter: unlabeled only ─────────────────────────────────────────────────────
VALID_LABELS  = ["POSITIVE", "NEGATIVE", "NEUTRAL"]
unlabeled_idx = df[~df["label"].isin(VALID_LABELS)].index.tolist()
labeled_mask  = df["label"].isin(VALID_LABELS)
n_total       = len(df)
n_labeled     = int(labeled_mask.sum())

# ── Cursor ─────────────────────────────────────────────────────────────────────
if "cursor" not in st.session_state:
    st.session_state.cursor = 0

cursor = max(0, min(st.session_state.cursor, len(unlabeled_idx) - 1))

# ── Done screen ────────────────────────────────────────────────────────────────
if len(unlabeled_idx) == 0:
    st.success("✅ Alle Kommentare gelabelt!")
    st.metric("Gesamt", n_total)
    dist = df[labeled_mask]["label"].value_counts()
    for lbl, n in dist.items():
        color = {"POSITIVE": "🟢", "NEGATIVE": "🔴", "NEUTRAL": "⚪"}.get(lbl, "")
        st.write(f"{color} {lbl}: **{n}**")
    st.stop()

current_idx = unlabeled_idx[cursor]
row = df.loc[current_idx]

# ── Progress bar ───────────────────────────────────────────────────────────────
st.progress(n_labeled / n_total)
st.caption(f"{n_labeled} / {n_total} gelabelt")

# ── Metadata ───────────────────────────────────────────────────────────────────
lang_flag = {"DE": "🇩🇪", "EN": "🇬🇧", "ES": "🇪🇸", "FR": "🇫🇷"}.get(row["language"], "🌐")
st.markdown(
    f"{lang_flag} **{row['language']}** &nbsp;·&nbsp; "
    f"🏟️ `{row['club']}` &nbsp;·&nbsp; "
    f"👤 **{row['player']}**",
    unsafe_allow_html=True,
)

st.markdown("---")

# ── Comment ────────────────────────────────────────────────────────────────────
original   = str(row["comment"])
comment_de = str(row.get("comment_de", "")).strip()
is_german  = row["language"] == "DE"
has_translation = (
    not is_german
    and bool(comment_de)
    and comment_de.lower() != original.strip().lower()
)

# Always show original comment
st.markdown(
    f"<div style='font-size:1.25em; padding:18px; background:#f8f9fa; "
    f"border-radius:10px; border-left:4px solid #2563eb; "
    f"line-height:1.6; color:#111'>{original}</div>",
    unsafe_allow_html=True,
)

# Show German translation below if available and different
if has_translation:
    st.markdown(
        f"<div style='font-size:1em; padding:12px 18px; background:#eef2ff; "
        f"border-radius:8px; border-left:4px solid #818cf8; "
        f"line-height:1.6; color:#374151; margin-top:8px'>"
        f"🇩🇪 <em>{comment_de}</em></div>",
        unsafe_allow_html=True,
    )

st.markdown("---")

# ── Label buttons ──────────────────────────────────────────────────────────────
def set_label(lbl: str) -> None:
    df.at[current_idx, "label"] = lbl
    st.session_state.df = df
    save_data(df)
    if cursor < len(unlabeled_idx) - 1:
        st.session_state.cursor = cursor + 1

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🟢 POSITIVE", use_container_width=True, type="secondary"):
        set_label("POSITIVE")
        st.rerun()

with col2:
    if st.button("⚪ NEUTRAL", use_container_width=True, type="secondary"):
        set_label("NEUTRAL")
        st.rerun()

with col3:
    if st.button("🔴 NEGATIVE", use_container_width=True, type="secondary"):
        set_label("NEGATIVE")
        st.rerun()

st.markdown("")

# ── Navigation ─────────────────────────────────────────────────────────────────
nav1, nav2, nav3 = st.columns([1, 3, 1])
with nav1:
    if st.button("◀", use_container_width=True):
        if cursor > 0:
            st.session_state.cursor = cursor - 1
            st.rerun()
with nav2:
    jump = st.number_input(
        "Position", min_value=1, max_value=len(unlabeled_idx),
        value=cursor + 1, step=1, label_visibility="collapsed",
    )
    if jump - 1 != cursor:
        st.session_state.cursor = jump - 1
        st.rerun()
with nav3:
    if st.button("▶", use_container_width=True):
        if cursor < len(unlabeled_idx) - 1:
            st.session_state.cursor = cursor + 1
            st.rerun()

# ── Sidebar: stats + reload ────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("📊 Fortschritt")
    dist = df[labeled_mask]["label"].value_counts()
    for lbl in ["POSITIVE", "NEUTRAL", "NEGATIVE"]:
        n = dist.get(lbl, 0)
        color = {"POSITIVE": "🟢", "NEGATIVE": "🔴", "NEUTRAL": "⚪"}.get(lbl, "")
        st.write(f"{color} {lbl}: **{n}**")

    st.divider()
    st.subheader("Sprache (offen)")
    open_df = df[~df["label"].isin(VALID_LABELS)]
    st.dataframe(
        open_df["language"].value_counts().rename("offen"),
        use_container_width=True,
    )

    st.divider()
    st.subheader("Verteilung (annotiert)")
    labeled_df = df[labeled_mask]
    if len(labeled_df) > 0:
        dist_table = (
            labeled_df.groupby(["language", "label"])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=["POSITIVE", "NEUTRAL", "NEGATIVE"], fill_value=0)
            .reindex(["DE", "EN", "ES", "FR"])
            .dropna(how="all")
        )
        dist_table.index.name = None
        st.dataframe(dist_table, use_container_width=True)
    else:
        st.caption("Noch keine Labels vergeben.")

    st.divider()
    if st.button("🔄 Neu laden"):
        st.cache_data.clear()
        del st.session_state["df"]
        st.rerun()
