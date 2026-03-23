"""
IAA Labeling Tool — Streamlit Frontend
=======================================
Annotation tool for the second annotator (Annotator B) — Inter-Annotator Agreement.

Start:
    uv run streamlit run src/iaa_labeling_tool.py

Input:  data/labeled/iaa_subset_annotator_b.csv       (100 comments, no labels)
Output: data/labeled/iaa_subset_annotator_b_labeled.csv

Instructions for Annotator B:
    - Label each comment as POSITIVE, NEGATIVE, or NEUTRAL
    - Base your judgment on the sentiment toward the mentioned player
    - Do NOT discuss labels with Annotator A before finishing all 100 comments
"""

import streamlit as st
import pandas as pd
import pathlib

st.set_page_config(
    page_title="IAA Labeling — Annotator B",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE      = pathlib.Path(__file__).parent.parent
CSV_PATH  = BASE / "data" / "labeled" / "iaa_subset_annotator_b.csv"
SAVE_PATH = BASE / "data" / "labeled" / "iaa_subset_annotator_b_labeled.csv"

# ── Load / Save ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=0)
def load_data() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    if "label" not in df.columns:
        df["label"] = ""
    df["label"] = df["label"].fillna("").astype(str)
    return df

def save_data(df: pd.DataFrame) -> None:
    df.to_csv(SAVE_PATH, index=False)

# ── Session init ───────────────────────────────────────────────────────────────
if "df" not in st.session_state:
    st.session_state.df = load_data()
    if SAVE_PATH.exists():
        saved = pd.read_csv(SAVE_PATH)
        if "label" in saved.columns:
            st.session_state.df["label"] = saved["label"].fillna("").astype(str)

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
    st.success("✅ All comments labeled! Thank you.")
    st.metric("Total", n_total)
    dist = df[labeled_mask]["label"].value_counts()
    for lbl, n in dist.items():
        icon = {"POSITIVE": "🟢", "NEGATIVE": "🔴", "NEUTRAL": "⚪"}.get(lbl, "")
        st.write(f"{icon} {lbl}: **{n}**")
    st.info(f"Your labels have been saved to:\n`{SAVE_PATH.name}`\nPlease send this file to Annotator A.")
    st.stop()

current_idx = unlabeled_idx[cursor]
row = df.loc[current_idx]

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("IAA Annotation")
st.caption("Inter-Annotator Agreement — Annotator B")

# ── Progress ───────────────────────────────────────────────────────────────────
st.progress(n_labeled / n_total)
st.caption(f"{n_labeled} / {n_total} labeled")

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

st.markdown(
    f"<div style='font-size:1.25em; padding:18px; background:#f8f9fa; "
    f"border-radius:10px; border-left:4px solid #2563eb; "
    f"line-height:1.6; color:#111'>{original}</div>",
    unsafe_allow_html=True,
)

if has_translation:
    st.markdown(
        f"<div style='font-size:1em; padding:12px 18px; background:#eef2ff; "
        f"border-radius:8px; border-left:4px solid #818cf8; "
        f"line-height:1.6; color:#374151; margin-top:8px'>"
        f"🇩🇪 <em>{comment_de}</em></div>",
        unsafe_allow_html=True,
    )

st.markdown("---")

# ── Task reminder ──────────────────────────────────────────────────────────────
st.caption("What sentiment does this comment express **toward the mentioned player**?")

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

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("📋 Instructions")
    st.markdown("""
Label each comment based on the sentiment expressed **toward the named player**:

🟢 **POSITIVE** — praise, admiration, support

⚪ **NEUTRAL** — factual, ambiguous, team-focused

🔴 **NEGATIVE** — criticism, disappointment, hostility

---
**When unsure:** default to **NEUTRAL**.

**Do not** discuss labels with Annotator A before completing all 100 comments.
""")

    st.divider()
    st.subheader("📊 Progress")
    dist = df[labeled_mask]["label"].value_counts()
    for lbl in ["POSITIVE", "NEUTRAL", "NEGATIVE"]:
        n = dist.get(lbl, 0)
        icon = {"POSITIVE": "🟢", "NEGATIVE": "🔴", "NEUTRAL": "⚪"}.get(lbl, "")
        st.write(f"{icon} {lbl}: **{n}**")

    st.divider()
    st.subheader("🌐 Remaining by language")
    open_df = df[~df["label"].isin(VALID_LABELS)]
    st.dataframe(
        open_df["language"].value_counts().rename("remaining"),
        use_container_width=True,
    )

    st.divider()
    if st.button("🔄 Reload"):
        st.cache_data.clear()
        del st.session_state["df"]
        st.rerun()
