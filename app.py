"""
UPOS Tagger Dashboard
Uses Stanza NLP to assign Universal POS tags to each word
in source and target sentences (Hindi / English).
"""

import sys
import os

# Allow PyTorch installed to a short path (workaround for Windows MAX_PATH)
SHORT_PATH = r"C:\pt"
if os.path.isdir(SHORT_PATH) and SHORT_PATH not in sys.path:
    sys.path.insert(0, SHORT_PATH)

import streamlit as st
import stanza
from collections import Counter
import pandas as pd
import transformers
transformers.logging.disable_progress_bar()
from sentence_transformers import SentenceTransformer, util

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UPOS Tagger Dashboard",
    page_icon="🏷️",
    layout="wide",
)

# ── UPOS colour palette ──────────────────────────────────────────────────────
UPOS_COLORS = {
    "NOUN":    "#4e79a7",
    "VERB":    "#f28e2b",
    "ADJ":     "#e15759",
    "ADV":     "#76b7b2",
    "PRON":    "#59a14f",
    "DET":     "#edc948",
    "ADP":     "#b07aa1",
    "CONJ":    "#ff9da7",
    "CCONJ":   "#ff9da7",
    "SCONJ":   "#ffbe7d",
    "NUM":     "#9c755f",
    "PART":    "#bab0ac",
    "INTJ":    "#d37295",
    "PROPN":   "#a0cbe8",
    "AUX":     "#f1ce63",
    "PUNCT":   "#8cd17d",
    "SYM":     "#86bcb6",
    "X":       "#e8e8e8",
}

UPOS_DESCRIPTIONS = {
    "NOUN":  "Noun", "VERB":  "Verb", "ADJ":   "Adjective",
    "ADV":   "Adverb", "PRON":  "Pronoun", "DET":   "Determiner",
    "ADP":   "Adposition", "CONJ":  "Conjunction", "CCONJ": "Coord. Conjunction",
    "SCONJ": "Subord. Conjunction", "NUM":   "Numeral", "PART":  "Particle",
    "INTJ":  "Interjection", "PROPN": "Proper Noun", "AUX":   "Auxiliary",
    "PUNCT": "Punctuation", "SYM":   "Symbol", "X":     "Other",
}

LANG_CODES = {"English": "en", "Hindi": "hi"}


# ── Model loading (cached) ───────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_pipeline(lang_code: str):
    """Load local Stanza pipeline safely."""
    return stanza.Pipeline(
        lang_code,
        processors="tokenize,pos",
        use_gpu=False,
        verbose=False,
        download_method=stanza.DownloadMethod.REUSE_RESOURCES
    )

@st.cache_resource(show_spinner=False)
def load_sbert_model():
    """Load XLM-RoBERTa based SBERT model for semantic similarity."""
    return SentenceTransformer('paraphrase-xlm-r-multilingual-v1')


# ── NLP processing ───────────────────────────────────────────────────────────
def tag_sentence(text: str, nlp) -> list[dict]:
    """Return list of {word, upos} dicts for the given text."""
    doc = nlp(text)
    tokens = []
    for sentence in doc.sentences:
        for word in sentence.words:
            tokens.append({"word": word.text, "upos": word.upos or "X"})
    return tokens


def upos_counts(tokens: list[dict]) -> pd.DataFrame:
    """Summarise UPOS tag counts as a sorted DataFrame."""
    counts = Counter(t["upos"] for t in tokens)
    df = pd.DataFrame(
        [(tag, cnt, UPOS_DESCRIPTIONS.get(tag, tag))
         for tag, cnt in counts.most_common()],
        columns=["UPOS Tag", "Count", "Description"],
    )
    return df


# ── Rendering helpers ────────────────────────────────────────────────────────
def render_tagged_words(tokens: list[dict]):
    """Render colour-coded word + UPOS badge HTML."""
    parts = []
    for t in tokens:
        word = t["word"]
        tag = t["upos"]
        color = UPOS_COLORS.get(tag, "#cccccc")
        parts.append(
            f'<span style="display:inline-block;margin:4px 6px;text-align:center;">'
            f'<span style="font-size:1.05rem;font-weight:600;">{word}</span><br>'
            f'<span style="background:{color};color:#fff;border-radius:4px;'
            f'padding:1px 7px;font-size:0.72rem;font-weight:700;letter-spacing:.5px;">'
            f'{tag}</span></span>'
        )
    html = (
        '<div style="background:#f8f9fb;border:1px solid #e0e0e0;border-radius:10px;'
        'padding:14px 10px;line-height:2.6;flex-wrap:wrap;display:flex;">'
        + "".join(parts)
        + "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def render_parallel_comparison(df_src: pd.DataFrame | None, df_tgt: pd.DataFrame | None, src_lang: str, tgt_lang: str) -> float:
    """Render a side-by-side parallel comparison table aligning same UPOS tags on the same row, and compute Match score."""
    src_map = dict(zip(df_src["UPOS Tag"], df_src["Count"])) if df_src is not None else {}
    tgt_map = dict(zip(df_tgt["UPOS Tag"], df_tgt["Count"])) if df_tgt is not None else {}

    # Union of all unique tags present in either source or target, ordered by total count
    all_tags = set(src_map.keys()) | set(tgt_map.keys())
    sorted_tags = sorted(
        all_tags,
        key=lambda t: (src_map.get(t, 0) + tgt_map.get(t, 0), src_map.get(t, 0)),
        reverse=True,
    )

    rows = ""
    match_sum = 0.0
    valid_tags_count = len(sorted_tags)

    for tag in sorted_tags:
        color = UPOS_COLORS.get(tag, "#cccccc")
        chip = (
            f'<span style="background:{color};color:#fff;border-radius:4px;'
            f'padding:3px 12px;font-size:0.82rem;font-weight:700;">{tag}</span>'
        )
        desc = UPOS_DESCRIPTIONS.get(tag, tag)
        src_c = src_map.get(tag, 0)
        tgt_c = tgt_map.get(tag, 0)

        # Calculate match using user formula: min(E,H)/max(E,H)
        if src_c == 0 and tgt_c == 0:
            match_score = 1.0 # Should not happen based on all_tags logic
        elif src_c == 0 or tgt_c == 0:
            match_score = 0.0
        else:
            match_score = min(src_c, tgt_c) / max(src_c, tgt_c)
            
        match_sum += match_score

        src_val = f"<b style='font-size:1.05rem;'>{src_c}</b>" if src_c > 0 else "<span style='color:#bbb;'>0</span>"
        tgt_val = f"<b style='font-size:1.05rem;'>{tgt_c}</b>" if tgt_c > 0 else "<span style='color:#bbb;'>0</span>"
        match_val = f"<b style='font-size:1.05rem;color:{'#28a745' if match_score==1.0 else '#ffc107' if match_score>0 else '#dc3545'};'>{match_score:.2f}</b>"

        rows += (
            f"<tr style='border-bottom:1px solid #f0f0f0;'>"
            f"<td style='padding:8px 12px;'>{chip}</td>"
            f"<td style='padding:8px 12px;color:#555;'>{desc}</td>"
            f"<td style='text-align:center;padding:8px 12px;background:#f9fbfd;'>{src_val}</td>"
            f"<td style='text-align:center;padding:8px 12px;background:#fbfbfb;'>{tgt_val}</td>"
            f"<td style='text-align:center;padding:8px 12px;background:#fefefe;'>{match_val}</td>"
            f"</tr>"
        )

    # Calculate average syntax score
    syntax_score = (match_sum / valid_tags_count) if valid_tags_count > 0 else 0.0

    # Total row
    tot_src = sum(src_map.values())
    tot_tgt = sum(tgt_map.values())
    total_row = (
        f"<tr style='border-top:2px solid #ccc;font-weight:bold;background:#f0f4f8;'>"
        f"<td style='padding:8px 12px;'>Total Words</td>"
        f"<td></td>"
        f"<td style='text-align:center;padding:8px 12px;'>{tot_src}</td>"
        f"<td style='text-align:center;padding:8px 12px;'>{tot_tgt}</td>"
        f"<td style='text-align:center;padding:8px 12px;color:#1e88e5;'>Syntax Avg: {syntax_score:.2f}</td>"
        f"</tr>"
    )

    html = (
        '<table style="width:100%;border-collapse:collapse;font-size:0.92rem;margin-top:10px;">'
        "<thead><tr style='border-bottom:2px solid #ddd;background:#f5f7fa;'>"
        "<th style='text-align:left;padding:10px 12px;'>UPOS Tag</th>"
        "<th style='text-align:left;padding:10px 12px;'>Description</th>"
        f"<th style='text-align:center;padding:10px 12px;'>Source ({src_lang})</th>"
        f"<th style='text-align:center;padding:10px 12px;'>Target ({tgt_lang})</th>"
        "<th style='text-align:center;padding:10px 12px;'>Match Score</th>"
        "</tr></thead><tbody>"
        + rows
        + total_row
        + "</tbody></table>"
    )
    st.markdown(html, unsafe_allow_html=True)
    return syntax_score


# ── Sentence panel ───────────────────────────────────────────────────────────
def sentence_panel(panel_label: str, key_prefix: str):
    """
    Render one sentence panel (language picker + text area + analyse button).
    Returns (tokens, lang_label, raw_text) or (None, None, None) if not yet analysed.
    """
    st.subheader(panel_label)

    lang = st.selectbox(
        "Language",
        list(LANG_CODES.keys()),
        key=f"{key_prefix}_lang",
    )

    text = st.text_area(
        "Enter sentence",
        placeholder=f"Type a {lang} sentence here…",
        height=100,
        key=f"{key_prefix}_text",
    )

    analyse = st.button("🔍 Analyse", key=f"{key_prefix}_btn", use_container_width=True)

    if analyse:
        if not text.strip():
            st.warning("Please enter a sentence first.")
            return None, lang, None

        lang_code = LANG_CODES[lang]
        with st.spinner(f"Tagging {lang} sentence…"):
            nlp = load_pipeline(lang_code)
            tokens = tag_sentence(text.strip(), nlp)

        st.session_state[f"{key_prefix}_tokens"] = tokens
        st.session_state[f"{key_prefix}_lang_label"] = lang
        st.session_state[f"{key_prefix}_raw_text"] = text.strip()

    tokens = st.session_state.get(f"{key_prefix}_tokens")
    lang_label = st.session_state.get(f"{key_prefix}_lang_label", lang)
    raw_text = st.session_state.get(f"{key_prefix}_raw_text")
    return tokens, lang_label, raw_text


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    # ── Header
    st.markdown(
        "<h1 style='text-align:center;'>🏷️ Composite Similarity Dashboard</h1>"
        "<p style='text-align:center;color:#666;'>Evaluates both <strong>Syntax Similarity</strong> (Stanza UPOS tag distribution) "
        "and <strong>Semantic Similarity</strong> (XLM-RoBERTa-based SBERT).</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Two-column input panels
    col_src, col_tgt = st.columns(2, gap="large")

    with col_src:
        src_tokens, src_lang, src_raw = sentence_panel("📥 Source Sentence", "src")

    with col_tgt:
        tgt_tokens, tgt_lang, tgt_raw = sentence_panel("📤 Target Sentence", "tgt")

    # ── Results
    syntax_score = 0.0
    if src_tokens or tgt_tokens:
        st.divider()
        st.markdown("## 📊 Word-Level UPOS Tagging")
        res_src, res_tgt = st.columns(2, gap="large")

        if src_tokens:
            with res_src:
                st.markdown(f"### Source — *{src_lang}*")
                render_tagged_words(src_tokens)

        if tgt_tokens:
            with res_tgt:
                st.markdown(f"### Target — *{tgt_lang}*")
                render_tagged_words(tgt_tokens)

        # ── Parallel Comparison Section
        st.divider()
        st.markdown("## ⚖️ Parallel UPOS Tag Count Comparison")
        df_src = upos_counts(src_tokens) if src_tokens else None
        df_tgt = upos_counts(tgt_tokens) if tgt_tokens else None
        syntax_score = render_parallel_comparison(df_src, df_tgt, src_lang, tgt_lang)

    # ── Composite Score Section
    if src_tokens and tgt_tokens and src_raw and tgt_raw:
        st.divider()
        st.markdown("## 🎯 Final Composite Score")
        st.write("Configure the weights for Syntax vs Semantic similarity.")
        
        col1, col2 = st.columns(2)
        with col1:
            alpha = st.slider("Alpha (Syntax Weight)", 0.0, 1.0, 0.5, 0.05)
        with col2:
            beta = st.slider("Beta (Semantic Weight)", 0.0, 1.0, 0.5, 0.05)
            
        with st.spinner("Calculating semantic similarity using SBERT..."):
            sbert_model = load_sbert_model()
            emb1 = sbert_model.encode(src_raw)
            emb2 = sbert_model.encode(tgt_raw)
            semantic_score = util.cos_sim(emb1, emb2).item()
            
        final_score = (alpha * syntax_score) + (beta * semantic_score)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Syntax Similarity Score (S_avg)", f"{syntax_score:.4f}")
        c2.metric("Semantic Similarity Score", f"{semantic_score:.4f}")
        c3.metric(f"Final Score (α={alpha}, β={beta})", f"{final_score:.4f}")
        
        st.info(f"**Formula**: Final Score = ({alpha} * {syntax_score:.4f}) + ({beta} * {semantic_score:.4f}) = **{final_score:.4f}**")

    # ── Legend
    with st.expander("📖 UPOS Tag Legend", expanded=False):
        cols = st.columns(3)
        for i, (tag, desc) in enumerate(UPOS_DESCRIPTIONS.items()):
            color = UPOS_COLORS.get(tag, "#cccccc")
            chip = (
                f'<span style="background:{color};color:#fff;border-radius:4px;'
                f'padding:2px 10px;font-size:0.85rem;font-weight:700;">{tag}</span>'
            )
            cols[i % 3].markdown(f"{chip} {desc}", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
