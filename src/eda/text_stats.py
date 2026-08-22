"""PHASE 3 — Text analysis.

Length statistics, n-grams, vocabulary characteristics, sentiment lexicon
signals, template detection, and train/test drift measurement.

Everything here produces a number or a table that feeds a Phase 4 decision.

    python -m src.eda.text_stats
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# ---------------------------------------------------------------------
# Domain lexicons. Small and hand-built rather than a general-purpose
# sentiment package, because support language is domain specific: "broken"
# and "damaged" are strong negatives here and neutral elsewhere.
# ---------------------------------------------------------------------
NEGATIVE_TERMS = {
    "angry", "furious", "upset", "disappointed", "disgusted", "terrible",
    "worst", "useless", "unacceptable", "ridiculous", "pathetic", "rude",
    "fraud", "theft", "scam", "never", "nobody", "ignored", "waiting",
    "still", "again", "third", "broken", "damaged", "faulty", "defective",
    "cracked", "dead", "failed", "wrong", "late", "delayed", "stuck",
    "complaint", "court", "lawyer", "chargeback", "ganda", "bekar",
}
POSITIVE_TERMS = {
    "thanks", "thank", "please", "great", "love", "good", "appreciate",
    "helpful", "quick", "happy", "excellent", "perfect", "kindly", "hope",
}
URGENCY_TERMS = {
    "urgent", "asap", "immediately", "now", "today", "emergency", "right",
    "quickly", "jaldi", "turant", "critical", "deadline",
}
# Romanised Hindi tokens present in the data. Used to measure code-mixing.
CODEMIX_TERMS = {
    "kahan", "hai", "kab", "kya", "nahi", "mujhe", "chahiye", "karna",
    "kaise", "bhai", "bhej", "kitne", "din", "paisa", "kar", "raha",
    "gaya", "jaldi", "aayega", "ho", "do", "ka", "ki", "ke", "me", "se",
    "pe", "bohot", "abhi", "tak", "band", "jata", "baar", "yaar", "sakte",
}

TOKEN = re.compile(r"[a-z0-9#\-]+")
SENT_SPLIT = re.compile(r"[.!?]+")


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(str(text).lower())


# =====================================================================
# Length statistics
# =====================================================================

def length_features(texts: pd.Series) -> pd.DataFrame:
    """Character, word and sentence counts plus derived ratios."""
    s = texts.astype(str)
    df = pd.DataFrame({"text": s})
    df["n_chars"] = s.str.len()
    df["n_words"] = s.str.split().str.len()
    df["n_sentences"] = s.apply(
        lambda t: max(1, len([p for p in SENT_SPLIT.split(t) if p.strip()]))
    )
    df["avg_word_len"] = (df["n_chars"] / df["n_words"].clip(lower=1)).round(2)
    df["n_upper"] = s.apply(lambda t: sum(c.isupper() for c in t))
    df["upper_ratio"] = (df["n_upper"] / df["n_chars"].clip(lower=1)).round(3)
    df["n_digits"] = s.str.count(r"\d")
    df["n_question"] = s.str.count(r"\?")
    df["n_exclaim"] = s.str.count(r"!")
    df["has_order_ref"] = s.str.contains(r"\d{5}", regex=True)
    df["has_error_code"] = s.str.contains(
        r"\b(?:PAY|ERR|BAT|WIFI|SYS|THRM|DSP|AUD|KEY|STO|MEM|CAM)[-\s]?[\w\-]*\d",
        regex=True, case=False,
    )
    return df


def length_summary(df: pd.DataFrame, group: str | None = None) -> pd.DataFrame:
    """Distribution summary of length features, optionally per group."""
    cols = ["n_chars", "n_words", "n_sentences"]
    if group:
        out = df.groupby(group, observed=True)[cols].agg(["mean", "median", "min", "max"])
        out.columns = [f"{a}_{b}" for a, b in out.columns]
        return out.round(1)
    return df[cols].describe().round(1)


# =====================================================================
# Lexicon signals
# =====================================================================

def lexicon_features(texts: pd.Series) -> pd.DataFrame:
    """Count domain-lexicon hits per message."""
    toks = texts.astype(str).apply(tokenize)
    return pd.DataFrame(
        {
            "n_negative": toks.apply(lambda t: sum(w in NEGATIVE_TERMS for w in t)),
            "n_positive": toks.apply(lambda t: sum(w in POSITIVE_TERMS for w in t)),
            "n_urgency": toks.apply(lambda t: sum(w in URGENCY_TERMS for w in t)),
            "n_codemix": toks.apply(lambda t: sum(w in CODEMIX_TERMS for w in t)),
            "n_tokens": toks.apply(len),
        }
    ).assign(
        polarity=lambda d: d["n_positive"] - d["n_negative"],
        is_codemixed=lambda d: d["n_codemix"] >= 2,
    )


# =====================================================================
# N-grams
# =====================================================================

def top_ngrams(
    texts: pd.Series, n: int = 1, top: int = 15, min_df: int = 2,
    stop_words: str | None = None,
) -> pd.DataFrame:
    """Most frequent n-grams by raw document frequency."""
    vec = CountVectorizer(
        ngram_range=(n, n), min_df=min_df, stop_words=stop_words, token_pattern=r"[a-z0-9#\-]+"
    )
    try:
        X = vec.fit_transform(texts.astype(str).str.lower())
    except ValueError:
        return pd.DataFrame(columns=["ngram", "count"])
    counts = np.asarray(X.sum(axis=0)).ravel()
    order = counts.argsort()[::-1][:top]
    names = vec.get_feature_names_out()
    return pd.DataFrame({"ngram": names[order], "count": counts[order]})


def distinctive_terms(texts: pd.Series, labels: pd.Series, top: int = 8) -> pd.DataFrame:
    """Terms most characteristic of each class, by mean TF-IDF weight.

    This is what tells you whether classes are lexically separable: if two
    classes share their top terms, a bag-of-words model will confuse them.
    """
    vec = TfidfVectorizer(
        min_df=2, sublinear_tf=True, token_pattern=r"[a-z0-9#\-]+", ngram_range=(1, 2)
    )
    X = vec.fit_transform(texts.astype(str).str.lower())
    names = np.array(vec.get_feature_names_out())

    rows = []
    for label in labels.dropna().unique():
        mask = (labels == label).to_numpy()
        mean_w = np.asarray(X[mask].mean(axis=0)).ravel()
        for i in mean_w.argsort()[::-1][:top]:
            rows.append({"intent": label, "term": names[i], "weight": round(mean_w[i], 4)})
    return pd.DataFrame(rows)


# =====================================================================
# Vocabulary and drift
# =====================================================================

def vocabulary_stats(texts: pd.Series) -> dict:
    """Size, type-token ratio, hapax rate, coverage curve."""
    toks = [w for t in texts.astype(str) for w in tokenize(t)]
    counts = Counter(toks)
    total = len(toks)
    vocab = len(counts)
    hapax = sum(1 for c in counts.values() if c == 1)

    ordered = np.array(sorted(counts.values())[::-1])
    cum = ordered.cumsum() / total
    cover_90 = int(np.searchsorted(cum, 0.90) + 1)

    return {
        "n_tokens": total,
        "vocab_size": vocab,
        "type_token_ratio": round(vocab / max(total, 1), 4),
        "hapax_count": hapax,
        "hapax_pct": round(100 * hapax / max(vocab, 1), 1),
        "types_for_90pct_coverage": cover_90,
        "top_terms": counts.most_common(10),
    }


def vocabulary_drift(train: pd.Series, test: pd.Series) -> dict:
    """How much test vocabulary is unseen in training.

    A high OOV rate means a bag-of-words model trained on `train` will be
    partially blind on `test`, which quantifies the expected performance gap.
    """
    tr_vocab = set(w for t in train.astype(str) for w in tokenize(t))
    te_tokens = [w for t in test.astype(str) for w in tokenize(t)]
    te_vocab = set(te_tokens)

    oov_types = te_vocab - tr_vocab
    oov_token_hits = sum(1 for w in te_tokens if w not in tr_vocab)

    return {
        "train_vocab": len(tr_vocab),
        "test_vocab": len(te_vocab),
        "oov_types": len(oov_types),
        "oov_type_rate_pct": round(100 * len(oov_types) / max(len(te_vocab), 1), 1),
        "oov_token_rate_pct": round(100 * oov_token_hits / max(len(te_tokens), 1), 1),
        "shared_vocab": len(te_vocab & tr_vocab),
        "example_oov": sorted(oov_types)[:20],
    }


# =====================================================================
# Template / boilerplate detection
# =====================================================================

def detect_templates(texts: pd.Series, min_shared: int = 4) -> pd.DataFrame:
    """Find repeated word-sequence skeletons indicating templated text.

    Replaces digits and known slot values with placeholders, then counts how
    many distinct surface forms collapse onto each skeleton. High collapse
    means the text was generated from templates rather than written.
    """
    def skeleton(t: str) -> str:
        t = str(t).lower()
        t = re.sub(r"#?\b\d[\d\-]*\b", "<NUM>", t)
        t = re.sub(r"pac-?2026-?<NUM>", "<ORDER>", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t

    sk = texts.astype(str).apply(skeleton)
    counts = sk.value_counts()
    repeated = counts[counts >= 2]

    return pd.DataFrame(
        {
            "skeleton": repeated.index[:20],
            "n_variants": repeated.values[:20],
        }
    ).assign(
        _summary=lambda d: None
    ).drop(columns=["_summary"])


def template_score(texts: pd.Series) -> dict:
    """Single number: what share of messages share a skeleton with another."""
    def skeleton(t: str) -> str:
        t = str(t).lower()
        t = re.sub(r"#?\b\d[\d\-]*\b", "<NUM>", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t

    sk = texts.astype(str).apply(skeleton)
    counts = sk.value_counts()
    repeated_rows = int(counts[counts >= 2].sum())
    return {
        "n_messages": len(texts),
        "n_unique_skeletons": int(sk.nunique()),
        "messages_sharing_skeleton": repeated_rows,
        "template_rate_pct": round(100 * repeated_rows / max(len(texts), 1), 1),
    }


# =====================================================================
# Class overlap
# =====================================================================

def class_lexical_overlap(texts: pd.Series, labels: pd.Series, top_k: int = 20) -> pd.DataFrame:
    """Pairwise Jaccard overlap of each class's top terms.

    High overlap predicts confusion. Use this to decide which class pairs
    need explicit disambiguation features in Phase 4.
    """
    vec = TfidfVectorizer(min_df=2, sublinear_tf=True, token_pattern=r"[a-z0-9#\-]+")
    X = vec.fit_transform(texts.astype(str).str.lower())
    names = np.array(vec.get_feature_names_out())

    top_terms: dict[str, set[str]] = {}
    for label in labels.dropna().unique():
        mask = (labels == label).to_numpy()
        mean_w = np.asarray(X[mask].mean(axis=0)).ravel()
        top_terms[label] = set(names[mean_w.argsort()[::-1][:top_k]])

    rows = []
    keys = sorted(top_terms)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            inter = top_terms[a] & top_terms[b]
            union = top_terms[a] | top_terms[b]
            rows.append(
                {
                    "intent_a": a,
                    "intent_b": b,
                    "jaccard": round(len(inter) / len(union), 3),
                    "shared_terms": ", ".join(sorted(inter)[:6]),
                }
            )
    return pd.DataFrame(rows).sort_values("jaccard", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    from src.eda import loaders

    pd.set_option("display.width", 160)
    tr = loaders.load_intent_train()
    te = loaders.load_intent_test()

    print("LENGTH (train)")
    print(length_summary(length_features(tr["text"])))

    print("\nVOCABULARY (train)")
    for k, v in vocabulary_stats(tr["text"]).items():
        if k != "top_terms":
            print(f"  {k:28s} {v}")

    print("\nDRIFT train -> test")
    for k, v in vocabulary_drift(tr["text"], te["text"]).items():
        if k != "example_oov":
            print(f"  {k:28s} {v}")

    print("\nTEMPLATE SCORE")
    print("  train:", template_score(tr["text"]))
    print("  test: ", template_score(te["text"]))

    print("\nTOP CLASS OVERLAPS")
    print(class_lexical_overlap(tr["text"], tr["intent"]).head(8).to_string(index=False))
