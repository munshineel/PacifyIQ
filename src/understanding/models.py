"""Candidate models for intent classification.

Hyperparameters are taken from the EDA findings rather than convention:

- `max_features=1500` — the vocabulary coverage curve showed 444 types reach
  95% of tokens; 1500 with bigrams leaves headroom without fitting noise.
- `min_df=2` — drops the 471 hapax terms (45.7% of the vocabulary).
- Character n-grams — the test set has a 22.6% OOV *token* rate and 12% of
  training messages carry injected typos. Word features alone go blind.
- `class_weight="balanced"` — the training set is imbalanced 10:1 by design,
  mirroring the real queue. We reweight the loss rather than resample, so the
  model still sees realistic priors.

See reports/eda_findings.md decisions 4, 6, 7, 8.
"""
from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from src.understanding.preprocessing import TextPreprocessor

RANDOM_STATE = 42


# =====================================================================
# Feature extractors
# =====================================================================

def word_tfidf(max_features: int = 1500) -> TfidfVectorizer:
    """Word unigrams and bigrams."""
    return TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_features=max_features,
        sublinear_tf=True,
        token_pattern=r"[a-z0-9#\-]+",
    )


def char_tfidf(max_features: int = 3000) -> TfidfVectorizer:
    """Character n-grams within word boundaries.

    Robust to typos and to unseen words, which is what the 22.6% OOV token
    rate on the hard test set demands.
    """
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=max_features,
        sublinear_tf=True,
    )


def word_char_union(char_weight: float = 0.6) -> FeatureUnion:
    """Word + character features.

    `char_weight` is a real hyperparameter, not a default to accept. EDA
    measured a 22.6% OOV *token* rate on the hard test set, which argues for
    weighting character features more heavily than the usual 0.5-0.6. The
    comparison grid below sweeps it.
    """
    return FeatureUnion(
        [("word", word_tfidf()), ("char", char_tfidf())],
        transformer_weights={"word": 1.0, "char": char_weight},
    )


# =====================================================================
# Model definitions
# =====================================================================

def build(name: str, features: str = "word", mask: bool = True) -> Pipeline:
    """Assemble a full pipeline: preprocess -> vectorise -> classify.

    Preprocessing lives inside the pipeline so it is persisted with the model
    and cannot drift from what was used at training time.
    """
    if features.startswith("union"):
        # "union", "union1.0", "union1.5"
        w = float(features[5:]) if len(features) > 5 else 0.6
        feats = word_char_union(char_weight=w)
    else:
        feats = {"word": word_tfidf(), "char": char_tfidf()}[features]

    clf = {
        "majority": DummyClassifier(strategy="most_frequent"),
        "stratified": DummyClassifier(strategy="stratified", random_state=RANDOM_STATE),
        "logreg": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            C=5.0,
            random_state=RANDOM_STATE,
        ),
        "linsvc": LinearSVC(
            class_weight="balanced",
            C=1.0,
            max_iter=5000,
            random_state=RANDOM_STATE,
        ),
        "cnb": ComplementNB(alpha=0.3),
        "sgd": SGDClassifier(
            loss="modified_huber",
            class_weight="balanced",
            alpha=1e-4,
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),
    }[name]

    return Pipeline(
        [
            ("pre", TextPreprocessor(mask=mask)),
            ("feat", feats),
            ("clf", clf),
        ]
    )


# The comparison grid. Each entry is (label, model, features, mask).
CANDIDATES: list[tuple[str, str, str, bool]] = [
    ("majority baseline",        "majority", "word",  True),
    ("stratified baseline",      "stratified", "word", True),
    ("TF-IDF word + LogReg",     "logreg",   "word",  True),
    ("TF-IDF word + LinearSVC",  "linsvc",   "word",  True),
    ("TF-IDF word + ComplementNB", "cnb",    "word",  True),
    ("TF-IDF word + SGD",        "sgd",      "word",  True),
    ("TF-IDF char + LogReg",     "logreg",   "char",  True),
    ("TF-IDF union + LogReg",    "logreg",   "union", True),
    ("TF-IDF union + LinearSVC", "linsvc",   "union", True),
    # char-weight sweep, motivated by the 22.6% OOV token rate
    ("TF-IDF union(char=1.0) + LogReg",   "logreg", "union1.0", True),
    ("TF-IDF union(char=1.5) + LogReg",   "logreg", "union1.5", True),
    ("TF-IDF union(char=1.0) + LinearSVC", "linsvc", "union1.0", True),
    ("TF-IDF union(char=1.5) + LinearSVC", "linsvc", "union1.5", True),
]

# Masking ablation: same models, masking switched off.
ABLATION: list[tuple[str, str, str, bool]] = [
    ("TF-IDF word + LogReg (no mask)",  "logreg", "word",  False),
    ("TF-IDF union + LogReg (no mask)", "logreg", "union", False),
    ("TF-IDF union + LinearSVC (no mask)", "linsvc", "union", False),
]
