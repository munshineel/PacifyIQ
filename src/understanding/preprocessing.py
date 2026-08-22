"""Text preprocessing for the understanding layer.

The masking behaviour here comes directly from an EDA finding: the highest
lexical overlap between any two intents (`order_tracking` and
`return_refund_request`, Jaccard 0.25) was driven *entirely* by order-reference
tokens. `PAC-2026-12345` and `PAC-2026-12350` carry identical intent signal but
TF-IDF treats them as unrelated features.

Masking is implemented as a switchable transformer so it can be ablated rather
than assumed. See reports/classification_report.md for the measured effect.
"""
from __future__ import annotations

import re

from sklearn.base import BaseEstimator, TransformerMixin

# --------------------------------------------------------------------
# Patterns, ordered so the most specific fires first.
# --------------------------------------------------------------------
RE_ORDER_FULL = re.compile(r"\bpac[-\s]?2026[-\s]?\d{4,6}\b", re.I)
RE_ORDER_BARE = re.compile(r"#\d{4,6}\b|\b\d{5}\b")
RE_ERROR_CODE = re.compile(
    r"\b(?:PAY|ERR|BAT|WIFI|SYS|THRM|DSP|AUD|KEY|STO|MEM|CAM)"
    r"(?:[-_][A-Z0-9]+)*[-_]?(?:0x)?[0-9A-F]{2,}\b",
    re.I,
)
RE_POLICY_REF = re.compile(r"\bPOL-[A-Z]{2,3}-\d{3}\b", re.I)
RE_MONEY = re.compile(r"(?:rs\.?|inr|₹)\s?\d[\d,]*|\b\d{4,6}\s?(?:rs|inr)\b", re.I)
RE_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
RE_URL = re.compile(r"https?://\S+")
RE_NUM = re.compile(r"\b\d+\b")
RE_WS = re.compile(r"\s+")

MASKS = {
    "order": " orderref ",
    "error": " errorcode ",
    "policy": " policyref ",
    "money": " moneyamount ",
    "email": " emailaddr ",
    "url": " urladdr ",
    "num": " numval ",
}


def mask_entities(text: str, mask_numbers: bool = True) -> str:
    """Replace identifiers with type placeholders.

    The *type* of entity is informative ("this message names an order") while
    the *value* is noise ("which order" tells you nothing about intent).
    """
    t = str(text)
    t = RE_URL.sub(MASKS["url"], t)
    t = RE_EMAIL.sub(MASKS["email"], t)
    t = RE_POLICY_REF.sub(MASKS["policy"], t)
    t = RE_ERROR_CODE.sub(MASKS["error"], t)
    t = RE_ORDER_FULL.sub(MASKS["order"], t)
    t = RE_ORDER_BARE.sub(MASKS["order"], t)
    t = RE_MONEY.sub(MASKS["money"], t)
    if mask_numbers:
        t = RE_NUM.sub(MASKS["num"], t)
    return RE_WS.sub(" ", t).strip()


def normalize(text: str) -> str:
    """Lowercase, collapse repeated characters, strip stray punctuation.

    Character repetition ("waaaarrraaanty") appears in the hard test set and
    would otherwise produce unseen tokens.
    """
    t = str(text).lower()
    t = re.sub(r"(.)\1{2,}", r"\1\1", t)      # waaaa -> waa
    t = re.sub(r"[?!]{2,}", lambda m: m.group(0)[0], t)
    t = re.sub(r"[^\w\s#@.\-/₹']", " ", t)
    return RE_WS.sub(" ", t).strip()


class TextPreprocessor(BaseEstimator, TransformerMixin):
    """Sklearn-compatible preprocessing, so it lives inside the Pipeline and
    is persisted with the model rather than applied ad hoc at call time.

    Parameters
    ----------
    mask : bool
        Replace order refs, error codes and numbers with type placeholders.
    lowercase : bool
        Normalise case and character repetition.
    mask_numbers : bool
        Whether bare numbers are masked in addition to structured identifiers.
    """

    def __init__(self, mask: bool = True, lowercase: bool = True,
                 mask_numbers: bool = True):
        self.mask = mask
        self.lowercase = lowercase
        self.mask_numbers = mask_numbers

    def fit(self, X, y=None):  # noqa: N803
        return self

    def transform(self, X):  # noqa: N803
        out = []
        for t in X:
            s = str(t)
            if self.mask:
                s = mask_entities(s, mask_numbers=self.mask_numbers)
            if self.lowercase:
                s = normalize(s)
            out.append(s)
        return out

    def get_feature_names_out(self, input_features=None):
        return input_features


if __name__ == "__main__":
    samples = [
        "where is my order PAC-2026-12345",
        "status of #12347 please",
        "getting error PAY-402 at checkout",
        "monitor shows ERR-DP-0x004",
        "refund of Rs 64,900 for order 12345",
        "waaaarrraaanty periood??????",
        "MY LAPTOP IS BROKEN AND NOBODY IS HELPING ME",
    ]
    pre = TextPreprocessor(mask=True)
    for s, o in zip(samples, pre.transform(samples)):
        print(f"  {s}\n  -> {o}\n")
