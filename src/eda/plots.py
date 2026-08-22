"""PHASE 3 — Visualisations.

Each function produces one figure that answers one question. No filler charts.
Every figure is referenced in reports/eda_findings.md.

    from src.eda import plots
    plots.setup()
    plots.fig_intent_distribution(tickets, save=True)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.eda.loaders import figures_dir

PALETTE = "crest"
NEG_COLOR = "#c0392b"
POS_COLOR = "#27ae60"
NEU_COLOR = "#95a5a6"


def setup() -> None:
    """Consistent styling across every figure."""
    sns.set_theme(style="whitegrid", palette=PALETTE)
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "figure.titlesize": 13,
            "figure.titleweight": "bold",
        }
    )


def _save(fig: plt.Figure, name: str, save: bool) -> Path | None:
    if not save:
        return None
    path = figures_dir() / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# =====================================================================
# 1. What problems exist, and how common are they?
# =====================================================================

def fig_intent_distribution(tickets: pd.DataFrame, save: bool = True):
    """Q1+Q2: what kinds of problems exist and which dominate.

    Combines volume with deflection rate, because volume alone does not tell
    you where the operational cost sits.
    """
    agg = (
        tickets.groupby("intent", observed=True)
        .agg(tickets=("ticket_id", "size"), escalated=("escalated", "sum"))
        .sort_values("tickets", ascending=True)
    )
    agg["escalation_pct"] = 100 * agg["escalated"] / agg["tickets"]
    agg["share"] = 100 * agg["tickets"] / agg["tickets"].sum()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    axes[0].barh(agg.index, agg["tickets"], color=sns.color_palette(PALETTE, len(agg)))
    for i, (n, s) in enumerate(zip(agg["tickets"], agg["share"])):
        axes[0].text(n + 40, i, f"{n:,} ({s:.1f}%)", va="center", fontsize=8)
    axes[0].set_title("Ticket volume by intent")
    axes[0].set_xlabel("tickets")
    axes[0].set_xlim(0, agg["tickets"].max() * 1.25)

    colors = [NEG_COLOR if v > 50 else "#e67e22" if v > 25 else POS_COLOR
              for v in agg["escalation_pct"]]
    axes[1].barh(agg.index, agg["escalation_pct"], color=colors)
    for i, v in enumerate(agg["escalation_pct"]):
        axes[1].text(v + 1, i, f"{v:.0f}%", va="center", fontsize=8)
    axes[1].set_title("Escalation rate by intent")
    axes[1].set_xlabel("% requiring a human")
    axes[1].set_xlim(0, 100)
    axes[1].axvline(50, ls="--", c="grey", lw=1)
    axes[1].set_yticklabels([])

    fig.suptitle("Volume is not the same as cost: the smallest intents escalate most")
    fig.tight_layout()
    return _save(fig, "01_intent_distribution", save)


# =====================================================================
# 2. Which issues generate negative sentiment, and which are urgent?
# =====================================================================

def fig_sentiment_urgency(tickets: pd.DataFrame, save: bool = True):
    """Q3+Q4: sentiment and urgency profile per intent."""
    order = (
        tickets.groupby("intent", observed=True)["sentiment"]
        .apply(lambda s: (s == "negative").mean())
        .sort_values(ascending=False)
        .index
    )

    sent = (
        pd.crosstab(tickets["intent"], tickets["sentiment"], normalize="index")
        .reindex(order)[["positive", "neutral", "negative"]] * 100
    )
    prio = (
        pd.crosstab(tickets["intent"], tickets["priority"], normalize="index")
        .reindex(order)[["low", "medium", "high"]] * 100
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    sent.plot(kind="barh", stacked=True, ax=axes[0],
              color=[POS_COLOR, NEU_COLOR, NEG_COLOR], width=0.75)
    axes[0].set_title("Sentiment composition by intent")
    axes[0].set_xlabel("% of tickets")
    axes[0].set_ylabel("")
    axes[0].legend(title="", loc="lower right", fontsize=8)
    for i, v in enumerate(sent["negative"]):
        axes[0].text(101, i, f"{v:.0f}% neg", va="center", fontsize=8, color=NEG_COLOR)
    axes[0].set_xlim(0, 118)

    prio.plot(kind="barh", stacked=True, ax=axes[1],
              color=["#dfe6e9", "#fdcb6e", "#d63031"], width=0.75)
    axes[1].set_title("Priority composition by intent")
    axes[1].set_xlabel("% of tickets")
    axes[1].set_ylabel("")
    axes[1].set_yticklabels([])
    axes[1].legend(title="", loc="lower right", fontsize=8)

    fig.suptitle("Complaints and payment issues carry the negative sentiment load")
    fig.tight_layout()
    return _save(fig, "02_sentiment_urgency", save)


# =====================================================================
# 3. Class imbalance
# =====================================================================

def fig_class_imbalance(train: pd.DataFrame, test: pd.DataFrame, save: bool = True):
    """Q5: how imbalanced are the labels, and does test match train?"""
    tr = train["intent"].value_counts()
    te = test["intent"].value_counts().reindex(tr.index).fillna(0)

    tr_pct = 100 * tr / tr.sum()
    te_pct = 100 * te / te.sum()

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5))

    x = np.arange(len(tr))
    axes[0].bar(x - 0.2, tr.values, 0.4, label=f"train (n={tr.sum():,})", color="#2c7873")
    axes[0].bar(x + 0.2, te.values, 0.4, label=f"test (n={int(te.sum())})", color="#e67e22")
    axes[0].set_yscale("log")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(tr.index, rotation=45, ha="right", fontsize=8)
    axes[0].set_ylabel("examples (log scale)")
    axes[0].set_title(f"Class counts — imbalance {tr.max()}:{tr.min()} = {tr.max()/tr.min():.0f}x")
    axes[0].legend(fontsize=8)
    axes[0].axhline(30, ls="--", c=NEG_COLOR, lw=1)
    axes[0].text(len(tr) - 0.5, 32, "30-example floor", fontsize=7, color=NEG_COLOR, ha="right")

    diff = (te_pct - tr_pct).sort_values()
    colors = [NEG_COLOR if v < 0 else POS_COLOR for v in diff.values]
    axes[1].barh(diff.index, diff.values, color=colors)
    axes[1].axvline(0, c="black", lw=0.8)
    axes[1].set_xlabel("test share − train share (percentage points)")
    axes[1].set_title("Distribution shift between train and test")
    axes[1].tick_params(labelsize=8)

    fig.suptitle("Deliberate imbalance in train; test is more uniform by design")
    fig.tight_layout()
    return _save(fig, "03_class_imbalance", save)


# =====================================================================
# 4. Language of different issue types
# =====================================================================

def fig_text_characteristics(
    train_feats: pd.DataFrame, test_feats: pd.DataFrame, save: bool = True
):
    """Q6: how does message language differ, and how far apart are the splits?"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    bins = np.arange(0, 90, 3)
    axes[0].hist(train_feats["n_words"], bins=bins, alpha=0.75,
                 label=f"train (median {train_feats['n_words'].median():.0f})",
                 color="#2c7873", density=True)
    axes[0].hist(test_feats["n_words"], bins=bins, alpha=0.65,
                 label=f"test (median {test_feats['n_words'].median():.0f})",
                 color="#e67e22", density=True)
    axes[0].set_xlabel("words per message")
    axes[0].set_ylabel("density")
    axes[0].set_title("Message length distribution")
    axes[0].legend(fontsize=8)

    # Vocabulary coverage curve — directly sizes TF-IDF max_features.
    from collections import Counter

    from src.eda.text_stats import tokenize

    counts = Counter(w for t in train_feats["text"] for w in tokenize(t))
    freqs = np.array(sorted(counts.values())[::-1])
    coverage = freqs.cumsum() / freqs.sum()
    ranks = np.arange(1, len(freqs) + 1)

    axes[1].plot(ranks, 100 * coverage, lw=2, color="#2c7873")
    for pct, c in [(0.90, "#e67e22"), (0.95, NEG_COLOR)]:
        n = int(np.searchsorted(coverage, pct) + 1)
        axes[1].axvline(n, ls="--", lw=1, color=c)
        axes[1].text(n + 12, 100 * pct - 12, f"{n} types\n= {pct:.0%}",
                     fontsize=7.5, color=c)
    axes[1].set_xlabel("vocabulary rank")
    axes[1].set_ylabel("% of tokens covered")
    axes[1].set_title(f"Vocabulary coverage (train, {len(freqs)} types)")
    axes[1].set_ylim(0, 103)

    props = pd.DataFrame(
        {
            "train": [
                100 * train_feats["has_order_ref"].mean(),
                100 * train_feats["has_error_code"].mean(),
                100 * train_feats["is_codemixed"].mean(),
                100 * (train_feats["n_negative"] > 0).mean(),
                100 * (train_feats["n_urgency"] > 0).mean(),
            ],
            "test": [
                100 * test_feats["has_order_ref"].mean(),
                100 * test_feats["has_error_code"].mean(),
                100 * test_feats["is_codemixed"].mean(),
                100 * (test_feats["n_negative"] > 0).mean(),
                100 * (test_feats["n_urgency"] > 0).mean(),
            ],
        },
        index=["order ref", "error code", "code-mixed", "negative words", "urgency words"],
    )
    props.plot(kind="barh", ax=axes[2], color=["#2c7873", "#e67e22"], width=0.75)
    axes[2].set_xlabel("% of messages")
    axes[2].set_title("Surface features: train vs test")
    axes[2].legend(fontsize=8)
    axes[2].tick_params(labelsize=8)

    fig.suptitle("Test messages are longer, messier, and lexically different")
    fig.tight_layout()
    return _save(fig, "04_text_characteristics", save)


# =====================================================================
# 5. Class separability
# =====================================================================

def fig_class_overlap(overlap: pd.DataFrame, save: bool = True):
    """Q7: which class pairs share vocabulary and will therefore confuse."""
    labels = sorted(set(overlap["intent_a"]) | set(overlap["intent_b"]))
    arr = np.zeros((len(labels), len(labels)))
    idx = {lab: i for i, lab in enumerate(labels)}
    for _, r in overlap.iterrows():
        i, j = idx[r["intent_a"]], idx[r["intent_b"]]
        arr[i, j] = arr[j, i] = r["jaccard"]
    np.fill_diagonal(arr, np.nan)
    mat = pd.DataFrame(arr, index=labels, columns=labels)

    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(mat, annot=True, fmt=".2f", cmap="Reds", ax=ax,
                cbar_kws={"label": "Jaccard overlap of top-20 TF-IDF terms"},
                linewidths=0.5, linecolor="white", annot_kws={"size": 7},
                vmin=0, vmax=0.3)
    ax.set_title("Lexical overlap between intents\n(high values predict classifier confusion)")
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    return _save(fig, "05_class_overlap", save)


# =====================================================================
# 6. Temporal patterns
# =====================================================================

def fig_temporal(tickets: pd.DataFrame, save: bool = True):
    """Q7: what temporal structure must the trend detector account for?"""
    daily = tickets.groupby("date").size()
    daily.index = pd.to_datetime(daily.index)
    ma7 = daily.rolling(7, min_periods=1).mean()

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    ax = axes[0, 0]
    ax.plot(daily.index, daily.values, lw=0.7, color="#b2bec3", label="daily")
    ax.plot(ma7.index, ma7.values, lw=2, color="#2c7873", label="7-day MA")
    ax.set_title("Daily ticket volume")
    ax.set_ylabel("tickets")
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=30, labelsize=8)

    ax = axes[0, 1]
    dow = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    counts = tickets["weekday"].value_counts().reindex(dow)
    colors = ["#2c7873"] * 5 + ["#e67e22", NEG_COLOR]
    ax.bar(range(7), counts.values, color=colors)
    ax.set_xticks(range(7))
    ax.set_xticklabels([d[:3] for d in dow])
    ax.set_title("Weekday seasonality")
    ax.set_ylabel("tickets")
    ax.text(6, counts.iloc[6] + 60, "policy says\nno Sunday support",
            ha="center", fontsize=7, color=NEG_COLOR)

    ax = axes[1, 0]
    hourly = tickets.groupby("hour").size()
    ax.bar(hourly.index, hourly.values, color="#2c7873")
    ax.set_title("Hour of day")
    ax.set_xlabel("hour (IST)")
    ax.set_ylabel("tickets")

    ax = axes[1, 1]
    recent = tickets[tickets["days_ago"] <= 35]
    pivot = (
        recent.groupby(["date", "intent"], observed=True).size().unstack(fill_value=0)
    )
    share = 100 * pivot.div(pivot.sum(axis=1), axis=0)
    for col, c in [("account_management", NEG_COLOR), ("technical_support", "#e67e22"),
                   ("order_tracking", "#b2bec3")]:
        if col in share.columns:
            ax.plot(pd.to_datetime(share.index), share[col].rolling(3, min_periods=1).mean(),
                    lw=2, label=col, color=c)
    ax.set_title("Recent intent share (planted trends T1/T2)")
    ax.set_ylabel("% of daily tickets")
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=30, labelsize=8)

    fig.suptitle("Strong weekday seasonality — trend detection must adjust for it")
    fig.tight_layout()
    return _save(fig, "06_temporal_patterns", save)


# =====================================================================
# 7. Corpus structure
# =====================================================================

def fig_corpus(corpus: pd.DataFrame, save: bool = True):
    """Q7: what do document lengths imply for chunk size?"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    per_doc = corpus.groupby("doc")["n_words"].sum().sort_values()
    axes[0].barh(per_doc.index, per_doc.values,
                 color=["#e67e22" if "ARCHIVED" in d or "eu_" in d else "#2c7873"
                        for d in per_doc.index])
    axes[0].set_xlabel("words")
    axes[0].set_title("Corpus size by document")
    axes[0].tick_params(labelsize=8)

    axes[1].hist(corpus["n_words"], bins=25, color="#2c7873", edgecolor="white")
    med = corpus["n_words"].median()
    axes[1].axvline(med, ls="--", c=NEG_COLOR, lw=1.5, label=f"median {med:.0f} words/page")
    axes[1].set_xlabel("words per page")
    axes[1].set_ylabel("pages")
    axes[1].set_title("Page length distribution")
    axes[1].legend(fontsize=8)

    fig.suptitle("47 pages across 13 documents — sizing input for the chunking ablation")
    fig.tight_layout()
    return _save(fig, "07_corpus_structure", save)


# =====================================================================
# 8. Leakage warning
# =====================================================================

def fig_confidence_leakage(tickets: pd.DataFrame, save: bool = True):
    """Documents the leakage found in the audit, so it cannot be forgotten."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for label, color, name in [
        (False, POS_COLOR, "resolved by AI"),
        (True, NEG_COLOR, "escalated to human"),
    ]:
        sub = tickets.loc[tickets["escalated"] == label, "confidence"]
        ax.hist(sub, bins=40, alpha=0.6, color=color, label=f"{name} (mean {sub.mean():.2f})")
    ax.set_xlabel("logged confidence")
    ax.set_ylabel("tickets")
    ax.set_title("LEAKAGE: confidence was generated from the outcome\n"
                 "Never use it as a feature to predict escalation")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, "08_confidence_leakage", save)
