"""The understanding layer.

Wraps intent classification, sentiment, urgency and entity extraction behind a
single call that returns one typed object.

    from src.understanding.pipeline import UnderstandingPipeline

    up = UnderstandingPipeline.load()
    u = up.understand("my laptop arrived damaged, order 12345")
    print(u.summary())

Runs entirely locally in a few milliseconds with no network call. That is the
point: intent routing and analytics do not need an LLM, and keeping them
independent means the dashboard's intent trend stays comparable when the LLM
behind the agent changes.
"""
from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np

from src.understanding import entities as ent
from src.understanding import sentiment as sent
from src.understanding.schema import Understanding

DEFAULT_MODEL = Path(__file__).resolve().parents[2] / "models" / "intent_classifier.joblib"

# Below this margin between the top two classes, the message is treated as
# potentially carrying more than one intent. 42% of the hard test set is
# genuinely compound, so this flag matters for routing.
MULTI_INTENT_MARGIN = 0.15


class UnderstandingPipeline:
    """Intent + sentiment + urgency + entities in one pass."""

    def __init__(self, model):
        self.model = model
        self.classes = list(model.named_steps["clf"].classes_)
        self._has_proba = hasattr(model.named_steps["clf"], "predict_proba")

    @classmethod
    def load(cls, path: Path | str = DEFAULT_MODEL) -> UnderstandingPipeline:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"No model at {path}. Run: python scripts/train_intent_classifier.py"
            )
        return cls(joblib.load(path))

    # ---------------------------------------------------------------
    def _intent_scores(self, text: str) -> list[tuple[str, float]]:
        """Class scores, normalised to sum to 1 regardless of estimator type.

        LinearSVC exposes `decision_function`, not `predict_proba`. Rather than
        wrap it in a calibrator (which would need a further holdout split on
        already-thin data), we softmax the margins. These are *scores*, not
        calibrated probabilities, and are labelled as such - proper calibration
        is Phase 11 work using multiple signals.
        """
        if self._has_proba:
            probs = self.model.predict_proba([text])[0]
        else:
            margins = self.model.decision_function([text])[0]
            margins = np.atleast_1d(margins)
            e = np.exp(margins - margins.max())
            probs = e / e.sum()
        pairs = sorted(zip(self.classes, probs), key=lambda x: -x[1])
        return [(str(c), float(p)) for c, p in pairs]

    # ---------------------------------------------------------------
    def understand(self, text: str) -> Understanding:
        t0 = time.perf_counter()

        scores = self._intent_scores(text)
        intent, conf = scores[0]
        runner_up = scores[1][1] if len(scores) > 1 else 0.0
        margin = conf - runner_up

        s = sent.score_sentiment(text, intent=intent)
        u = sent.score_urgency(text, intent=intent, sentiment=s.label)
        e = ent.extract(text)

        return Understanding(
            text=text,
            intent=intent,
            intent_confidence=round(conf, 4),
            intent_margin=round(margin, 4),
            intent_top3=[(c, round(p, 4)) for c, p in scores[:3]],
            sentiment=s.label,
            sentiment_score=s.score,
            sentiment_explain=s.explain(),
            urgency=u.label,
            urgency_score=u.score,
            urgency_explain=u.explain(),
            entities=e,
            is_multi_intent=margin < MULTI_INTENT_MARGIN,
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
        )

    def understand_batch(self, texts: list[str]) -> list[Understanding]:
        return [self.understand(t) for t in texts]


if __name__ == "__main__":
    up = UnderstandingPipeline.load()
    samples = [
        "where is my order PAC-2026-12345",
        "my laptop arrived damaged and i want a refund, order 12345",
        "this is the THIRD time. refund today or i'm doing a chargeback",
        "getting ERR-DP-0x004 on my Vision 27",
        "what is your return policy",
        "order kahan hai bro delivery kab hoga",
        "Where is my order and can I return it if it arrives tomorrow?",
        "hello",
    ]
    total = 0.0
    for s in samples:
        u = up.understand(s)
        total += u.latency_ms
        print(f"\n{s}")
        print(f"  {u.summary()}")
        print(f"  top3: {u.intent_top3}")
        print(f"  {u.latency_ms:.2f} ms")
    print(f"\nmean latency: {total / len(samples):.2f} ms")
