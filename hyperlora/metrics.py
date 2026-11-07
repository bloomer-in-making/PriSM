"""ROUGE metrics. The six tasks covered by this release are all generation
tasks, scored with ROUGE-1 / ROUGE-L."""
import numpy as np
from rouge_score import rouge_scorer


_SCORER = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)


def rouge_scores(pred: str, ref: str):
    s = _SCORER.score(ref, pred)
    return float(s["rouge1"].fmeasure), float(s["rougeL"].fmeasure)


def evaluate_predictions(task_name: str, preds, refs) -> dict:
    """ROUGE-1 and ROUGE-L (mean F1 over the dev set)."""
    if not preds:
        return {"rouge1": 0.0, "rougeL": 0.0}
    r1s, rLs = zip(*[rouge_scores(p, r) for p, r in zip(preds, refs)])
    return {
        "rouge1": float(np.mean(r1s)),
        "rougeL": float(np.mean(rLs)),
    }
