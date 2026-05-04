"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
from dataclasses import dataclass, fields
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> dict:
    """Run RAGAS evaluation with 4 metrics.

    Args:
        questions: List of question strings.
        answers: List of answer strings from the pipeline.
        contexts: List of context lists (retrieved chunks per question).
        ground_truths: List of ground truth answers.

    Returns:
        Dict with aggregate scores and per_question results.
    """
    try:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
        from datasets import Dataset

        # Create dataset
        dataset = Dataset.from_dict(
            {
                "question": questions,
                "answer": answers,
                "contexts": contexts,
                "ground_truth": ground_truths,
            }
        )

        # Run evaluation
        result = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
        )

        df = result.to_pandas()

        # Extract aggregate scores
        aggregate = {
            "faithfulness": float(df["faithfulness"].mean()),
            "answer_relevancy": float(df["answer_relevancy"].mean()),
            "context_precision": float(df["context_precision"].mean()),
            "context_recall": float(df["context_recall"].mean()),
        }

        # Per-question results
        per_question = []
        for _, row in df.iterrows():
            per_question.append(
                EvalResult(
                    question=row["question"],
                    answer=row["answer"],
                    contexts=row["contexts"]
                    if isinstance(row["contexts"], list)
                    else [row["contexts"]],
                    ground_truth=row["ground_truth"],
                    faithfulness=float(row["faithfulness"]),
                    answer_relevancy=float(row["answer_relevancy"]),
                    context_precision=float(row["context_precision"]),
                    context_recall=float(row["context_recall"]),
                )
            )

        return {**aggregate, "per_question": per_question}

    except ImportError:
        # Fallback if ragas not installed
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "per_question": [],
        }


def failure_analysis(
    eval_results: list[EvalResult], bottom_n: int = 10
) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree.

    Args:
        eval_results: List of EvalResult objects.
        bottom_n: Number of worst questions to analyze.

    Returns:
        List of failure analysis dicts with diagnosis and suggested fixes.
    """
    if not eval_results:
        return []

    # Calculate average score for each result and find worst
    scored_results = []
    for result in eval_results:
        avg_score = mean_ignore_none(
            result.faithfulness,
            result.answer_relevancy,
            result.context_precision,
            result.context_recall,
        )
        scored_results.append((avg_score, result))

    # Sort by average score ascending
    scored_results.sort(key=lambda x: x[0])

    # Take bottom N
    bottom_results = scored_results[:bottom_n]

    failures = []
    for avg_score, result in bottom_results:
        # Find worst metric
        metrics = {
            "faithfulness": result.faithfulness,
            "answer_relevancy": result.answer_relevancy,
            "context_precision": result.context_precision,
            "context_recall": result.context_recall,
        }

        worst_metric = min(metrics, key=metrics.get)
        worst_score = metrics[worst_metric]

        # Map to diagnosis based on worst metric
        diagnosis, suggested_fix = get_diagnosis(worst_metric, worst_score)

        failures.append(
            {
                "question": result.question,
                "worst_metric": worst_metric,
                "score": worst_score,
                "avg_score": avg_score,
                "diagnosis": diagnosis,
                "suggested_fix": suggested_fix,
            }
        )

    return failures


def mean_ignore_none(*values) -> float:
    """Calculate mean ignoring None values."""
    valid = [v for v in values if v is not None and v != 0]
    return sum(valid) / len(valid) if valid else 0.0


def get_diagnosis(metric: str, score: float) -> tuple[str, str]:
    """Map metric and score to diagnosis and suggested fix."""
    if metric == "faithfulness":
        if score < 0.85:
            return (
                "LLM hallucinating",
                "Tighten prompt, lower temperature, or add context grounding",
            )
        return ("Faithful", "No fix needed")

    elif metric == "context_recall":
        if score < 0.75:
            return (
                "Missing relevant chunks",
                "Improve chunking strategy or add BM25 to hybrid search",
            )
        return ("Context recall adequate", "No fix needed")

    elif metric == "context_precision":
        if score < 0.75:
            return (
                "Too many irrelevant chunks",
                "Add reranking or metadata filtering to improve precision",
            )
        return ("Context precision adequate", "No fix needed")

    elif metric == "answer_relevancy":
        if score < 0.80:
            return (
                "Answer doesn't match question",
                "Improve prompt template or add query rewriting",
            )
        return ("Answer relevancy adequate", "No fix needed")

    return ("Unknown", "Investigate further")


def save_report(
    results: dict, failures: list[dict], path: str = "ragas_report.json"
):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")