"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import os, sys, time
from dataclasses import dataclass
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


class CrossEncoderReranker:
    """Cross-encoder reranker using bge-reranker-v2-m3."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        """Load cross-encoder model (FlagReranker or CrossEncoder)."""
        if self._model is None:
            try:
                from FlagEmbedding import FlagReranker

                self._model = FlagReranker(self.model_name, use_fp16=True)
            except ImportError:
                try:
                    from sentence_transformers import CrossEncoder

                    self._model = CrossEncoder(self.model_name)
                except ImportError:
                    # Fallback: use basic scoring based on keyword overlap
                    self._model = None
        return self._model

    def _basic_rerank(self, query: str, documents: list[dict]) -> list[tuple]:
        """Basic reranking fallback using keyword overlap."""
        query_words = set(query.lower().split())
        doc_scores = []
        for doc in documents:
            doc_words = set(doc["text"].lower().split())
            overlap = len(query_words & doc_words)
            doc_scores.append((doc, overlap))
        return doc_scores

    def rerank(
        self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K
    ) -> list[RerankResult]:
        """Rerank documents: top-20 → top-k."""
        if not documents:
            return []

        model = self._load_model()

        if model is None:
            # Fallback: use basic keyword-based scoring
            doc_scores = self._basic_rerank(query, documents)
        else:
            pairs = [(query, doc["text"]) for doc in documents]
            try:
                scores = model.compute_score(pairs)
            except AttributeError:
                scores = model.predict(pairs)
            doc_scores = list(zip(documents, scores))

        # Sort by rerank score descending
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        # Return top_k results
        results = []
        for i, (doc, score) in enumerate(doc_scores[:top_k]):
            if isinstance(doc, dict):
                text = doc["text"]
                orig_score = doc.get("score", 0.0)
                metadata = doc.get("metadata", {})
            else:
                text = doc.text if hasattr(doc, "text") else str(doc)
                orig_score = 0.0
                metadata = {}
            results.append(
                RerankResult(
                    text=text,
                    original_score=orig_score,
                    rerank_score=float(score),
                    metadata=metadata,
                    rank=i + 1,
                )
            )

        return results


class FlashrankReranker:
    """Lightweight alternative reranker (<5ms)."""

    def __init__(self):
        self._model = None

    def _load_model(self):
        """Load flashrank model."""
        if self._model is None:
            from flashrank import Ranker

            self._model = Ranker()

    def rerank(
        self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K
    ) -> list[RerankResult]:
        """Rerank using flashrank."""
        if not documents:
            return []

        self._load_model()

        from flashrank import RerankRequest

        passages = [{"text": d["text"]} for d in documents]
        results = self._model.rerank(RerankRequest(query=query, passages=passages))

        # Convert to RerankResult
        rerank_results = []
        for i, item in enumerate(results[:top_k]):
            doc_idx = int(item["id"]) if item["id"].isdigit() else i
            rerank_results.append(
                RerankResult(
                    text=item["text"],
                    original_score=documents[doc_idx].get("score", 0.0)
                    if doc_idx < len(documents)
                    else 0.0,
                    rerank_score=float(item["score"]),
                    metadata={},
                    rank=i + 1,
                )
            )

        return rerank_results


def benchmark_reranker(
    reranker, query: str, documents: list[dict], n_runs: int = 5
) -> dict:
    """Benchmark latency over n_runs.

    Returns dict with avg_ms, min_ms, max_ms.
    """
    times = []

    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        elapsed_ms = (time.perf_counter() - start) * 1000
        times.append(elapsed_ms)

    return {
        "avg_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
    }


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")

    # Benchmark
    stats = benchmark_reranker(reranker, query, docs, n_runs=3)
    print(f"\nBenchmark: avg={stats['avg_ms']:.2f}ms, min={stats['min_ms']:.2f}ms, max={stats['max_ms']:.2f}ms")