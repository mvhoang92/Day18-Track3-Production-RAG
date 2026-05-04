"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os, sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """
    Segment Vietnamese text thành words dùng underthesea.
    BM25 cần word boundaries đúng: "nghỉ phép" = 1 token, không phải 2.
    """
    try:
        from underthesea import word_tokenize
        return word_tokenize(text, format="text")
    except ImportError:
        # Fallback nếu chưa cài underthesea
        return text


class BM25Search:
    def __init__(self):
        self.corpus_tokens: list[list[str]] = []
        self.documents: list[dict] = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index từ danh sách chunks."""
        self.documents = chunks

        from rank_bm25 import BM25Okapi
        self.corpus_tokens = [
            segment_vietnamese(chunk["text"]).split()
            for chunk in chunks
        ]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Tìm kiếm bằng BM25."""
        if not self.documents:
            return []

        tokenized_query = segment_vietnamese(query).split()
        bm25_scores = self.bm25.get_scores(tokenized_query)

        top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            results.append(SearchResult(
                text=self.documents[idx]["text"],
                score=float(bm25_scores[idx]),
                metadata=self.documents[idx].get("metadata", {}),
                method="bm25",
            ))

        return results


class DenseSearch:
    def __init__(self):
        from qdrant_client import QdrantClient
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self._encoder = None

    def _get_encoder(self):
        """Lazy load encoder để tránh load model khi không cần."""
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(EMBEDDING_MODEL)  # BAAI/bge-m3 -> 1024 dim
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks vào Qdrant vector database."""
        from qdrant_client.models import Distance, VectorParams, PointStruct

        # Tạo (hoặc recreate) collection
        self.client.recreate_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )

        # Encode tất cả texts thành vectors
        texts = [c["text"] for c in chunks]
        vectors = self._get_encoder().encode(texts, show_progress_bar=True)

        # Upload lên Qdrant
        points = [
            PointStruct(
                id=i,
                vector=v.tolist(),
                payload={**c.get("metadata", {}), "text": c["text"]},
            )
            for i, (c, v) in enumerate(zip(chunks, vectors))
        ]
        self.client.upsert(collection_name=collection, points=points)
        print(f"Indexed {len(points)} chunks into '{collection}'")

    def search(self, query: str, top_k: int = DENSE_TOP_K,
               collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Tìm kiếm bằng dense vector similarity."""
        # Encode query thành vector
        query_vector = self._get_encoder().encode(query).tolist()

        # Tìm kiếm trong Qdrant
        hits = self.client.query_points(
            collection_name=collection,
            query=query_vector,
            limit=top_k,
        ).points

        return [
            SearchResult(
                text=hit.payload.get("text", ""),
                score=hit.score,
                metadata={k: v for k, v in hit.payload.items() if k != "text"},
                method="dense",
            )
            for hit in hits
        ]


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """
    Merge nhiều ranked lists bằng RRF.
    score(d) = Σ 1/(k + rank_i(d))

    Args:
        results_list: Danh sách các ranked lists (BM25, Dense, ...)
        k: Hằng số RRF (thường = 60)
        top_k: Số kết quả trả về

    Returns:
        Merged list với method="hybrid", sorted by RRF score.
    """
    # rrf_scores: text → {"score": float, "result": SearchResult}
    rrf_scores: dict[str, dict] = {}

    for result_list in results_list:
        for rank, result in enumerate(result_list):
            key = result.text
            if key not in rrf_scores:
                rrf_scores[key] = {"score": 0.0, "result": result}
            # Cộng dồn RRF score từ mỗi list
            rrf_scores[key]["score"] += 1.0 / (k + rank + 1)

    # Sort theo RRF score giảm dần
    sorted_items = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)

    # Trả về top_k kết quả với method="hybrid"
    merged = []
    for item in sorted_items[:top_k]:
        r = item["result"]
        merged.append(SearchResult(
            text=r.text,
            score=item["score"],
            metadata=r.metadata,
            method="hybrid",
        ))

    return merged


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print(f"Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
