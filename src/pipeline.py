"""Production RAG Pipeline — Bài tập NHÓM: ghép M1+M2+M3+M4."""

import os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.m1_chunking import load_documents, chunk_hierarchical
from src.m2_search import BM25Search, HybridSearch, SearchResult
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import load_test_set, evaluate_ragas, failure_analysis, save_report
from src.m5_enrichment import enrich_chunks
from config import RERANK_TOP_K, OPENAI_API_KEY


def _try_hybrid_search(all_chunks: list[dict]):
    """Thử HybridSearch (cần Qdrant), fallback về BM25 nếu không có."""
    try:
        search = HybridSearch()
        search.index(all_chunks)
        print("  ✓ HybridSearch (BM25 + Dense) ready")
        return search, "hybrid"
    except Exception as e:
        print(f"  ⚠️  Qdrant không khả dụng ({e.__class__.__name__}), dùng BM25-only fallback")
        bm25 = BM25Search()
        bm25.index(all_chunks)
        print("  ✓ BM25Search ready")
        return bm25, "bm25"


def _generate_answer(query: str, contexts: list[str]) -> str:
    """
    Sinh câu trả lời từ contexts.
    Dùng OpenAI nếu có API key, fallback về extractive (trả context đầu tiên).
    """
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            context_str = "\n\n".join(contexts)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Bạn là trợ lý trả lời câu hỏi dựa trên tài liệu. "
                            "Trả lời CHỈ dựa trên context được cung cấp, bằng tiếng Việt. "
                            "Nếu context không có thông tin → trả lời 'Không tìm thấy thông tin trong tài liệu.'"
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Context:\n{context_str}\n\nCâu hỏi: {query}",
                    },
                ],
                max_tokens=300,
                temperature=0.1,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"  ⚠️  OpenAI error: {e}, dùng extractive fallback")

    # Extractive fallback: trả về context đầu tiên
    return contexts[0] if contexts else "Không tìm thấy thông tin."


def build_pipeline():
    """Build production RAG pipeline."""
    print("=" * 60)
    print("PRODUCTION RAG PIPELINE")
    print("=" * 60)

    # Step 1: Load & Chunk (M1)
    print("\n[1/4] Chunking documents (M1 — Hierarchical)...")
    docs = load_documents()
    if not docs:
        print("  ⚠️  Không tìm thấy documents trong data/")
    all_chunks = []
    for doc in docs:
        parents, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
        for child in children:
            all_chunks.append({
                "text": child.text,
                "metadata": {**child.metadata, "parent_id": child.parent_id},
            })
    print(f"  ✓ {len(all_chunks)} child chunks từ {len(docs)} documents")

    # Step 2: Enrichment (M5) — optional
    print("\n[2/4] Enrichment (M5)...")
    enriched = enrich_chunks(all_chunks, methods=["contextual"])
    if enriched:
        all_chunks = [{"text": e.enriched_text, "metadata": e.auto_metadata} for e in enriched]
        print(f"  ✓ Enriched {len(enriched)} chunks")
    else:
        print("  ⚠️  M5 not implemented — dùng raw chunks (fallback)")

    # Step 3: Index (M2)
    print("\n[3/4] Indexing (M2)...")
    search, search_mode = _try_hybrid_search(all_chunks)

    # Step 4: Reranker (M3)
    print("\n[4/4] Loading reranker (M3)...")
    reranker = CrossEncoderReranker()
    print(f"  ✓ Pipeline ready (search={search_mode})")

    return search, reranker


def run_query(query: str, search, reranker: CrossEncoderReranker) -> tuple[str, list[str]]:
    """Run single query through pipeline: Search → Rerank → Generate."""
    # Search
    results = search.search(query)
    docs = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results]

    # Rerank (M3) — fallback nếu chưa implement
    reranked = reranker.rerank(query, docs, top_k=RERANK_TOP_K)
    if reranked:
        contexts = [r.text for r in reranked]
    else:
        # M3 chưa implement → dùng top-3 từ search
        contexts = [r.text for r in results[:3]]

    # Generate answer
    answer = _generate_answer(query, contexts)
    return answer, contexts


def evaluate_pipeline(search, reranker: CrossEncoderReranker):
    """Run evaluation trên toàn bộ test set."""
    print("\n[Eval] Running queries on test set...")
    test_set = load_test_set()
    questions, answers, all_contexts, ground_truths = [], [], [], []

    timings = []
    for i, item in enumerate(test_set):
        t0 = time.perf_counter()
        answer, contexts = run_query(item["question"], search, reranker)
        timings.append((time.perf_counter() - t0) * 1000)

        questions.append(item["question"])
        answers.append(answer)
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        print(f"  [{i+1:02d}/{len(test_set)}] {item['question'][:55]}...")

    avg_ms = sum(timings) / len(timings) if timings else 0
    print(f"\n  ⏱  Avg query latency: {avg_ms:.0f}ms")

    print("\n[Eval] Running RAGAS evaluation (M4)...")
    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)

    print("\n" + "=" * 60)
    print("PRODUCTION RAG SCORES")
    print("=" * 60)
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        s = results.get(m, 0)
        status = "✓" if s >= 0.75 else "✗"
        print(f"  {status} {m:<25}: {s:.4f}")

    failures = failure_analysis(results.get("per_question", []))
    save_report(results, failures)
    return results


if __name__ == "__main__":
    start = time.time()
    search, reranker = build_pipeline()
    evaluate_pipeline(search, reranker)
    print(f"\nTotal: {time.time() - start:.1f}s")
