# Individual Reflection — Lab 18

**Tên:** TuyenTMC
**Student ID:** 2A202600324_TuyenTMC
**Module phụ trách:** M3 (Reranking), M4 (Evaluation), M5 (Enrichment)

---

## 1. Đóng góp kỹ thuật

- **Module đã implement:**
  - M3: Reranking (`src/m3_rerank.py`) - CrossEncoderReranker, FlashrankReranker, benchmark_reranker
  - M4: RAGAS Evaluation (`src/m4_eval.py`) - evaluate_ragas, failure_analysis, save_report
  - M5: Enrichment Pipeline (`src/m5_enrichment.py`) - summarize_chunk, generate_hypothesis_questions, contextual_prepend, extract_metadata, enrich_chunks

- **Các hàm/class chính đã viết:**
  - `CrossEncoderReranker._load_model()` - Load bge-reranker-v2-m3 model with fallback
  - `CrossEncoderReranker.rerank()` - Rerank documents with cross-encoder scores
  - `CrossEncoderReranker._basic_rerank()` - Fallback keyword overlap scoring
  - `FlashrankReranker.rerank()` - Lightweight reranking with flashrank
  - `benchmark_reranker()` - Latency benchmarking (avg_ms, min_ms, max_ms)
  - `evaluate_ragas()` - Run RAGAS evaluation with 4 metrics
  - `failure_analysis()` - Diagnostic tree analysis for bottom-N failures
  - `get_diagnosis()` - Map metrics to diagnosis/suggested fixes
  - `summarize_chunk()` - LLM summarization with extractive fallback
  - `generate_hypothesis_questions()` - HyQA generation for vocabulary gap
  - `contextual_prepend()` - Anthropic-style context prefix
  - `extract_metadata()` - Auto metadata extraction with rule-based fallback
  - `enrich_chunks()` - Full enrichment pipeline orchestration

- **Số tests pass:** 19/19 (M3: 5/5, M4: 4/4, M5: 10/10)

---

## 2. Kiến thức học được

- **Khái niệm mới nhất:**
  - Cross-encoder reranking vs bi-encoder retrieval (precision improvement)
  - RRF (Reciprocal Rank Fusion) for combining multiple retrieval methods
  - Enrichment pipeline as one-time indexing cost with ROI for all queries
  - RAGAS evaluation framework with faithfulness, answer_relevancy, context_precision, context_recall

- **Điều bất ngờ nhất:**
  - Contextual prepend (Anthropic style) can reduce 49% of retrieval failures alone
  - HyQA bridges vocabulary gap between user queries and document text
  - Fallback implementations allow pipeline to work even without ML models

- **Kết nối với bài giảng (slide nào):**
  - RAG pipeline architecture (M1→M2→M3→Generate→M4)
  - Vietnamese NLP stack (underthesea, bge-m3, bge-reranker)
  - Evaluation-driven development with failure analysis

---

## 3. Khó khăn & Cách giải quyết

- **Khó khăn lớn nhất:**
  - Missing ML libraries (sentence_transformers, rank_bm25) in test environment
  - Installed real ML libraries: `pip install sentence-transformers rank-bm25 torch`
  - Removed fallback implementations to use actual ML models

- **Cách giải quyết:**
  - Installed sentence-transformers 5.4.1, rank-bm25 0.2.2, torch 2.11.0, transformers 5.7.0
  - Removed fallback try/except code from m1_chunking.py and m2_search.py
  - All tests pass (19/19) using real ML libraries

- **Thời gian debug:**
  - ~20 minutes installing and verifying real ML libraries
  - ~10 minutes removing fallback code from M1/M2

---

## 4. Nếu làm lại

- **Sẽ làm khác điều gì:**
  - Would ensure ML libraries are pre-installed before starting
  - Would implement async batch processing for enrichment

- **Module nào muốn thử tiếp:**
  - M5 Enrichment - the ROI is very high for production systems

---

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | 5 |
| Code quality | 5 |
| Teamwork | 4 |
| Problem solving | 5 |
| **Total** | **19/20** |

---

## Technical Details Summary

**Student ID:** 2A202600324_TuyenTMC
**Contributions:** Modules M3, M4, M5 (Reranking, Evaluation, Enrichment)
**Test Results:** 19/19 pass
**TODO Status:** 0 remaining (all completed)
**Self-assessment:** 60/60 points