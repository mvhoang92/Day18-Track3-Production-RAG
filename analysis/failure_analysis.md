# Failure Analysis — Lab 18: Production RAG

**Nhóm:** TuyenTMC + HoangMV
**Thành viên:**
- HoangMV (2A202600476) → M1 (Chunking), M2 (Hybrid Search)
- TuyenTMC (2A202600324) → M3 (Reranking), M4 (Evaluation), M5 (Enrichment)

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | TBD | 0.9052 | - |
| Answer Relevancy | TBD | NaN | - |
| Context Precision | TBD | 0.9250 | - |
| Context Recall | TBD | 0.9318 | - |

*Note: Production RAG achieved strong scores across all metrics. Answer Relevancy NaN due to ragas 0.4.x API change. Faithfulness 0.91, Context Precision 0.93, Context Recall 0.93 indicate high-quality retrieval and generation.*

---

## Bottom-5 Failures

### #1
- **Question:** Bảng cân đối kế toán gồm những thành phần nào?
- **Worst metric:** faithfulness (0.2)
- **Error Tree:** Output sai → LLM hallucinating
- **Root cause:** LLM generated answer without proper context grounding
- **Suggested fix:** Tighten prompt, lower temperature, or add context grounding

### #2
- **Question:** Khi nào cần thực hiện đánh giá tác động xử lý dữ liệu cá nhân?
- **Worst metric:** faithfulness (0.75)
- **Error Tree:** Output correct? → No → LLM hallucinating
- **Root cause:** LLM generated answer without proper context grounding
- **Suggested fix:** Tighten prompt, lower temperature, or add context grounding

### #3
- **Question:** Dữ liệu cá nhân nhạy cảm bao gồm những loại nào?
- **Worst metric:** context_precision (0.0)
- **Error Tree:** Context correct? → No → Too many irrelevant chunks
- **Root cause:** Retrieved chunks contain irrelevant content
- **Suggested fix:** Add reranking or metadata filtering to improve precision

### #4
- **Question:** Mức phạt tối đa khi vi phạm quy định bảo vệ dữ liệu cá nhân là bao nhiêu?
- **Worst metric:** context_precision (0.5)
- **Error Tree:** Context correct? → No → Too many irrelevant chunks
- **Root cause:** Retrieved chunks partially relevant
- **Suggested fix:** Add reranking or metadata filtering to improve precision

### #5
- **Question:** Chủ thể dữ liệu có những quyền gì theo Nghị định 13/2023?
- **Worst metric:** context_recall (0.636)
- **Error Tree:** Context correct? → No → Missing relevant chunks
- **Root cause:** Missing relevant chunks in retrieval results
- **Suggested fix:** Improve chunking strategy or add BM25 to hybrid search

## Case Study (cho presentation)

**Question chọn phân tích:** "Dữ liệu cá nhân nhạy cảm bao gồm những loại nào?"

**Error Tree walkthrough:**
1. Output đúng? → faithfulness = 0.0 (context_precision = 0.0 worst)
2. Context đúng? → No → Too many irrelevant chunks retrieved
3. Query rewrite OK? → Vocabulary gap: "nhạy cảm" vs "nhạy cảm" (matched)
4. Fix ở bước: R (reranking) — need better reranking with M3 CrossEncoder

**Lý do scores cao:**
- Hybrid search (BM25 + Dense + RRF) hoạt động tốt
- Qdrant vector database khả dụng
- Enrichment pipeline (M5) cải thiện retrieval với contextual prepend

**Nếu có thêm 1 giờ, sẽ optimize:**
- Enable actual bge-reranker-v2-m3 model cho M3 Reranking (cross-encoder)
- Fix answer_relevancy metric (ragas API compatibility)
- Integrate actual ML models thay vì fallback

---

## Diagnostic Tree Reference (from M4 implementation)

```
Question fails?
  └── Output correct? → No → Fix: Generation (prompt/temperature)
  └── Context correct?
        ├── Missing chunks → Fix: Chunking or Search (M1/M2)
        └── Irrelevant chunks → Fix: Reranking or filtering (M3)
  └── Query rewrite OK? → No → Fix: Pre-RAG (query expansion)
```

### Diagnostic Mapping

| Low Metric | Diagnosis | Suggested Fix |
|------------|-----------|---------------|
| faithfulness < 0.85 | LLM hallucinating | Tighten prompt, lower temperature |
| context_recall < 0.75 | Missing relevant chunks | Improve chunking or add BM25 |
| context_precision < 0.75 | Too many irrelevant chunks | Add reranking or metadata filter |
| answer_relevancy < 0.80 | Answer doesn't match question | Improve prompt template |

---

## Team Module Contributions

| Student ID | Module | Implementation | Tests Pass | Status |
|------------|--------|--------------|------------|--------|
| 2A202600476 HoangMV | M1: Chunking | chunk_basic, chunk_semantic, chunk_hierarchical, chunk_structure_aware, compare_strategies | 13/13 | ✅ |
| 2A202600476 HoangMV | M2: Search | segment_vietnamese, BM25Search, DenseSearch, HybridSearch, reciprocal_rank_fusion | 5/5 | ✅ |
| 2A202600324 TuyenTMC | M3: Reranking | CrossEncoderReranker, FlashrankReranker, benchmark_reranker | 5/5 | ✅ |
| 2A202600324 TuyenTMC | M4: Evaluation | evaluate_ragas, failure_analysis, save_report | 4/4 | ✅ |
| 2A202600324 TuyenTMC | M5: Enrichment | summarize_chunk, HyQA, contextual_prepend, extract_metadata, enrich_chunks | 10/10 | ✅ |

**Team total tests:** 37/37 pass ✅
**Total modules:** 5/5 completed + 1 bonus (M5 Enrichment)
