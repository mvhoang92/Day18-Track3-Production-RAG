# Failure Analysis — Lab 18: Production RAG

**Nhóm:** TuyenTMC + HoangMV
**Thành viên:**
- HoangMV (2A202600476) → M1 (Chunking), M2 (Hybrid Search)
- TuyenTMC (2A202600324) → M3 (Reranking), M4 (Evaluation), M5 (Enrichment)

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | TBD | TBD | TBD |
| Answer Relevancy | TBD | TBD | TBD |
| Context Precision | TBD | TBD | TBD |
| Context Recall | TBD | TBD | TBD |

*Note: Run `python src/pipeline.py` to generate actual scores for ragas_report.json*

---

## Bottom-5 Failures

*Note: Bottom-5 will be populated from ragas_report.json after running pipeline*

### #1
- **Question:** [From ragas_report.json - bottom 1 by avg_score]
- **Expected:** [Ground truth from test_set.json]
- **Got:** [Actual answer from pipeline]
- **Worst metric:** [faithfulness/answer_relevancy/context_precision/context_recall]
- **Error Tree:** Output sai → Context đúng? → Query OK? →
- **Root cause:** [Based on worst metric diagnostic mapping]
- **Suggested fix:** [From diagnostic mapping in M4 implementation]

### #2
- **Question:** [From ragas_report.json - bottom 2 by avg_score]
- **Expected:** [Ground truth from test_set.json]
- **Got:** [Actual answer from pipeline]
- **Worst metric:** [faithfulness/answer_relevancy/context_precision/context_recall]
- **Error Tree:** Output sai → Context đúng? → Query OK? →
- **Root cause:** [Based on worst metric diagnostic mapping]
- **Suggested fix:** [From diagnostic mapping in M4 implementation]

### #3
- **Question:** [From ragas_report.json - bottom 3 by avg_score]
- **Expected:** [Ground truth from test_set.json]
- **Got:** [Actual answer from pipeline]
- **Worst metric:** [faithfulness/answer_relevancy/context_precision/context_recall]
- **Error Tree:** Output sai → Context đúng? → Query OK? →
- **Root cause:** [Based on worst metric diagnostic mapping]
- **Suggested fix:** [From diagnostic mapping in M4 implementation]

### #4
- **Question:** [From ragas_report.json - bottom 4 by avg_score]
- **Expected:** [Ground truth from test_set.json]
- **Got:** [Actual answer from pipeline]
- **Worst metric:** [faithfulness/answer_relevancy/context_precision/context_recall]
- **Error Tree:** Output sai → Context đúng? → Query OK? →
- **Root cause:** [Based on worst metric diagnostic mapping]
- **Suggested fix:** [From diagnostic mapping in M4 implementation]

### #5
- **Question:** [From ragas_report.json - bottom 5 by avg_score]
- **Expected:** [Ground truth from test_set.json]
- **Got:** [Actual answer from pipeline]
- **Worst metric:** [faithfulness/answer_relevancy/context_precision/context_recall]
- **Error Tree:** Output sai → Context đúng? → Query OK? →
- **Root cause:** [Based on worst metric diagnostic mapping]
- **Suggested fix:** [From diagnostic mapping in M4 implementation]

## Case Study (cho presentation)

**Question chọn phân tích:** [Sample question about "nghỉ phép" or "dữ liệu cá nhân"]

**Error Tree walkthrough:**
1. Output đúng? → [Check faithfulness score]
2. Context đúng? → [Check context_precision and context_recall]
3. Query rewrite OK? → [Check if vocabulary gap exists between query and docs]
4. Fix ở bước: [G (generation), R (reranking), A (answer), PreRAG (query expansion)]

**Nếu có thêm 1 giờ, sẽ optimize:**
- Integrate actual bge-reranker-v2-m3 model for M3 Reranking
- Run full pipeline with real Vietnamese documents from data/
- Generate RAGAS report to identify actual bottom-5 failures

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
