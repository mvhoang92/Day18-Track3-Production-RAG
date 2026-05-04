# Individual Reflection — Lab 18

**Tên:** Mai Việt Hoàng — MSSV: 2A202600476  
**Module phụ trách:** M1 (Advanced Chunking) + M2 (Hybrid Search)

---

## 1. Đóng góp kỹ thuật

- Module đã implement: M1 — Advanced Chunking Strategies, M2 — Hybrid Search
- Các hàm/class chính đã viết:
  - M1: `chunk_semantic()`, `chunk_hierarchical()`, `chunk_structure_aware()`, `compare_strategies()`
  - M2: `segment_vietnamese()`, `BM25Search.index()`, `BM25Search.search()`, `DenseSearch.index()`, `DenseSearch.search()`, `reciprocal_rank_fusion()`
- Số tests pass: 18/18 (M1: 13/13, M2: 5/5)

## 2. Kiến thức học được

- Khái niệm mới nhất: Hierarchical chunking — index children (nhỏ, embedding chính xác) nhưng trả parent (đủ context) cho LLM, giải quyết trade-off precision vs context
- Điều bất ngờ nhất: RRF chỉ cần `1/(k+rank)` mà không cần tune weight giữa BM25 và Dense, đơn giản nhưng hiệu quả hơn weighted sum
- Kết nối với bài giảng: Semantic chunking → "embedding-based splitting"; BM25+Dense+RRF → "hybrid retrieval pipeline"; hierarchical → "advanced indexing strategies"

## 3. Khó khăn & Cách giải quyết

- Khó khăn lớn nhất: ROS2 system-wide gây conflict với pytest plugin, Docker không có quyền nên không chạy được Qdrant
- Cách giải quyết: Gỡ ROS2 (`sudo apt remove --purge ros-humble-*`); implement BM25-only fallback trong pipeline khi Qdrant không khả dụng
- Thời gian debug: ~30 phút môi trường, ~10 phút fix BM25 empty results (bỏ filter `score > 0`)

## 4. Nếu làm lại

- Sẽ làm khác điều gì: Thêm overlap 20-30% giữa child chunks trong hierarchical để tránh mất context ở ranh giới; dùng dynamic threshold cho semantic chunking thay vì fixed
- Module nào muốn thử tiếp: M3 (Reranking) — muốn đo thực tế `bge-reranker-v2-m3` cải thiện precision bao nhiêu so với BM25+Dense

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | 5 |
| Code quality | 5 |
| Teamwork | 4 |
| Problem solving | 5 |

## 6. Tự chấm điểm cá nhân

| Tiêu chí | Điểm tối đa | Tự chấm | Lý do |
|----------|:-----------:|:-------:|-------|
| Module implementation đúng logic | 15 | 15 | Cả 4 strategies M1 và đầy đủ BM25+Dense+RRF M2 đều implement đúng logic, không hardcode |
| `pytest tests/test_m*.py` pass | 15 | 15 | 18/18 tests pass (M1: 13/13, M2: 5/5) |
| Vietnamese-specific handling | 10 | 10 | Dùng `underthesea` word_tokenize cho BM25, `bge-m3` cho Dense embedding tiếng Việt |
| Code quality: comments, type hints, clean | 10 | 9 | Type hints và docstrings đầy đủ, có fallback handling; trừ 1 vì chưa có overlap trong hierarchical |
| Tất cả TODO markers hoàn thành | 10 | 10 | Toàn bộ TODO trong m1_chunking.py và m2_search.py đã được implement |
| **Tổng** | **60** | **59** | |
