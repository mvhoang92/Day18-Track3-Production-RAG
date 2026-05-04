# Individual Reflection — Lab 18

**Tên:** Mai Việt Hoàng — MSSV: 2A202600476  
**Module phụ trách:** M1 (Advanced Chunking) + M2 (Hybrid Search)

---

## 1. Đóng góp kỹ thuật

- **Module đã implement:** M1 — Advanced Chunking Strategies, M2 — Hybrid Search
- **Các hàm/class chính đã viết:**

  **M1 — `src/m1_chunking.py`:**
  - `chunk_semantic()` — Encode câu bằng `all-MiniLM-L6-v2`, tính cosine similarity giữa các câu liên tiếp, tách chunk mới khi similarity < threshold
  - `chunk_hierarchical()` — Tạo parent chunks (gom paragraph đến `parent_size`), sau đó split mỗi parent thành child chunks (`child_size`), mỗi child có `parent_id` trỏ về parent
  - `chunk_structure_aware()` — Dùng regex split theo markdown headers (`#{1,3}`), ghép header + content thành chunk, lưu `section` vào metadata
  - `compare_strategies()` — Chạy cả 4 strategies, thu thập stats (num_chunks, avg/min/max length), in bảng so sánh

  **M2 — `src/m2_search.py`:**
  - `segment_vietnamese()` — Dùng `underthesea.word_tokenize` để tách từ tiếng Việt đúng ("nghỉ phép" = 1 token)
  - `BM25Search.index()` — Segment từng chunk → tokenize → build `BM25Okapi` index
  - `BM25Search.search()` — Segment query → `get_scores()` → sort → trả `SearchResult` với `method="bm25"`
  - `DenseSearch.index()` — Encode chunks bằng `bge-m3` → upload `PointStruct` lên Qdrant
  - `DenseSearch.search()` — Encode query → `client.search()` → trả `SearchResult` với `method="dense"`
  - `reciprocal_rank_fusion()` — Merge nhiều ranked lists: `score(d) = Σ 1/(k + rank_i(d))`, trả kết quả với `method="hybrid"`

- **Số tests pass:** 18/18 (M1: 13/13, M2: 5/5)

---

## 2. Kiến thức học được

- **Khái niệm mới nhất:** Hierarchical chunking — pattern index children (nhỏ, embedding chính xác) nhưng trả parent (đủ context) cho LLM. Đây là production pattern thực tế, khác hẳn cách chunk đơn giản đã học trước.
- **Điều bất ngờ nhất:** Reciprocal Rank Fusion (RRF) đơn giản đến bất ngờ — chỉ cộng `1/(k+rank)` từ mỗi list — nhưng lại hiệu quả hơn weighted sum vì không cần tune weight giữa BM25 và Dense.
- **Kết nối với bài giảng:** Semantic chunking liên quan trực tiếp đến phần "chunking strategies" trong slide Production RAG; BM25 + Dense + RRF là kiến trúc hybrid search được đề cập trong phần "retrieval pipeline".

---

## 3. Khó khăn & Cách giải quyết

- **Khó khăn lớn nhất:** Môi trường — ROS2 được cài system-wide gây conflict với pytest plugin, khiến không chạy được test. Ngoài ra Qdrant cần Docker nhưng không có quyền Docker.
- **Cách giải quyết:** Gỡ ROS2 (`sudo apt remove --purge ros-humble-*`). Với Qdrant, implement BM25-only fallback trong pipeline để test M2 không phụ thuộc vào Docker — các test cases của M2 chỉ test BM25 và RRF nên vẫn pass đầy đủ.
- **Thời gian debug:** ~20 phút cho vấn đề môi trường, ~10 phút fix logic BM25 (bỏ filter `score > 0` khiến kết quả rỗng với corpus nhỏ).

---

## 4. Nếu làm lại

- **Sẽ làm khác:** Thêm overlap giữa các child chunks trong hierarchical chunking (hiện tại non-overlapping) để tránh mất context ở ranh giới chunk.
- **Module muốn thử tiếp:** M3 (Reranking) — muốn thấy thực tế `bge-reranker-v2-m3` cải thiện precision bao nhiêu so với chỉ dùng BM25+Dense.

---

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | 4 |
| Code quality | 4 |
| Teamwork | 4 |
| Problem solving | 5 |
