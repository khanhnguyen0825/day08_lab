# Nhật ký Tối ưu hóa Hệ thống RAG (Tuning Log)

## 1. Giới thiệu
Tài liệu này ghi nhận sự thay đổi về điểm số tổng thể của toàn bộ hệ thống trước (Baseline) và sau (Variant) khi áp dụng kỹ thuật tìm kiếm nâng cao ở Sprint 3.

## 2. Thông tin cấu hình
- **Baseline (Sprint 2)**: Dense Retrieval (Vector Search) sử dụng ChromaDB. Model: `text-embedding-3-small`. Lấy `top_k = 3`.
- **Variant (Sprint 3)**: Hybrid Retrieval. Kết hợp kết quả Dense Search (ChromaDB) và Keyword Search (Sparse - `rank-bm25`). Trộn điểm bằng Reciprocal Rank Fusion (RRF). Lấy `top_k = 3`, `use_rerank = True`.

## 3. Phân tích kết quả Scorecard A/B
Dưới đây là bảng tổng hợp tích hợp từ `results/scorecard_baseline.md` và `results/scorecard_variant.md`.

| Metric           | Baseline (Dense) | Variant (Hybrid+RRF) | Delta   |
|------------------|------------------|----------------------|---------|
| Faithfulness     |   5.00/5         |   5.00/5             |  **0**  |
| Answer Relevance |   5.00/5         |   5.00/5             |  **0**  |
| Context Recall   |   N/A            |   N/A                |   -     |
| Completeness     |   4.83/5         |   4.75/5             | **-0.08** |

**Kết quả per-question nổi bật:**

| Câu | Baseline (F/R/Rc/C) | Variant (F/R/Rc/C) | Winner   |
|-----|---------------------|--------------------|----------|
| Q01 | 5/5/0/5             | 5/5/0/3            | Baseline |
| Q10 | 5/5/0/3             | 5/5/0/4            | **Variant** |
| Q02–Q09, Q11–Q12 | 5/5/-/5 | 5/5/-/5 | Tie |

## 4. Phân tích và nhận xét
## 4. Phân tích và nhận xét

**Cả hai hệ thống đều đạt Faithfulness & Relevance hoàn hảo (5.00/5)**
- Điều này chứng minh Prompt Engineering của Minh (Anti-Hallucination + Citation forcing) hoạt động cực kỳ hiệu quả trên cả hai chế độ retrieval.
- Hai câu hỏi bẫy (Q11, Q12) đều bị từ chối đúng cách ở cả Baseline lẫn Variant — cơ chế Abstain hoạt động hoàn hảo.

**Trường hợp Variant vượt Baseline (Q10 - SLA v2026.1)**
- Câu hỏi hỏi về bản cập nhật SLA mới nhất (thay đổi từ 6 giờ → 4 giờ).
- Hybrid Search kết hợp BM25 đã bắt được từ khóa chính xác "v2026.1" trong tài liệu, giúp AI trả lời đầy đủ hơn (Completeness: **4** vs 3).
- Đây là minh chứng rõ ràng nhất cho giá trị của Sparse/Keyword Search trong corpus có mã phiên bản cụ thể.

**Trường hợp Baseline vượt Variant (Q01 - Access Control)**
- Câu Q01 hỏi về quy trình cấp quyền cho nhân viên mới — câu hỏi mang tính ngữ nghĩa thuần túy, không có từ khóa đặc biệt.
- Dense Search cho Completeness = 5, trong khi Hybrid làm nhiễu ranking → Completeness = 3.
- Đây là trade-off điển hình: Hybrid mạnh với keyword search nhưng đôi khi noise với câu ngữ nghĩa thông thường.

## 5. Kết luận

Kết quả A/B cho thấy hai hệ thống có hiệu năng tương đương nhau về tổng thể (delta Completeness chỉ -0.08). Tuy nhiên, Variant (Hybrid+RRF) vượt trội ở nhóm câu hỏi có **từ khóa kỹ thuật đặc thù** (mã phiên bản, tên sự cố). Nhóm quyết định giữ Variant làm cấu hình Production vì trong môi trường IT Helpdesk thực tế, nhân viên thường xuyên truy vấn bằng mã lỗi, tên policy và số phiên bản cụ thể.

