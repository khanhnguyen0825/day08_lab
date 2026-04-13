# Báo Cáo Cá Nhân — Lab Day 08: RAG Pipeline

**Họ và tên:** Nguyễn Thành Đại Khánh  
**MSSV:** 2A202600404  
**Vai trò trong nhóm:** Tech Lead & Retrieval Owner (Thành viên 1)  
**Ngày nộp:** 2026-04-13  
**Độ dài yêu cầu:** 500–800 từ

---

## 1. Tôi đã làm gì trong lab này?

Với vai trò là **Tech Lead** và **Retrieval Owner**, tôi chịu trách nhiệm chính trong việc xây dựng "nền móng" dữ liệu và chiến lược tìm kiếm cho toàn bộ hệ thống. Các công việc cụ thể tôi đã thực hiện bao gồm:

*   **Xây dựng Indexing Pipeline (Sprint 1):** Tôi đã thiết kế logic chunking tại file `index.py`, sử dụng kỹ thuật tách đoạn theo Heading và Paragraph để đảm bảo ngữ cảnh không bị cắt vụn. Tôi cũng trực tiếp thực hiện phần trích xuất Metadata (source, department, effective_date) để phục vụ cho việc lọc dữ liệu nâng cao sau này.
*   **Tích hợp OpenAI Embeddings:** Tôi đã thực hiện kết nối với mô hình `text-embedding-3-small`, đồng thời xử lý lỗi **Dimension Mismatch (384 vs 1536)** khi chuyển đổi từ mô hình Local sang mô hình Cloud. Đây là bước quan trọng giúp team đồng bộ hóa kho dữ liệu và đảm bảo độ chính xác cho khâu retrieval.
*   **Tối ưu hóa Retrieval (Sprint 3):** Tôi đã cài đặt thư viện `rank-bm25` và xây dựng hàm **Hybrid Search** kết hợp giữa Dense Search (ngữ nghĩa) và Sparse Search (từ khóa). Tôi đã áp dụng thuật toán **Reciprocal Rank Fusion (RRF)** để trộn kết quả, giúp hệ thống nhặt được các mã lỗi kỹ thuật (như ERR-403) chính xác hơn so với chỉ dùng Dense Search đơn thuần.
*   **Quản trị và Đồng bộ Code:** Tôi chịu trách nhiệm merge code của các thành viên, quản lý cấu trúc file `requirements.txt`, và đảm bảo tính nhất quán của các tài liệu thực nghiệm trong `README.md` và `tuning-log.md`.

---

## 2. Điều tôi hiểu rõ hơn sau lab này

Sau thực hành, tôi ngộ ra rằng **kỹ thuật Retrieval là "linh hồn" của hệ thống RAG**. Một mô hình LLM mạnh như GPT-4o-mini cũng sẽ trở nên vô dụng nếu dữ liệu context đưa vào bị nhiễu hoặc không chính xác. Tôi đã hiểu rõ sự khác biệt giữa Vector Search (hiểu nghĩa) và Keyword Search (khớp chữ), cũng như cách kết hợp chúng để tối ưu cho các loại câu hỏi khác nhau của nhân viên IT/Helpdesk.

Tôi cũng học được tầm quan trọng của việc quản lý **Schema dữ liệu**. Việc thay đổi mô hình Embedding không chỉ đơn thuần là đổi một dòng code, mà nó kéo theo việc phải tái cấu trúc lại toàn bộ database vector. Khái niệm về Reciprocal Rank Fusion (RRF) cũng giúp tôi có cái nhìn mới về cách xếp hạng kết quả từ nhiều nguồn dữ liệu khác nhau.

---

## 3. Điều tôi ngạc nhiên hoặc gặp khó khăn

**Khó khăn lớn nhất** mà tôi gặp phải chính là lỗi lệch số chiều vector (Dimension Mismatch). Lúc đầu, tôi khá lúng túng khi hệ thống báo lỗi không thể query ChromaDB, sau đó tôi mới phát hiện ra database cũ dùng mô hình local (384 dims) không thể tương thích với mô hình OpenAI (1536 dims). Việc phải "flush" (xóa sạch) database và build lại từ đầu là một kinh nghiệm thực tế quý giá về bảo trì dữ liệu.

**Điều tôi ngạc nhiên** là sự đánh đổi (trade-off) trong Hybrid Search. Trước khi chạy Eval, tôi luôn nghĩ rằng càng kết hợp nhiều kỹ thuật (Hybrid + Rerank) thì điểm càng cao. Tuy nhiên, kết quả thực tế cho thấy đôi khi BM25 lại gây nhiễu cho các câu hỏi thuần về ngữ nghĩa. Điều này nhắc nhở tôi rằng trong AI, "phức tạp hơn chưa chắc đã tốt hơn".

---

## 4. Phân tích một câu hỏi trong scorecard

**Câu hỏi:** [Q01] "Tôi là nhân viên mới vào làm, ai sẽ phê duyệt quyền truy cập hệ thống của tôi và mức quyền của tôi là gì?"

**Phân tích:**
*   **Kết quả:** Baseline (Dense) đạt điểm Completeness là 5, trong khi Variant (Hybrid) chỉ đạt 3.
*   **Lý do Failure:** Câu hỏi này mang tính ngữ nghĩa thuần túy (semantic query). Người dùng đang hỏi về chính sách cho "nhân viên mới". Dense Search hoạt động cực kỳ hiệu quả vì nó hiểu được mối liên hệ giữa "nhân viên mới" và tài liệu Access Control SOP.
*   **Vấn đề của Hybrid:** Thuật toán BM25 (Sparse) đã quét trúng từ khóa "nhân viên" xuất hiện dày đặc trong các tài liệu không liên quan (như HR Policy, IT FAQ). Vì tần suất từ khóa cao, BM25 đã chấm điểm cao cho những đoạn văn bản "nhiễu" này, đẩy các đoạn văn bản quan trọng của Access Control ra khỏi Top-3 chunk được gửi vào LLM. 
*   **Bài học:** Qua câu này, tôi nhận ra rằng với các câu hỏi đàm thoại, đôi khi Dense Search mang lại kết quả cô đọng và chính xác hơn, và việc tinh chỉnh trọng số trộn (fusion weights) là cực kỳ quan trọng.

---

## 5. Nếu có thêm thời gian, tôi sẽ làm gì?

*   **Tinh chỉnh trọng số RRF:** Tôi sẽ thử nghiệm các tỷ lệ khác nhau (ví dụ: Dense 0.8, Sparse 0.2) thay vì chia đều, để ưu tiên ngữ nghĩa cho các câu hỏi hội thoại mà vẫn không bỏ lỡ từ khóa kỹ thuật.
*   **Áp dụng Metadata Filtering:** Tôi muốn tích hợp thêm bước lọc dữ liệu theo `Department` trước khi tìm kiếm. Nếu nhân viên hỏi về IT, hệ thống sẽ ưu tiên các tài liệu thuộc tag IT Security, giúp giảm nhiễu từ các văn bản HR như đã gặp ở câu Q01.
*   **Implement Reranker chuyên sâu:** Tôi sẽ thay thế hàm rerank placeholder hiện tại bằng một mô hình Cross-Encoder thực thụ để đảm bảo chỉ những thông tin chất lượng nhất mới được đưa vào prompt cho LLM.
