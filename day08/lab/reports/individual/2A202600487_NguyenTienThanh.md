# Báo Cáo Cá Nhân — Lab Day 08: RAG Pipeline

**Họ và tên:** Nguyễn Tiến Thành
**Vai trò trong nhóm:** Eval Owner / Documentation Owner (Thành viên 3)
**Ngày nộp:** 13/04/2026
**Độ dài yêu cầu:** 500–800 từ

---

## 1. Tôi đã làm gì trong lab này? 

Đảm nhận vai trò Eval & Documentation Owner, công việc của tôi bao trùm từ bước chuẩn bị baseline đến chốt sổ kết quả.
- **Sprint 1:** Nghiên cứu 5 tài liệu chính sách (SOP, SLA, FAQ...) để xây dựng Ground Truth. Tôi soạn `test_questions.json` với 10 câu hỏi bao phủ nghiệp vụ và 2 câu hỏi "bẫy" (outlier) nhằm thử nghiệm tính năng Abstain.
- **Sprint 2:** Đánh giá bản Baseline rà soát, phát hiện điểm yếu chí mạng khi Dense Search thường xuyên bỏ sót các từ khóa kỹ thuật (như "ERR-403-AUTH").
- **Sprint 4:** Triển khai cơ chế *LLM-as-a-judge* bằng prompt cho GPT-4o-mini tự động chấm điểm các tiêu chí Faithfulness, Relevance và Completeness trong `eval.py`. Chạy luồng quét toàn bộ câu hỏi (A/B testing) so sánh Baseline và Variant (Hybrid), trích xuất CSV và hoàn thiện thiết kế kiến trúc (`architecture.md`) cùng nhật ký thực nghiệm (`tuning-log.md`).

---

## 2. Điều tôi hiểu rõ hơn sau lab này 

Sau bài lab, tôi thực sự thấu hiểu sâu sắc khái niệm **LLM-as-a-judge** và **Evaluation Loop**. Trái với suy nghĩ ban đầu rằng đánh giá AI phải dựa vào các công thức đếm từ vựng cứng nhắc (ROUGE/BLEU), việc vận dụng LLM đóng vai giám khảo chấm điểm dựa theo Rubric giúp tiết kiệm 80% thời gian đánh giá thủ công và mang lại sự tinh tế của góc nhìn con người.

Thứ hai là nguyên lý **Hybrid Retrieval & Reciprocal Rank Fusion (RRF)**. Tôi nhận ra Dense Search (Vector) rất giỏi hiểu ý định ngữ nghĩa, nhưng thường bị mù với từ khóa mã hóa (như `SLA P1` hay `v2026.1`). Việc lai tạp với thuật toán đếm từ khóa Sparse (BM25) giải quyết triệt để vấn đề này, và công thức RRF chính là cây cầu toán học hoàn hảo mang hai hệ quy chiếu khác biệt trộn thành một bảng xếp hạng (Ranking) duy nhất.

---

## 3. Điều tôi ngạc nhiên hoặc gặp khó khăn 

Phần khó khăn nhất và tốn nhiều thời gian debug nhất khi làm phần Evaluation là việc ép mô hình (GPT-4o-mini) trả về đúng định dạng JSON để chấm điểm theo cơ chế LLM-as-a-judge. Lúc đầu, AI thường xuyên dính tật "nhiễu lời", trả về cả chuỗi giải thích dài dòng hoặc bọc JSON trong Markdown block (như ` ```json... `) khiến cho các hàm parse data trong `eval.py` bị văng lỗi liên tục. Điều này buộc mình phải sử dụng kỹ thuật Regular Expression (RegEx) để bóc tách JSON và tinh chỉnh lại System Prompt.

Đổi lại, điều làm mình thực sự ngạc nhiên là sức mạnh của **Grounded Prompt**. Kỹ thuật Anti-Hallucination do Minh tinh chỉnh hoạt động trơn tru đến kỳ lạ: 100% các câu hỏi tạo bẫy mà mình soạn ra (hỏi về tham nhũng hay tiền ăn trưa) đều bị AI nhận diện và từ chối cung cấp thông tin cực kỳ chuyên nghiệp.

---

## 4. Phân tích một câu hỏi trong scorecard 

**Câu hỏi:** [Q10] "Theo thay đổi mới nhất (v2026.1), thời gian khắc phục sự cố P1 là bao lâu?"

**Phân tích:**
- Mức độ của Baseline (Dense): Hệ thống mắc kẹt ở Completeness = 3. Lỗi nằm tại quá trình Retrieval, khi mà Vector Search bị mờ mắt trước mã phiên bản. Nó kéo lên thông tin cũ ở những file nói về thời gian khắc phục P1 (là 6 giờ), nhưng lại vô tình bỏ qua tài liệu có bản đính chính mới nhất chỉ vì khoảng cách vector không đủ gần.
- Điểm vượt trội của Variant (Hybrid): Đạt điểm Completeness = 4, cải thiện cục diện hiển nhiên. Khối lệnh `rank_bm25` (Sparse Retrieval) đã cứu cánh bằng năng lực bắt dính chính xác keyword "v2026.1" của file tài liệu mới. Vectơ thì làm nhiệm vụ bám ý nghĩa "thời gian khắc phục P1". Qua hệ số hợp thể RRF, phiên bản tài liệu mới tinh ghi nhận mốc 4 giờ đã được đẩy lên đầu top 3 chunks gửi vào LLM. Đáp án sinh ra vì thế đầy đủ và cập nhật tuyệt đối.

---

## 5. Nếu có thêm thời gian, tôi sẽ làm gì? 

Nếu dự án được kéo dài thêm, tôi sẽ tập trung hoàn thiện cơ chế **Context Recall Scoring** tự động (Thay vì để kết quả N/A). Logic sẽ là sử dụng cơ chế String Matching gạch chéo ID source trả về của hệ thống đọ với tập Ground Truth "expected_sources" trong file JSON. Ngoài ra, tôi rất mong muốn được trải nghiệm kỹ thuật **Query Transformation (HyDE)** để kiểm chứng xem việc để AI tự ảo tưởng (hallucinate) ra đáp án mẫu trước rồi dùng nó đi tìm tài liệu, liệu nó có đạp đổ kỷ lục tra cứu của thuật toán Hybrid hiện tại hay không.
