# Báo Cáo Cá Nhân — Lab Day 08: RAG Pipeline

**Họ và tên:** Đỗ Trọng Minh  
**Vai trò trong nhóm:** AI & System Lead / Tech Lead (Thành viên 2)  
**Ngày nộp:** 2026-04-13  
**Độ dài yêu cầu:** 500–800 từ

---

## 1. Tôi đã làm gì trong lab này?

Tôi theo sát Sprint 1 và Sprint 2 dưới tư cách là Tech Lead, đồng thời chịu trách nhiệm tinh chỉnh ở Sprint 3. 
Cụ thể, tôi đã:
- Thiết lập môi trường tối ưu (giảm tải các thư viện local nặng, tập trung xử lý cho OpenAI API trên `requirements.txt`).
- Hoàn thiện mã nguồn nối ghép trong `rag_answer.py`, thiết kế hàm `call_llm` để gọi OpenAI với `temperature = 0`.
- Thiết kế và nâng cấp hệ thống System Prompt (Grounded prompt): Ép LLM tuân thủ chặt ngặt các quy tắc về việc ghi nguồn `[1]`, trình bày `bullet points`, và chống "hallucination" (bịa đặt) kể cả khi hỏi những câu vượt quá dữ liệu hoặc hỏi cấu hình nhạy cảm.
- Sau khi Khánh (Retrieval) thực hiện thuật toán Hybrid search và đưa metadata, tôi bọc các metadata này (Department, Date) vào text context để gửi cho LLM. Công việc của tôi là nhịp cầu sống còn giúp luồng dữ liệu thô của Khánh được "nói" thành lời mượt mà, bàn giao để Thành chấm điểm.

---

## 2. Điều tôi hiểu rõ hơn sau lab này

Sau thực hành, tôi ngộ ra vai trò sống còn của **Grounded Prompt và Deep Sanitization**:
- Trước đây tôi chỉ nghĩ đưa Text vào là AI tự phân tích. Nhưng thực tế prompt engineering quyết định hành vi của Helpdesk. Việc ra lệnh "Anti-hallucination" và bắt đóng đinh citation dạng ngoặc vuông ở cuối câu là cách duy nhất ép tính "trung thực" vào RAG.
- Tôi cũng hiểu khái niệm **System Integration (Tích hợp hệ thống)**: ChromaDB đôi khi trả về những khoảng whitespace / unescaped characters từ file document PDF/TXT cũ. Dẫn đến lỗi HTTP 400 Json Parser ở tầng client OpenAI. Do đó, kỹ thuật `encode("utf-16", "surrogatepass")` giúp làm sạch luồng context từ retrieval đi sang generation.

---

## 3. Điều tôi ngạc nhiên hoặc gặp khó khăn

**Lỗi khó chịu nhất** mà tôi tốn nhiều công sức để debug là lỗi `HTTP 400 - We could not parse the JSON body`. Ban đầu tôi tưởng do code logic của mình, nhưng nguyên nhân thực sự nằm ở luồng Text data (pipeline dirty data). File policy text vốn có ký tự ẩn (surrogates) cắt không khéo, nên khi ghép vào JSON HTTP Request của OpenAI nó báo lỗi format.
**Sự ngạc nhiên lớn nhất:** Tôi không ngờ **Vector / Dense Search (Cosine Similarity)** đôi khi lại "ngu" đến vậy. Dù là mô hình OpenAI Embedding tiên tiến, nó vẫn trượt những keyword ngắn tẹo hoặc tên mã code như "ERR-403". Trực giác ban đầu của tôi là mô hình vector đắt tiền sẽ giải quyết mọi thứ, nhưng hóa ra ta vẫn cần BM25 truyền thống cứu giá!

---

## 4. Phân tích một câu hỏi trong scorecard

**Câu hỏi:** "ERR-403-AUTH là lỗi gì?"

**Phân tích:**
- Ở Baseline (Dense search 100%), đối với câu này, các score trả về khá thấp (`0.365`, `0.354`). Vector search tập trung truy tìm ý nghĩa câu chữ, nên một từ kỹ thuật khô khan như "ERR-403" không có nhiều vector context lân cận. Kết quả là Retrieval kéo lên các đoạn policy không liên quan lắm. Prompt của tôi làm rất tốt nhiệm vụ của nó: Nó chọn abstain (nói "Tôi không đủ dữ liệu") thay vì nói phét. Nhưng xét về độ hữu dụng của RAG là 0/5 Completeness.
- **Lỗi ở đâu:** Hoàn toàn ở bước Retrieval (bỏ qua exact match).
- **Variant có cải thiện không:** Có. Sau khi cập nhật Variant (Hybrid), thuật toán BM25 đã bắt trúng chính xác cụm từ cấu trúc dạng "-" là "ERR-403-AUTH". Điểm RRF kéo văn bản trúng đích lên Top 1, từ đó AI dễ dàng bóc tách thông tin và trả lời đúng quy trình tạo ticket xin quyền. Điểm số từ 0 nhảy vọt!

---

## 5. Nếu có thêm thời gian, tôi sẽ làm gì?

- **Áp dụng Hybrid Query Expansion:** Đôi khi nhân viên gọi "Approval Matrix" bằng tiếng Việt lóng là "bảng xin quyền". Tôi sẽ dùng LLM gõ riêng một nhịp để sinh từ đồng nghĩa ra thành 1 list trước khi thảy danh sách này vào hàm tìm kiếm.
- **Structured JSON Output:** Tôi sẽ ép hàm `call_llm` nhả return bằng chuẩn `Format Schema JSON` native thay vì Markdown, để cho lập trình viên Front-end (hoặc các Agent AI khác ở Day 09) thao tác trích data trực tiếp (ví dụ bóc tách action required = "tạo_ticket").
