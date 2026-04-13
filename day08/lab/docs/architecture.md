# Kiến trúc hệ thống RAG - Helpdesk Assistant

## 1. Pipeline Tổng Quan

Hệ thống được ứng dụng mô hình RAG (Retrieval-Augmented Generation) kết hợp giữa Dense Search và Sparse Search (Hybrid Search). 

```mermaid
graph TD
    subgraph KNOWLEDGE BASE (index.py)
        A[Raw Docs txt/md/pdf] --> B(Preprocess & Chunking)
        B -->|OpenAI text-embedding-3-small| C[(ChromaDB Vector Store)]
        B -->|BM25| G[Sparse Index]
    end

    subgraph RAG SYSTEM (rag_answer.py)
        User[Nhân viên] -->|Nhập câu hỏi| D{Router / Retrieval}
        C -->|Dense Results| D
        G -->|Sparse Results| D
        D -->|Reciprocal Rank Fusion| E(Grounded Prompt)
        E -->|Call OpenAI GPT-4o-mini| F[Câu trả lời trích dẫn]
        F --> User
    end
```

## 2. Công nghệ sử dụng
1. **Vector Database**: `ChromaDB` (Persist local)
2. **Embeddings & LLM**: `OpenAI` (`text-embedding-3-small` và `gpt-4o-mini`).
3. **Keyword Search**: `rank-bm25` (sử dụng `BM25Okapi`).
4. **Đánh giá tự động**: Prompting sử dụng LLM-as-a-judge (tích hợp trong file `eval.py` tự chấm Faithfulness, Relevance, Completeness).
