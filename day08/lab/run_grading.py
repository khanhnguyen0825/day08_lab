import json
import os
from datetime import datetime
from pathlib import Path
from rag_answer import rag_answer

# Cấu hình đường dẫn
GRADING_QUESTIONS_PATH = Path("rag_eval_questions.txt")
GRADING_LOG_PATH = Path("logs/grading_run.json")

def run_grading():
    # 1. Đảm bảo thư mục logs tồn tại
    GRADING_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # 2. Load câu hỏi
    try:
        with open(GRADING_QUESTIONS_PATH, "r", encoding="utf-8") as f:
            questions = json.load(f)
    except Exception as e:
        print(f"Lỗi khi load file câu hỏi: {e}")
        return

    print(f"--- Starting Grading Run with {len(questions)} questions ---")
    
    grading_log = []
    
    for q in questions:
        qid = q["id"]
        question_text = q["question"]
        
        print(f"Processing {qid}...")
        
        # Chạy pipeline ở chế độ tốt nhất: Hybrid
        # Tăng top_k_select lên 5 để đảm bảo an toàn cho các câu hỏi khó (gq02, gq06)
        try:
            result = rag_answer(
                query=question_text,
                retrieval_mode="hybrid",
                top_k_search=15, 
                top_k_select=5,
                use_rerank=False,
                verbose=False
            )
            
            # Format log đúng chuẩn SCORING.md
            entry = {
                "id": qid,
                "question": question_text,
                "answer": result["answer"],
                "sources": result["sources"],
                "chunks_retrieved": len(result["chunks_used"]),
                "retrieval_mode": "hybrid",
                "timestamp": datetime.now().isoformat()
            }
            grading_log.append(entry)
            
        except Exception as e:
            print(f"Lỗi xử lý câu {qid}: {e}")
            grading_log.append({
                "id": qid,
                "question": question_text,
                "answer": f"PIPELINE_ERROR: {str(e)}",
                "sources": [],
                "chunks_retrieved": 0,
                "retrieval_mode": "hybrid",
                "timestamp": datetime.now().isoformat()
            })

    # 3. Lưu file log
    with open(GRADING_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(grading_log, f, ensure_ascii=False, indent=2)
    
    print(f"--- Hoàn thành! File log đã được lưu tại: {GRADING_LOG_PATH} ---")

if __name__ == "__main__":
    run_grading()
