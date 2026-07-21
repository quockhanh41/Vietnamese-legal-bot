"""Query condensation for multi-turn chat (M4.1).

Rewrites a follow-up into a standalone question BEFORE routing, so every
downstream route stays history-unaware. Pure function — the eval harness calls
it directly. Fail-open: empty history skips the LLM; any error/empty output
returns the original question."""
from __future__ import annotations

from typing import Dict, List

from src.observability import observe
from src.rag.orchestrator import MODEL_QUERY_TRANSFORM

CONDENSE_PROMPT = """Bạn viết lại CÂU HỎI TIẾP THEO thành một câu hỏi ĐỘC LẬP, đầy đủ ngữ cảnh, dựa trên lịch sử hội thoại.

Quy tắc:
- Thay đại từ / tham chiếu ("vậy còn...", "nó", "trường hợp đó", "loại này") bằng chủ thể cụ thể lấy từ lịch sử.
- Nếu câu hỏi ĐÃ độc lập (không phụ thuộc lịch sử), giữ NGUYÊN VĂN.
- Chỉ xuất ra ĐÚNG MỘT câu hỏi tiếng Việt. Không giải thích, không tiền tố, không dấu ngoặc kép.

Lịch sử hội thoại:
{history}

Câu hỏi tiếp theo: {question}

Câu hỏi độc lập:"""


def _format_history(history: List[Dict]) -> str:
    lines = []
    for turn in history:
        q = (turn.get("question") or "").strip()
        a = (turn.get("answer") or "").strip()
        if q:
            lines.append(f"Người dùng: {q}")
        if a:
            lines.append(f"Trợ lý: {a}")
    return "\n".join(lines)


@observe
def condense_query(history: List[Dict], question: str, llm) -> str:
    if not history:
        return question
    try:
        rewritten = llm.generate(
            CONDENSE_PROMPT.format(history=_format_history(history), question=question),
            model=MODEL_QUERY_TRANSFORM,
            temperature=0,
            max_tokens=120,
            reasoning_effort="none",
        )
    except Exception:
        return question
    rewritten = (rewritten or "").strip()
    return rewritten or question
