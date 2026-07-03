"""
Generate synthetic test queries for chunking evaluation.

Uses LLM with FULL document content to generate realistic, unbiased test questions.
Each question references a specific article, which is then used to extract
ground truth context — independent of any particular chunking strategy.

Usage:
    python scripts/generate_eval_queries.py --docs 20 --output data/eval/test_queries.json
"""

import json
import re
import sys
import os
import random
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import click
import pyarrow.dataset as ds
from tqdm import tqdm

from src.data.chunker import create_chunker
from src.data.constants import LEGAL_TYPES, MIN_CONTENT_LENGTH
from src.llm.gateway import LLMGateway

TEST_QUERY_GENERATION_PROMPT = """Bạn là chuyên gia tạo câu hỏi kiểm tra cho hệ thống tra cứu văn bản pháp luật Việt Nam.

Dựa trên TOÀN BỘ văn bản pháp luật dưới đây, hãy tạo {n_questions} câu hỏi mà NGƯỜI DÙNG THỰC TẾ sẽ hỏi khi tra cứu pháp luật.

Hãy tưởng tượng bạn là: luật sư, doanh nhân, kế toán, người dân đang cần tìm hiểu quy định.

QUAN TRỌNG - TRÁNH các loại câu hỏi sau (đây là câu hỏi "mẫu", không ai thực sự hỏi):
- Ngày ban hành, ngày hiệu lực của văn bản
- Văn bản này dựa trên những căn cứ pháp lý nào?
- Ai là người đề nghị/soạn thảo ban hành?
- Cơ quan nào ban hành văn bản này?
- Văn bản này có nội dung chính là gì? (quá chung chung)

TẬP TRUNG vào các câu hỏi có giá trị thực tiễn:
- Điều kiện để được hưởng/miễn/giảm một chế độ
- Trình tự, thủ tục thực hiện một quyền/nghĩa vụ cụ thể
- Mức phí, lệ phí, mức phạt áp dụng
- Ai thuộc đối tượng áp dụng, ai được miễn
- Hồ sơ, giấy tờ cần chuẩn bị
- Thời hạn, thời gian giải quyết
- Trách nhiệm của cơ quan/tổ chức/cá nhân trong một quy trình

Với mỗi câu hỏi, cung cấp:
- "question": câu hỏi bằng ngôn ngữ tự nhiên, như cách người dùng thực sự hỏi
- "question_type": "factoid" (tra cứu thông tin cụ thể) | "definition" (định nghĩa) | "procedural" (trình tự/thủ tục) | "cross_article" (liên quan nhiều điều khoản)
- "ground_truth_answer": câu trả lời chính xác, đầy đủ dựa trên văn bản
- "source_article": Điều khoản chứa câu trả lời (vd: "Điều 5. Quyền và nghĩa vụ của..." hoặc "Khoản 2 Điều 3...")

VĂN BẢN:
---
{context}
---

Trả lời bằng JSON array:
[
  {{
    "question": "...",
    "question_type": "...",
    "ground_truth_answer": "...",
    "source_article": "..."
  }}
]"""


def _extract_article_text(doc_content: str, article_ref: str) -> str:
    """
    Extract the text of a referenced article from the document content.
    Uses regex to find the article by its reference string.
    """
    if not article_ref or not doc_content:
        return ""

    # Normalize: extract just the article number/name
    # e.g. "Điều 5. Quyền và nghĩa vụ" or "Khoản 2 Điều 3"
    art_clean = article_ref.strip().replace("\n", " ")

    # Try exact match first
    idx = doc_content.find(art_clean)
    if idx >= 0:
        # Extract from this point to the next article or end
        end_idx = len(doc_content)
        next_article = re.search(r'\nĐiều \d+\.', doc_content[idx + len(art_clean):])
        if next_article:
            end_idx = idx + len(art_clean) + next_article.start()
        return doc_content[idx:end_idx].strip()

    # Try fuzzy: search for "Điều X" pattern
    art_num_match = re.search(r'Điều\s+(\d+)', art_clean)
    if art_num_match:
        art_num = art_num_match.group(1)
        pattern = rf'(Điều\s+{art_num}\.)[^\n]*\n'
        match = re.search(pattern, doc_content)
        if match:
            start = match.start()
            next_art = re.search(rf'\nĐiều\s+(?!{art_num}\b)\d+\.', doc_content[start + len(match.group()):])
            end = start + len(match.group()) + (next_art.start() if next_art else len(doc_content) - start - len(match.group()))
            return doc_content[start:end].strip()

    return ""


def generate_queries(
    data_path: str = "data/raw/legal_docs_merged.parquet",
    num_docs: int = 20,
    queries_per_doc: int = 2,
    output: str = "data/eval/test_queries.json",
    seed: int = 42,
    limit_legal_types: bool = True,
    max_doc_chars: int = 12000,
):
    """
    Generate synthetic test queries from FULL legal documents (no chunk bias).

    Args:
        data_path: Path to parquet file
        num_docs: Number of documents to sample
        queries_per_doc: Target queries per document (1-4)
        output: Output JSON file path
        seed: Random seed
        limit_legal_types: Only use Luật/Bộ luật/Nghị định/Thông tư/Nghị quyết/Quyết định
        max_doc_chars: Max document chars to pass to LLM (avoid token overflow)
    """
    random.seed(seed)

    dataset = ds.dataset(data_path, format="parquet")
    total = dataset.count_rows()
    print(f"Dataset has {total:,} total rows, sampling up to {num_docs} documents...")

    llm = LLMGateway()

    # Target doc size: enough content for good questions, but fits within Groq TPM
    # Groq free tier: 12K TPM. With 30s delay, ~3K tokens/doc is safe
    MIN_DOC_CHARS = 2000
    MAX_DOC_CHARS = 6000  # Keep docs short to respect rate limits

    # Collect full documents of manageable size
    selected_docs = []
    seen_legal_types = set()
    for batch in dataset.to_batches(batch_size=200):
        if len(selected_docs) >= num_docs:
            break

        batch_df = batch.to_pandas()
        rows = batch_df.to_dict("records")
        random.shuffle(rows)

        for row in rows:
            if len(selected_docs) >= num_docs:
                break

            content = row.get("content", "")
            # Filter: substantive but not too long
            if not content or len(content) < MIN_DOC_CHARS:
                continue
            if len(content) > MAX_DOC_CHARS:
                continue
            if limit_legal_types and row.get("legal_type") not in LEGAL_TYPES:
                continue

            selected_docs.append(row)
            seen_legal_types.add(row.get("legal_type", ""))

    print(f"Selected {len(selected_docs)} documents from: {seen_legal_types}")
    doc_lengths = [len(d["content"]) for d in selected_docs]
    print(f"Doc lengths: min={min(doc_lengths)}, max={max(doc_lengths)}, "
          f"avg={sum(doc_lengths)//len(doc_lengths)}")

    # Use article-level chunker only for extracting ground_truth_contexts
    article_chunker = create_chunker("article_level", min_article_length=100)

    # Generate queries from each full document
    all_queries = []
    for i, doc in enumerate(tqdm(selected_docs, desc="Generating queries from full docs")):
        n_q = random.choice([2, 2, 3, 3, 4])  # Bias towards 2-3 queries per doc
        n_q = min(n_q, queries_per_doc + 2)

        full_content = doc["content"]
        # Truncate if needed (keep preamble + as many articles as fit)
        if len(full_content) > max_doc_chars:
            context = full_content[:max_doc_chars]
            # Try to cut at a clean article boundary
            last_article = context.rfind("\nĐiều ")
            if last_article > max_doc_chars * 0.7:
                context = context[:last_article]
        else:
            context = full_content

        prompt = TEST_QUERY_GENERATION_PROMPT.format(
            n_questions=n_q,
            context=context,
        )

        # Rate-limit delay: Groq free tier has 12K TPM, each doc uses ~2-4K tokens
        if i > 0:
            wait = 25 + random.uniform(0, 10)
            print(f"  ⏳ Rate-limit wait {wait:.0f}s...")
            time.sleep(wait)

        # Retry loop with backoff
        generated = None
        for attempt in range(3):
            try:
                raw = llm.generate(
                    prompt=prompt,
                    temperature=0.7,
                    max_tokens=1024,  # Lower to save output tokens
                    json_mode=True,
                )
                generated = json.loads(raw)
                break
            except Exception as e:
                err = str(e)
                if "rate" in err.lower() or "429" in err:
                    wait = 30 * (attempt + 1)
                    print(f"  ⚠ Rate limited, waiting {wait}s (attempt {attempt+1}/3)...")
                    time.sleep(wait)
                else:
                    print(f"  ✗ LLM error (attempt {attempt+1}/3): {e}")
                    time.sleep(5)

        if generated is not None:
            if isinstance(generated, dict):
                generated = [generated]

            for q in generated:
                question = q.get("question", "")
                answer = q.get("ground_truth_answer", "")
                source_article = q.get("source_article", "")

                # Filter: skip empty or too-short questions
                if not question or not answer:
                    continue
                if len(question) < 15 or len(answer) < 30:
                    continue

                # Filter: skip template-style questions
                template_patterns = [
                    r'dựa trên (những )?(căn cứ|luật|văn bản) nào',
                    r'ai là người (đề nghị|soạn thảo|ban hành|ký)',
                    r'(được )?ban hành (vào |từ )?(ngày|năm|khi) nào',
                    r'(căn cứ|văn bản) pháp lý nào',
                    r'có (nội dung|mục đích) (chính |cơ bản )?là gì',
                ]
                is_template = any(re.search(p, question.lower(), re.IGNORECASE)
                                  for p in template_patterns)
                if is_template:
                    print(f"  ⏭ Skipping template: {question[:80]}...")
                    continue

                # Extract ground_truth_contexts using article reference
                gt_contexts = []
                if source_article:
                    art_text = _extract_article_text(full_content, source_article)
                    if art_text:
                        gt_contexts.append(art_text)
                    else:
                        # Fallback: chunk the doc and find matching article
                        chunks = article_chunker.chunk_document(doc)
                        for c in chunks:
                            if c.article and source_article[:30] in c.article:
                                gt_contexts.append(c.chunk_text)
                                break

                if not gt_contexts:
                    # Last resort: include the full visible context
                    gt_contexts = [context[:3000]]

                all_queries.append({
                    "question": question,
                    "ground_truth_answer": answer,
                    "ground_truth_contexts": gt_contexts,
                    "question_type": q.get("question_type", "factoid"),
                    "source_doc_id": doc["id"],
                    "source_article": source_article,
                    "source_title": doc["title"],
                    "source_legal_type": doc.get("legal_type", ""),
                })
        else:
            print(f"  ✗ Failed doc {i}: all 3 attempts exhausted")

    # Save
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(all_queries, f, ensure_ascii=False, indent=2)

    # Report
    type_counts = {}
    legal_type_counts = {}
    for q in all_queries:
        t = q["question_type"]
        type_counts[t] = type_counts.get(t, 0) + 1
        lt = q["source_legal_type"]
        legal_type_counts[lt] = legal_type_counts.get(lt, 0) + 1

    print(f"\n{'='*60}")
    print(f"Generated {len(all_queries)} test queries → {output}")
    print(f"Question types: {type_counts}")
    print(f"Source legal types: {legal_type_counts}")
    print(f"\n⚠️  IMPORTANT: Review the queries before using them for evaluation!")
    print(f"   Edit: {output}")
    print(f"   Remove any remaining template/low-quality questions.")

    return all_queries


@click.command()
@click.option("--data-path", default="data/raw/legal_docs_merged.parquet")
@click.option("--docs", "num_docs", type=int, default=20, help="Number of source documents")
@click.option("--queries-per-doc", type=int, default=2)
@click.option("--output", default="data/eval/test_queries.json")
@click.option("--seed", type=int, default=42)
@click.option("--max-doc-chars", type=int, default=12000,
              help="Max chars of doc content to pass to LLM")
def main(data_path, num_docs, queries_per_doc, output, seed, max_doc_chars):
    generate_queries(data_path, num_docs, queries_per_doc, output, seed,
                     max_doc_chars=max_doc_chars)


if __name__ == "__main__":
    main()
