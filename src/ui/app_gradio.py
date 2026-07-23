"""HF Space (Gradio + ZeroGPU) front-end -- M5 pivot. Additive: calls the
existing RAG pipeline in-process, no HTTP hop, no separate FastAPI/uvicorn
process. src/api/main.py and src/ui/app.py (Streamlit) are untouched and
remain a self-hosted Docker deploy path -- this file is the live public demo.
See docs/superpowers/specs/2026-07-23-m5-gradio-zerogpu-pivot-design.md."""
import spaces  # MUST be first -- before any torch-touching import

import json

from dotenv import load_dotenv
load_dotenv()

import gradio as gr
import pandas as pd
from FlagEmbedding import FlagReranker

from src.data.constants import ALL_LEGAL_TYPES, DEFAULT_COLLECTION_NAME
from src.data.metadata_store import MetadataStore
from src.data.mixed_embedder import build_embedder
from src.db.qdrant_client import LegalVectorStore
from src.llm.gateway import LLMGateway
from src.pipeline.retriever import EnhancedRetriever
from src.rag.citation_lookup import CitationLookup
from src.rag.condense import condense_query
from src.rag.degraded import DEGRADED_MSG, is_exhaustion
from src.rag.orchestrator import AdaptiveRAGPipeline, RAGResponse
from src.rag.semantic_cache import SemanticCache
from src.ui.gradio_helpers import (
    document_choices,
    format_citations_markdown,
    format_effective_laws,
    format_system_info,
    graph_type_counts,
    history_to_turns,
    is_quota_exceeded,
    parse_effective_date,
    rename_document_columns,
)

QUOTA_MSG = ("Bạn đã dùng hết lượt dùng thử miễn phí hôm nay trên hạ tầng "
             "ZeroGPU. Vui lòng quay lại vào ngày mai!")

# --- Module-scope singletons: loaded ONCE, device="cuda" eagerly. ---
# `import spaces` above monkey-patches torch.cuda so this works with no
# physical GPU attached at import time -- weights get "packed" to disk and
# streamed into real VRAM on the first @spaces.GPU call below. This is HF's
# documented ZeroGPU pattern (references/zerogpu.md): "Load models at module
# scope... Do NOT lazy-load inside @spaces.GPU -- deferring it puts tens of
# seconds of checkpoint I/O + dtype cast + GPU move inside every cold
# request." A prior revision of this file moved loading into a lazy
# get_pipeline() called from inside handle_turn; that violated the rule
# above (cold requests paid full model-load time out of the GPU duration
# budget) and raced on its module-level globals under ZeroGPU's
# concurrent-by-default handlers ("No mutable global state"). Reverted.
vector_store = LegalVectorStore(collection_name=DEFAULT_COLLECTION_NAME)
embedder = build_embedder("mixed", "AITeamVN/Vietnamese_Embedding", device="cuda")
reranker = FlagReranker("AITeamVN/Vietnamese_Reranker", use_fp16=True, device="cuda")
llm = LLMGateway()
metadata_store = MetadataStore()
citation_lookup = CitationLookup(vector_store)
retriever = EnhancedRetriever(vector_store=vector_store, metadata_store=metadata_store,
                               embedder=embedder)


def _embed_query(text: str):
    out = embedder.encode([text], return_dense=True, return_sparse=False)
    return out["dense_vecs"][0].tolist()


semantic_cache = SemanticCache.from_env(_embed_query)

pipeline = AdaptiveRAGPipeline(
    vector_store=vector_store,
    embedder=embedder,
    llm_gateway=llm,
    metadata_store=metadata_store,
    reranker_model="AITeamVN/Vietnamese_Reranker",
    reranker=reranker,
    citation_lookup=citation_lookup,
)


# duration is a placeholder pending re-measurement (the account's daily
# ZeroGPU quota is exhausted as of 2026-07-23 -- see Task 9 in
# docs/superpowers/plans/2026-07-23-m5-gradio-zerogpu-deploy.md). Do NOT
# read the previous duration=90 as a real measurement: the live Space
# rejected a request with "135s requested vs. 98s left", which doesn't
# match 90 and was never reconciled. Re-measure via gradio_client once
# quota resets and set duration = round(measured_max * 1.4) for real.
@spaces.GPU(duration=180)
def handle_turn(message, history):
    """One @spaces.GPU allocation per turn -- covers condense, the semantic-
    cache lookup (needs an embedding), and on a miss: retrieve/rerank/
    generate(Groq, network)/NLI verify. Note: is_quota_exceeded() below can
    only catch a quota failure raised *during* this function's execution.
    ZeroGPU's "quota exceeded" pre-flight check (requested vs remaining)
    happens at the @spaces.GPU call boundary, before this function body
    starts -- that case surfaces as Gradio's own default error UI, not
    QUOTA_MSG, and there is no way to intercept it from inside the handler."""
    turns = history_to_turns(history)
    condensed = condense_query(turns, message, llm)

    cached = semantic_cache.get(condensed, DEFAULT_COLLECTION_NAME) if semantic_cache else None
    if cached is not None:
        yield cached.answer + format_citations_markdown(cached.citations)
        return

    full_answer = ""
    citations = []
    refused = False
    try:
        for token in pipeline.query_stream(condensed):
            if token.startswith("__CITATIONS__"):
                try:
                    citations = json.loads(token[len("__CITATIONS__"):])
                except json.JSONDecodeError:
                    pass
                continue
            if token.startswith("__VERIFICATION__"):
                try:
                    refused = json.loads(token[len("__VERIFICATION__"):]).get("refused", False)
                except json.JSONDecodeError:
                    pass
                continue
            if token.startswith("__STEP__"):
                continue
            full_answer += token
            yield full_answer
    except Exception as e:
        if is_exhaustion(e):
            yield DEGRADED_MSG
            return
        if is_quota_exceeded(e):
            yield QUOTA_MSG
            return
        raise

    yield full_answer + format_citations_markdown(citations)

    if semantic_cache is not None and full_answer and not refused:
        response = RAGResponse(
            answer=full_answer, citations=citations, confidence="unverified",
            disclaimer="", latency_ms=0, pipeline_used="gradio", refused=refused,
        )
        semantic_cache.set(condensed, DEFAULT_COLLECTION_NAME, response)


chat_interface = gr.ChatInterface(
    fn=handle_turn,
    type="messages",
    title="Vietnamese Legal Knowledge Bot",
    description=("Adaptive RAG tra cứu văn bản pháp luật Việt Nam (600 văn bản). "
                 "Chạy trên HF ZeroGPU (free tier) -- lần hỏi đầu có thể chậm "
                 "(cold start nạp model)."),
)


def search_documents(legal_type: str, keyword: str):
    """Documents tab search -- pure CPU/Qdrant call, no @spaces.GPU needed."""
    lt = None if legal_type == "Tất cả" else legal_type
    result = vector_store.list_documents(legal_type=lt, keyword=keyword or None,
                                          page=1, page_size=50)
    df = rename_document_columns(pd.DataFrame(result["documents"]))
    return f"Tìm thấy {result['total']} văn bản", df


def system_info_display():
    """Called eagerly at Blocks-construction time (see call site below) --
    unlike the .click()-wired functions in this file, an exception here isn't
    caught by Gradio's per-event error handling, it fails the whole module
    import. get_collection_info() is a real Qdrant network round-trip, so a
    transient outage here must not take down the entire Space (Chat included)
    over an optional info panel."""
    try:
        info = vector_store.get_collection_info()
        return format_system_info(LLMGateway.MODEL_CHAIN[0], info["name"], info["points_count"])
    except Exception as e:
        return f"Không lấy được thông tin hệ thống: {e}"


def refresh_document_choices(keyword: str):
    result = vector_store.list_documents(keyword=keyword or None, page=1, page_size=50)
    return gr.update(choices=document_choices(result["documents"]))


def effective_laws_display(at_date_str: str):
    try:
        at_date = parse_effective_date(at_date_str)
    except ValueError:
        return f"Định dạng ngày không hợp lệ: '{at_date_str}' (cần YYYY-MM-DD)"
    laws = metadata_store.get_effective_laws(at_date)
    return format_effective_laws(laws)


def article_history_display(doc_id, article: str):
    if not doc_id or not article:
        return {"error": "Vui lòng chọn văn bản và nhập số điều."}
    return retriever.get_article_history(int(doc_id), article)


def graph_stats_display():
    """Same eager-at-import call-site risk as system_info_display() above --
    guarded the same way for consistency, even though get_graph() itself
    currently tolerates an empty/missing graph row without raising."""
    try:
        graph = metadata_store.get_graph()
        counts = graph_type_counts(graph.get("nodes", []), graph.get("edges", []))
    except Exception as e:
        return f"Không lấy được thống kê đồ thị: {e}"
    lines = [f"**Tổng số node:** {counts['nodes']} · **Tổng số edge:** {counts['edges']}", "",
             "**Theo loại node:**"]
    lines += [f"- {k}: {v}" for k, v in counts["node_types"].items()]
    lines += ["", "**Theo loại quan hệ (edge):**"]
    lines += [f"- {k}: {v}" for k, v in counts["edge_types"].items()]
    return "\n".join(lines)


with gr.Blocks(title="Vietnamese Legal Knowledge Bot") as demo:
    with gr.Tabs():
        with gr.Tab("💬 Chat"):
            chat_interface.render()

        with gr.Tab("📁 Documents"):
            with gr.Row():
                doc_legal_type = gr.Dropdown(["Tất cả"] + ALL_LEGAL_TYPES, value="Tất cả",
                                              label="Loại văn bản")
                doc_keyword = gr.Textbox(label="Từ khoá", placeholder="Nhập từ khoá tìm kiếm...")
                doc_search_btn = gr.Button("🔍 Tìm kiếm")
            doc_result_label = gr.Markdown()
            doc_result_table = gr.Dataframe()
            doc_search_btn.click(fn=search_documents, inputs=[doc_legal_type, doc_keyword],
                                  outputs=[doc_result_label, doc_result_table])

        with gr.Tab("📊 Thống kê"):
            gr.Markdown(system_info_display())

        with gr.Tab("🕸️ Đồ thị & Hiệu lực"):
            gr.Markdown("### 📅 Luật có hiệu lực tại 1 ngày")
            with gr.Row():
                eff_date_box = gr.Textbox(label="Ngày (YYYY-MM-DD)", placeholder="2026-07-23")
                eff_btn = gr.Button("🔍 Tra cứu hiệu lực")
            eff_table = gr.Dataframe()
            eff_btn.click(fn=effective_laws_display, inputs=[eff_date_box], outputs=[eff_table])

            gr.Markdown("### 📜 Lịch sử sửa đổi 1 điều luật")
            gr.Markdown("Ví dụ: gõ **Luật đất đai** để tìm văn bản, chọn *Luật đất đai 2013*, "
                        "nhập **Điều 5** rồi bấm Tra cứu lịch sử.")
            with gr.Row():
                hist_keyword = gr.Textbox(label="Tìm văn bản theo từ khoá",
                                           placeholder="VD: Luật đất đai")
                hist_search_btn = gr.Button("Tìm văn bản")
            hist_doc_dropdown = gr.Dropdown(label="Chọn văn bản", choices=[])
            hist_article_box = gr.Textbox(label="Số điều", placeholder="Điều 5")
            hist_btn = gr.Button("🔍 Tra cứu lịch sử")
            hist_result = gr.JSON()
            hist_search_btn.click(fn=refresh_document_choices, inputs=[hist_keyword],
                                   outputs=[hist_doc_dropdown])
            hist_btn.click(fn=article_history_display,
                           inputs=[hist_doc_dropdown, hist_article_box], outputs=[hist_result])

            gr.Markdown("### 📊 Thống kê đồ thị trích dẫn chéo")
            gr.Markdown(graph_stats_display())


if __name__ == "__main__":
    demo.launch()
