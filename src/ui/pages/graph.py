"""
Streamlit Graph & Effective-Law Page — laws in effect on a given date,
per-article amendment history, and cross-reference graph stats.
"""

from datetime import date

import os
import streamlit as st
import requests

API_BASE = os.environ.get("API_URL", "http://localhost:8000")


def render_graph_page():
    st.title("🕸️ Đồ thị & Hiệu lực")
    st.caption("Tra cứu luật còn hiệu lực theo ngày, lịch sử sửa đổi điều luật, "
               "và thống kê đồ thị trích dẫn chéo.")

    _render_effective_laws()
    st.divider()
    _render_article_history()
    st.divider()
    _render_graph_stats()


def _render_effective_laws():
    st.subheader("📅 Luật có hiệu lực tại 1 ngày")
    at_date = st.date_input("Chọn ngày", value=date.today())
    if st.button("🔍 Tra cứu hiệu lực"):
        try:
            resp = requests.get(f"{API_BASE}/api/timeline/effective",
                                 params={"at_date": at_date.isoformat()}, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                st.success(f"{data.get('count', 0)} luật có hiệu lực tại {data.get('at_date')}")
                if data.get("laws"):
                    st.dataframe(data["laws"], use_container_width=True,
                                 column_config={"law_name": "Tên luật",
                                                "effective_date": "Ngày hiệu lực"})
            else:
                st.error(f"Lỗi API: {resp.status_code}")
        except Exception as e:
            st.error(f"Không thể kết nối đến backend: {e}")


def _render_article_history():
    st.subheader("📜 Lịch sử sửa đổi 1 điều luật")
    keyword = st.text_input("Tìm văn bản theo từ khoá", key="graph_doc_keyword")
    documents = []
    if keyword:
        try:
            resp = requests.get(f"{API_BASE}/api/documents",
                                 params={"keyword": keyword, "page_size": 20}, timeout=10)
            if resp.status_code == 200:
                documents = resp.json().get("documents", [])
            else:
                st.error(f"Lỗi API: {resp.status_code}")
        except Exception as e:
            st.error(f"Không thể kết nối đến backend: {e}")

    if documents:
        selected = st.selectbox("Chọn văn bản", options=documents,
                                 format_func=lambda d: d["title"])
        article = st.text_input("Số điều", placeholder="Điều 5")
        if st.button("🔍 Tra cứu lịch sử") and article:
            try:
                resp = requests.get(f"{API_BASE}/api/article/history",
                                     params={"doc_id": selected["document_id"],
                                             "article": article}, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("error"):
                        st.warning(data["error"])
                    else:
                        st.json(data)
                else:
                    st.error(f"Lỗi API: {resp.status_code}")
            except Exception as e:
                st.error(f"Không thể kết nối đến backend: {e}")
    elif keyword:
        st.info("Không tìm thấy văn bản khớp từ khoá.")


def _render_graph_stats():
    st.subheader("📊 Thống kê đồ thị trích dẫn chéo")
    try:
        resp = requests.get(f"{API_BASE}/api/graph/stats", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            col1, col2 = st.columns(2)
            col1.metric("Tổng số node", data.get("nodes", 0))
            col1.write(data.get("node_types", {}))
            col2.metric("Tổng số edge", data.get("edges", 0))
            col2.write(data.get("edge_types", {}))
        else:
            st.error(f"Lỗi API: {resp.status_code}")
    except Exception as e:
        st.error(f"Không thể kết nối đến backend: {e}")
