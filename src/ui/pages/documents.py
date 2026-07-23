"""
Streamlit Documents Page — Browse and search legal documents.
"""

import os
import streamlit as st
import requests
import pandas as pd

from src.data.constants import ALL_LEGAL_TYPES

API_BASE = os.environ.get("API_URL", "http://localhost:8000")

LEGAL_TYPE_OPTIONS = ["Tất cả"] + ALL_LEGAL_TYPES


PAGE_SIZE = 50


def render_documents_page():
    st.title("📁 Thư viện Văn bản Pháp luật")
    st.caption("Duyệt và tìm kiếm văn bản pháp luật Việt Nam trong cơ sở dữ liệu.")

    # --- Filters ---
    col1, col2, col3 = st.columns([2, 3, 1])
    with col1:
        legal_type = st.selectbox("Loại văn bản", LEGAL_TYPE_OPTIONS)
    with col2:
        keyword = st.text_input("Từ khóa", placeholder="Nhập từ khóa tìm kiếm...")
    with col3:
        st.write("")  # Spacing
        search_btn = st.button("🔍 Tìm kiếm", use_container_width=True)

    # A search result can span more than 1 page (page_size=50) -- persist the
    # filters + current page across reruns so "Trang sau"/"Trang trước" can
    # re-issue the same search at a different page instead of only ever
    # being able to see page 1's 50 rows no matter how many results matched.
    if search_btn:
        st.session_state.doc_search = {"legal_type": legal_type, "keyword": keyword}
        st.session_state.doc_page = 1

    # --- Results Table ---
    if "doc_search" in st.session_state:
        search = st.session_state.doc_search
        page = st.session_state.get("doc_page", 1)
        with st.spinner("Đang tìm kiếm..."):
            try:
                params = {"page": page, "page_size": PAGE_SIZE}
                if search["legal_type"] != "Tất cả":
                    params["legal_type"] = search["legal_type"]
                if search["keyword"]:
                    params["keyword"] = search["keyword"]

                resp = requests.get(f"{API_BASE}/api/documents", params=params, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    total = data.get("total", 0)
                    st.success(f"Tìm thấy {total} văn bản")

                    if data.get("documents"):
                        df = pd.DataFrame(data["documents"])
                        st.dataframe(df, use_container_width=True,
                                     column_config={
                                         "title": "Tên văn bản",
                                         "legal_type": "Loại",
                                         "issuance_date": "Ngày ban hành",
                                     })

                        total_pages = max(1, -(-total // PAGE_SIZE))  # ceil div
                        col_prev, col_mid, col_next = st.columns([1, 2, 1])
                        with col_prev:
                            if st.button("⬅️ Trang trước", disabled=(page <= 1)):
                                st.session_state.doc_page = page - 1
                                st.rerun()
                        with col_mid:
                            st.caption(f"Trang {page}/{total_pages}")
                        with col_next:
                            if st.button("Trang sau ➡️", disabled=(page >= total_pages)):
                                st.session_state.doc_page = page + 1
                                st.rerun()
                else:
                    st.error(f"Lỗi API: {resp.status_code}")
            except Exception as e:
                st.error(f"Không thể kết nối đến backend: {e}")

    # --- Stats ---
    st.divider()
    st.subheader("📊 Thống kê")
    try:
        resp = requests.get(f"{API_BASE}/api/stats", timeout=5)
        if resp.status_code == 200:
            stats = resp.json()
            col1, col2 = st.columns(2)
            col1.metric("Tổng số chunks", f"{stats.get('indexed_documents', 0):,}")
            col2.metric("Collection", stats.get("collection_name", "N/A"))
    except Exception:
        st.info("Backend chưa sẵn sàng. Chạy `docker compose up` để khởi động.")
