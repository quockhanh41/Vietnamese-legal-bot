"""
Streamlit Documents Page — Browse and search legal documents.
"""

import os
import streamlit as st
import requests
import pandas as pd

API_BASE = os.environ.get("API_URL", "http://localhost:8000")

LEGAL_TYPES = [
    "Tất cả",
    "Luật",
    "Bộ luật",
    "Nghị định",
    "Thông tư",
    "Nghị quyết",
    "Quyết định",
]


def render_documents_page():
    st.title("📁 Thư viện Văn bản Pháp luật")
    st.caption("Duyệt và tìm kiếm văn bản pháp luật Việt Nam trong cơ sở dữ liệu.")

    # --- Filters ---
    col1, col2, col3 = st.columns([2, 3, 1])
    with col1:
        legal_type = st.selectbox("Loại văn bản", LEGAL_TYPES)
    with col2:
        keyword = st.text_input("Từ khóa", placeholder="Nhập từ khóa tìm kiếm...")
    with col3:
        st.write("")  # Spacing
        search_btn = st.button("🔍 Tìm kiếm", use_container_width=True)

    # --- Results Table ---
    if search_btn:
        with st.spinner("Đang tìm kiếm..."):
            try:
                params = {"page": 1, "page_size": 50}
                if legal_type != "Tất cả":
                    params["legal_type"] = legal_type
                if keyword:
                    params["keyword"] = keyword

                resp = requests.get(f"{API_BASE}/api/documents", params=params, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    st.success(f"Tìm thấy {data.get('total_documents', 0)} văn bản")

                    if data.get("documents"):
                        df = pd.DataFrame(data["documents"])
                        st.dataframe(df, use_container_width=True,
                                     column_config={
                                         "title": "Tên văn bản",
                                         "legal_type": "Loại",
                                         "issuance_date": "Ngày ban hành",
                                     })
                else:
                    st.error(f"Lỗi API: {resp.status_code}")
            except Exception as e:
                st.warning(f"Chức năng đang được phát triển. {e}")

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
