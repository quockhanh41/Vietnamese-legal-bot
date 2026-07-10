"""
M1.1 — Trích xuất corpus 100 văn bản đa lĩnh vực từ legal_docs_merged.parquet.

Danh sách được TUYỂN CHỌN TAY (curated) theo 3 nguyên tắc, chốt 2026-07-10
(docs/notes/corpus-100docs-selection-2026-07-10.md có toàn bộ lai lịch):

1. Luật gốc còn hiệu lực của 6 lĩnh vực: hình sự (giữ 6 docs của corpus cũ),
   dân sự, lao động, doanh nghiệp, đất đai, hôn nhân gia đình.
2. Cặp temporal (luật cũ → luật thay thế) chỉ đưa vào khi CẢ HAI đầu đều có
   trong dataset — để enricher đánh dấu is_in_effect đúng. Luật đã bị thay
   thế mà bản thay thế VẮNG trong dataset thì loại hẳn (Nhà ở 2014, Kinh
   doanh BĐS 2014): giữ lại sẽ làm bot trích luật hết hiệu lực như luật hiện
   hành — sai nguy hiểm hơn là không phủ.
3. Nghị định hướng dẫn chính của từng luật gốc (đăng ký DN, hướng dẫn BLLĐ,
   hướng dẫn Luật Đất đai, xử phạt từng lĩnh vực...).

Khoảng trống dataset đã biết (ghi vào UI disclaimer, plan §5):
  - Luật Đầu tư 2020 (61/2020/QH14)         — thiếu; có Đầu tư 2014 + 2025
  - Luật Nhà ở 2023 (27/2023/QH15)          — thiếu; loại luôn Nhà ở 2014
  - Luật Kinh doanh BĐS 2023 (29/2023/QH15) — thiếu; loại luôn KDBĐS 2014

Usage:
    .venv/bin/python scripts/select_corpus.py
    # → data/raw/corpus_100docs.parquet (+ in bảng tóm tắt per-sector)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import pyarrow.parquet as pq

MERGED_PATH = "data/raw/legal_docs_merged.parquet"
OUTPUT_PATH = "data/raw/corpus_100docs.parquet"

# (dataset id, sector) — sector là nhãn CHÍNH gán tay cho báo cáo per-sector
# (Gate M1); cột legal_sectors gốc của dataset đa nhãn/nhiễu nên không dùng
# trực tiếp. id là khoá duy nhất trong dataset (đuôi URL thuvienphapluat).
CORPUS: list[tuple[int, str]] = [
    # ── HÌNH SỰ (14) — 6 docs corpus cũ + thi hành án hình sự ──────────
    (46056,  "hinh_su"),   # Bộ luật Hình sự 1999 (đã thay thế — temporal)
    (296661, "hinh_su"),   # Bộ luật Hình sự 2015
    (354053, "hinh_su"),   # Luật sửa đổi BLHS 2017
    (662377, "hinh_su"),   # Luật sửa đổi BLHS 2025 (86/2025/QH15)
    (296884, "hinh_su"),   # Bộ luật Tố tụng hình sự 2015
    (489113, "hinh_su"),   # Luật sửa đổi BLTTHS 2021
    (662829, "hinh_su"),   # Luật sửa đổi BLTTHS 2025 (99/2025/QH15)
    (387991, "hinh_su"),   # Luật Thi hành án hình sự 2019 (→ 2025)
    (686325, "hinh_su"),   # Luật Thi hành án hình sự 2025 (127/2025/QH15)
    (427247, "hinh_su"),   # NĐ 133/2020 hướng dẫn THAHS (→ 118/2024)
    (626392, "hinh_su"),   # NĐ 118/2024 hướng dẫn Luật THAHS
    (298373, "hinh_su"),   # Luật thi hành tạm giữ, tạm giam 2015 (→ 2025)
    (679014, "hinh_su"),   # Luật THTGTG và cấm đi khỏi nơi cư trú 2025
    (376934, "hinh_su"),   # NĐ 30/2018 định giá tài sản trong tố tụng HS
    # ── DÂN SỰ (21) — BLDS/BLTTDS + thi hành án DS + SHTT + căn cước ───
    (2463,   "dan_su"),    # Bộ luật Dân sự 2005 (đã thay thế — temporal)
    (296215, "dan_su"),    # Bộ luật Dân sự 2015
    (296861, "dan_su"),    # Bộ luật Tố tụng dân sự 2015
    (651570, "dan_su"),    # Luật sửa đổi BLTTDS/TTHC/TPNCTN 2025 (85/2025)
    (298372, "dan_su"),    # Luật Tố tụng hành chính 2015
    (468069, "dan_su"),    # NĐ 21/2021 hướng dẫn BLDS về bảo đảm nghĩa vụ
    (542323, "dan_su"),    # NĐ 99/2022 đăng ký biện pháp bảo đảm
    (82197,  "dan_su"),    # Luật Thi hành án dân sự 2008 (→ 2025)
    (259728, "dan_su"),    # Luật THADS sửa đổi 2014
    (628717, "dan_su"),    # Luật Thi hành án dân sự 2025 (106/2025/QH15)
    (284071, "dan_su"),    # NĐ 62/2015 hướng dẫn Luật THADS
    (108083, "dan_su"),    # Luật Trọng tài thương mại 2010
    (395767, "dan_su"),    # Luật Hòa giải, đối thoại tại Tòa án 2020
    (7022,   "dan_su"),    # Luật Sở hữu trí tuệ 2005
    (458435, "dan_su"),    # Luật SHTT sửa đổi 2022
    (313517, "dan_su"),    # Luật Trách nhiệm bồi thường của Nhà nước 2017
    (625628, "dan_su"),    # Luật Bảo vệ dữ liệu cá nhân 2025 (91/2025/QH15)
    (259727, "dan_su"),    # Luật Hộ tịch 2014
    (259782, "dan_su"),    # Luật Căn cước công dân 2014 (→ 2023)
    (552422, "dan_su"),    # Luật Căn cước 2023
    (12981,  "dan_su"),    # Luật Luật sư 2006
    # ── LAO ĐỘNG (19) — BLLĐ + BHXH/BHYT/việc làm/công đoàn ────────────
    (142187, "lao_dong"),  # Bộ luật Lao động 2012 (đã thay thế — temporal)
    (333670, "lao_dong"),  # Bộ luật Lao động 2019
    (459400, "lao_dong"),  # NĐ 145/2020 hướng dẫn BLLĐ về điều kiện LĐ
    (259700, "lao_dong"),  # Luật BHXH 2014 (→ 2024)
    (557190, "lao_dong"),  # Luật BHXH 2024 (41/2024/QH15)
    (634792, "lao_dong"),  # NĐ 158/2025 hướng dẫn Luật BHXH về BHXH bắt buộc
    (281961, "lao_dong"),  # Luật An toàn, vệ sinh lao động 2015
    (215628, "lao_dong"),  # Luật Việc làm 2013 (→ 2025)
    (530912, "lao_dong"),  # Luật Việc làm 2025 (74/2025/QH15)
    (268405, "lao_dong"),  # NĐ 28/2015 hướng dẫn Luật Việc làm về BHTN
    (142186, "lao_dong"),  # Luật Công đoàn 2012 (→ 2024)
    (445264, "lao_dong"),  # Luật Công đoàn 2024 (50/2024/QH15)
    (15865,  "lao_dong"),  # Luật NLĐ VN đi làm việc ở nước ngoài 2006 (→ 2020)
    (439844, "lao_dong"),  # Luật NLĐ VN đi làm việc ở nước ngoài 2020
    (280261, "lao_dong"),  # NĐ 152/2020 về NLĐ nước ngoài làm việc tại VN
    (603278, "lao_dong"),  # NĐ 74/2024 mức lương tối thiểu
    (479312, "lao_dong"),  # NĐ 12/2022 xử phạt VPHC lĩnh vực lao động
    (82196,  "lao_dong"),  # Luật Bảo hiểm y tế 2008
    (505750, "lao_dong"),  # Luật BHYT sửa đổi 2024 (51/2024/QH15)
    # ── DOANH NGHIỆP (25) — LDN/đầu tư/thương mại/phá sản/chứng khoán ──
    (259730, "doanh_nghiep"),  # Luật Doanh nghiệp 2014 (→ 2020)
    (427301, "doanh_nghiep"),  # Luật Doanh nghiệp 2020 (59/2020/QH14)
    (659899, "doanh_nghiep"),  # Luật Doanh nghiệp sửa đổi 2025 (76/2025)
    (470561, "doanh_nghiep"),  # NĐ 47/2021 hướng dẫn Luật Doanh nghiệp
    (283247, "doanh_nghiep"),  # NĐ 01/2021 về đăng ký doanh nghiệp
    (259729, "doanh_nghiep"),  # Luật Đầu tư 2014 (bản 2020 THIẾU trong dataset)
    (681550, "doanh_nghiep"),  # Luật Đầu tư 2025 (143/2025/QH15)
    (462291, "doanh_nghiep"),  # NĐ 31/2021 hướng dẫn Luật Đầu tư
    (2633,   "doanh_nghiep"),  # Luật Thương mại 2005
    (345182, "doanh_nghiep"),  # Luật Cạnh tranh 2018
    (238641, "doanh_nghiep"),  # Luật Phá sản 2014 (→ 2025)
    (675210, "doanh_nghiep"),  # Luật Phục hồi, phá sản 2025 (142/2025/QH15)
    (399763, "doanh_nghiep"),  # Luật Chứng khoán 2019
    (461323, "doanh_nghiep"),  # NĐ 155/2020 hướng dẫn Luật Chứng khoán
    (320905, "doanh_nghiep"),  # Luật Hỗ trợ DN nhỏ và vừa 2017
    (500102, "doanh_nghiep"),  # Luật Bảo vệ quyền lợi người tiêu dùng 2023
    (518805, "doanh_nghiep"),  # Luật Đấu thầu 2023 (22/2023/QH15)
    (662035, "doanh_nghiep"),  # Luật Thương mại điện tử 2025 (122/2025/QH15)
    (322219, "doanh_nghiep"),  # Luật Quản lý ngoại thương 2017
    (152716, "doanh_nghiep"),  # Luật Hợp tác xã 2012 (→ 2023)
    (499239, "doanh_nghiep"),  # Luật Hợp tác xã 2023 (17/2023/QH15)
    (280115, "doanh_nghiep"),  # Luật Đấu giá tài sản 2016
    (588810, "doanh_nghiep"),  # Luật Đấu giá tài sản sửa đổi 2024
    (406627, "doanh_nghiep"),  # NĐ 98/2020 xử phạt VPHC thương mại
    (285024, "doanh_nghiep"),  # NĐ 122/2021 xử phạt VPHC kế hoạch-đầu tư
    # ── ĐẤT ĐAI (11) — LĐĐ 2024 + hệ nghị định hướng dẫn 2024 ──────────
    (215836, "dat_dai"),   # Luật Đất đai 2013 (đã thay thế — temporal)
    (523642, "dat_dai"),   # Luật Đất đai 2024 (31/2024/QH15)
    (612195, "dat_dai"),   # Luật sđ Đất đai/Nhà ở/KDBĐS/Tổ chức TD 2024 (43/2024)
    (603982, "dat_dai"),   # NĐ 102/2024 hướng dẫn Luật Đất đai
    (599145, "dat_dai"),   # NĐ 71/2024 quy định về giá đất
    (600715, "dat_dai"),   # NĐ 88/2024 bồi thường, hỗ trợ, tái định cư
    (613131, "dat_dai"),   # NĐ 101/2024 điều tra cơ bản; đăng ký, cấp GCN
    (550020, "dat_dai"),   # NĐ 103/2024 tiền sử dụng đất, tiền thuê đất
    (664033, "dat_dai"),   # NĐ 226/2025 sửa đổi các NĐ hướng dẫn LĐĐ
    (626370, "dat_dai"),   # NĐ 123/2024 xử phạt VPHC lĩnh vực đất đai
    (108065, "dat_dai"),   # Luật thuế sử dụng đất phi nông nghiệp 2010
    # ── HÔN NHÂN GIA ĐÌNH (10) — HNGĐ + BLGĐ + trẻ em ───────────────────
    (46450,  "hngd"),      # Luật HNGĐ 2000 (đã thay thế — temporal)
    (238640, "hngd"),      # Luật Hôn nhân và gia đình 2014 (52/2014/QH13)
    (262379, "hngd"),      # NĐ 126/2014 hướng dẫn thi hành Luật HNGĐ
    (490095, "hngd"),      # Luật Phòng, chống bạo lực gia đình 2022
    (557470, "hngd"),      # NĐ 76/2023 hướng dẫn Luật PCBLGĐ
    (303313, "hngd"),      # Luật Trẻ em 2016 (102/2016/QH13)
    (108082, "hngd"),      # Luật Nuôi con nuôi 2010
    (15866,  "hngd"),      # Luật Bình đẳng giới 2006
    (392611, "hngd"),      # NĐ 82/2020 xử phạt VPHC hộ tịch/HNGĐ/bổ trợ tư pháp
    (425471, "hngd"),      # NĐ 144/2021 xử phạt VPHC ANTT (có mảng BLGĐ)
]


def extract_corpus(merged_path: str = MERGED_PATH, output_path: str = OUTPUT_PATH) -> pd.DataFrame:
    wanted = {doc_id: sector for doc_id, sector in CORPUS}
    assert len(wanted) == len(CORPUS), "ID trùng lặp trong danh sách CORPUS"

    pf = pq.ParquetFile(merged_path)
    parts = []
    for i in range(pf.metadata.num_row_groups):
        t = pf.read_row_group(i).to_pandas()
        hit = t[t["id"].isin(wanted)]
        if len(hit):
            parts.append(hit)
    df = pd.concat(parts, ignore_index=True).drop_duplicates(subset="id")

    missing = set(wanted) - set(df["id"])
    if missing:
        raise SystemExit(f"KHÔNG TÌM THẤY {len(missing)} id trong dataset: {sorted(missing)}")

    df["sector"] = df["id"].map(wanted)
    df["date"] = pd.to_datetime(df["issuance_date"], format="%d/%m/%Y", errors="coerce")
    df = df.sort_values(["sector", "date"]).reset_index(drop=True).drop(columns=["date"])
    df.to_parquet(output_path)
    return df


def print_summary(df: pd.DataFrame) -> None:
    lens = df["content"].str.len()
    print(f"Tổng: {len(df)} văn bản | content: {lens.sum() / 1e6:.1f}M ký tự "
          f"(min {lens.min():,}, median {lens.median():,.0f}, max {lens.max():,})")
    print(f"{'sector':<15}{'docs':>6}{'Luật':>7}{'NĐ':>5}{'M ký tự':>10}")
    for sector, g in df.groupby("sector"):
        n_law = (g["legal_type"] == "Luật").sum()
        n_decree = (g["legal_type"] == "Nghị định").sum()
        print(f"{sector:<15}{len(g):>6}{n_law:>7}{n_decree:>5}"
              f"{g['content'].str.len().sum() / 1e6:>10.1f}")
    short = df[df["content"].str.len() < 2000]
    if len(short):
        print("\n⚠️ Văn bản content < 2000 ký tự (kiểm tra tay):")
        for _, r in short.iterrows():
            print(f"  {r['id']} | {len(r['content']):,} ký tự | {r['title'][:70]}")


if __name__ == "__main__":
    frame = extract_corpus()
    print_summary(frame)
    print(f"\nĐã ghi {OUTPUT_PATH}")
