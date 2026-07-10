# Lai lịch chọn corpus 100 văn bản (M1.1) — 2026-07-10

Script tái lập: `scripts/select_corpus.py` → `data/raw/corpus_100docs.parquet`.
Nguồn: `data/raw/legal_docs_merged.parquet` (HF `vohuutridung/vietnamese-legal-documents`,
518.255 văn bản, phủ đến 2026-03).

## Kết quả

| Sector | Docs | Luật | Nghị định | M ký tự |
|---|---|---|---|---|
| dan_su | 21 | 18 | 3 | 3.3 |
| dat_dai | 11 | 4 | 7 | 2.2 |
| doanh_nghiep | 25 | 19 | 6 | 4.1 |
| hinh_su | 14 | 11 | 3 | 2.8 |
| hngd | 10 | 6 | 4 | 1.0 |
| lao_dong | 19 | 13 | 6 | 1.9 |
| **Tổng** | **100** | **71** | **29** | **15.3** |

Content: min 3.596 (Luật sđ BLTTHS 2021 — hợp lệ, luật sửa đổi ngắn),
median 112K, max 574K ký tự. Ước tính chunk: corpus 6 docs 1.72M ký tự →
2.242 chunks ⇒ 15.3M ký tự ≈ **~20K chunks** (worst-case plan §2.2 là 167K
cho 600 docs — dư địa lớn).

## 3 nguyên tắc chọn

1. **Luật gốc còn hiệu lực** của 6 lĩnh vực; giữ nguyên 6 docs hình sự của
   corpus cũ (baseline so sánh M0→M1).
2. **Cặp temporal chỉ khi cả hai đầu đều có trong dataset** — luật đã bị
   thay thế mà bản thay thế vắng mặt thì LOẠI HẲN, vì enricher sẽ không thấy
   quan hệ thay thế → `is_in_effect` giữ True → bot trích luật chết như luật
   sống. Vì lý do này **Luật Nhà ở 2014 và Luật Kinh doanh BĐS 2014 bị loại**
   (bản 2023 thiếu trong dataset).
3. **Nghị định hướng dẫn chính** của từng luật gốc + nghị định xử phạt VPHC
   của lĩnh vực đó (câu hỏi "phạt bao nhiêu" rất phổ biến).

## Cặp temporal có chủ đích (test is_in_effect/amendment ở quy mô mới)

- BLHS 1999 → BLHS 2015 (+ sđ 2017, sđ 2025)
- BLDS 2005 → BLDS 2015; Luật THADS 2008 (+sđ 2014) → THADS 2025
- Luật THAHS 2019 → THAHS 2025; NĐ 133/2020 → NĐ 118/2024 (cặp nghị định)
- Luật thi hành tạm giữ, tạm giam 2015 → 2025
- BLLĐ 2012 → BLLĐ 2019; BHXH 2014 → BHXH 2024; Việc làm 2013 → 2025;
  Công đoàn 2012 → 2024; NLĐ đi làm việc nước ngoài 2006 → 2020
- LDN 2014 → LDN 2020 (+sđ 2025); Phá sản 2014 → Phục hồi, phá sản 2025;
  HTX 2012 → 2023; Căn cước công dân 2014 → Căn cước 2023
- LĐĐ 2013 → LĐĐ 2024; HNGĐ 2000 → HNGĐ 2014; SHTT 2005 + sđ 2022

## Khoảng trống dataset đã biết → UI disclaimer (plan §5 risk item)

| Văn bản thiếu | Hệ quả xử lý |
|---|---|
| Luật Đầu tư 2020 (61/2020/QH14) | Có Đầu tư 2014 + Đầu tư 2025; NĐ 31/2021 (hướng dẫn bản 2020) vẫn giữ vì nội dung chi tiết hữu ích — enrichment sẽ không link được parent |
| Luật Nhà ở 2023 (27/2023/QH15) | Loại cả Nhà ở 2014 khỏi corpus — lĩnh vực đất đai KHÔNG phủ nhà ở |
| Luật KDBĐS 2023 (29/2023/QH15) | Loại cả KDBĐS 2014 — không phủ kinh doanh BĐS |

Disclaimer cần thêm khi làm UI (M5): "Dữ liệu phủ đến 2026-03; một số văn
bản có thể vắng mặt — bot không phủ Luật Nhà ở, Luật Kinh doanh BĐS."

## Ngoài phạm vi có chủ đích

**Luật Xử lý VPHC 2012 cố tình KHÔNG đưa vào** — nó là mỏ neo của định nghĩa
out-of-scope mới ở M1.6 (hành chính/quốc tế/tư vấn cá nhân). Các NĐ xử phạt
VPHC theo lĩnh vực (12/2022, 98/2020, 123/2024...) vẫn trong scope vì gắn
trực tiếp lĩnh vực đã phủ.
