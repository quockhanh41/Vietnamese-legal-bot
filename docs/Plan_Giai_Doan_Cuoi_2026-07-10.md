# 🏁 Plan Giai Đoạn Cuối — Portfolio Production-Grade, SOTA, $0

> **Ngày lập**: 2026-07-10 · **Trạng thái**: bản sống — cập nhật sau mỗi milestone
> **Tiền đề**: đọc kèm [`Danh_Gia_va_Roadmap_Production_2026-07-06.md`](Danh_Gia_va_Roadmap_Production_2026-07-06.md)
> (số đo baseline + checklist scale §6) và [`ARCHITECTURE.md`](ARCHITECTURE.md) (hiện trạng code).

---

## 0. Bốn quyết định khung (đã chốt với user 2026-07-10)

| Quyết định | Lựa chọn | Hệ quả lên plan |
|---|---|---|
| **Định vị** | Portfolio production-grade | Mục tiêu là demo public + số đo eval + tài liệu kiến trúc thuyết phục được nhà tuyển dụng/khách hàng VN. KHÔNG làm multi-tenant, billing, SLA thật. |
| **Corpus đích** | ~600 docs đa lĩnh vực | Co giãn theo trần free tier — nhưng phép tính dưới (§2) cho thấy 600 docs vừa thoải mái, khả năng cao KHÔNG phải cắt. |
| **Ngân sách** | $0 — free tier toàn bộ | Qdrant Cloud free + HF Spaces free + Groq free (key rotation sẵn) + Colab/Kaggle GPU + Langfuse Cloud Hobby. |
| **Timeline** | Không gấp, chất lượng trước | Plan theo **milestone có gate đo lường**, không theo tuần. Được phép làm các hạng mục SOTA sâu (fine-tune embedding, benchmark ngoài, corrective RAG). |

**Nguyên tắc xuyên suốt (kế thừa, không thương lượng):**
1. *Thà từ chối còn hơn trả lời sai* — mọi thay đổi phải giữ 2 cổng từ chối nguyên vẹn.
2. *Đo trước, tối ưu sau* — mỗi kỹ thuật mới chỉ merge khi có số trước/sau trên eval set. Kỹ thuật không cải thiện số đo → bỏ, ghi lại lý do (bản thân việc đó cũng là nội dung portfolio).
3. *Scale theo bậc thang* (§6.0 roadmap): 6 → ~100 → ~600. Sau mỗi bậc chạy Tier-1 + validate eval set; recall tụt >5 điểm → dừng, tìm nguyên nhân.
4. *Threshold là hàm của phân bố* — mỗi khi context/corpus/signal đổi, re-run Tier-2 + `calibrate_threshold.py`. Không bao giờ tin 0.725 cũ sau thay đổi lớn.
5. Workflow mặc định: Sonnet sub-agent implement → Fable verify độc lập → commit.

---

## 1. Vị trí hiện tại (tóm tắt 1 đoạn)

Lõi Adaptive RAG đã hoàn chỉnh và **đã đo được**: routing 3 cấp + citation-lookup route (recall@5 1.0 trên citation query), contextual embedding (recall@5 tổng 0.92 routed), grounded citations `[n]` + verifier 3 tầng, scope gate + confidence gate (0.725, calibrate leak-safe), parent-document retrieval, temporal filter + amendment badge, 116 tests, CI Tier-1 nightly, README/LICENSE/.env.example. **Chưa có**: corpus thật (đang 6 docs hình sự), multi-turn memory, observability, security, deployment public, benchmark ngoài. **Nợ kỹ thuật ưu tiên cao**: threshold 0.725 chưa recalibrate sau parent-doc (context dài hơn → `context_overlap` 15% + `llm_self_eval` 30% đổi phân bố).

---

## 2. Bản đồ hạ tầng $0 + phép tính khả thi

### 2.1. Stack free tier

| Lớp | Dịch vụ (free) | Giới hạn | Đủ không? |
|---|---|---|---|
| Vector DB | **Qdrant Cloud free** | 1GB RAM / 4GB disk / 1 node | ✅ Xem 2.2 — 600 docs worst-case dùng ~26% RAM |
| Serving (API+UI+3 model) | **HF Spaces free (Docker Space)** | 2 vCPU / 16GB RAM, sleep sau 48h idle | ✅ RAM (BGE-M3 ~2GB + reranker ~2GB + NLI ~1GB + app); ⚠️ CPU chậm → phải đo P95 reranker (M5 gate) |
| LLM | **Groq free** + key rotation (sẵn có) | TPD/TPM theo key | ✅ đã sống sót full Tier-2 eval 70 câu |
| GPU offline (embed/fine-tune) | **Kaggle** (30h GPU/tuần) + Colab free | phiên ~9-12h | ✅ embed 600 docs ước tính vài giờ T4; fine-tune bi-encoder vừa vặn |
| Session/cache | **Upstash Redis free** (10K lệnh/ngày) hoặc Redis in-container trên Space | mất persistence khi Space restart | ✅ đủ cho demo; chọn Upstash nếu muốn giữ session qua restart |
| Observability | **Langfuse Cloud Hobby** (free, 50K obs/tháng) | không self-host được với $0 | ✅ demo cá nhân dùng ~vài trăm query/tháng |
| CI | GitHub Actions free | 2000 phút/tháng | ✅ Tier-1 không LLM, nhẹ |

### 2.2. Phép tính dung lượng (đã chạy 2026-07-10)

Baseline 6 docs = 1.670 chunks (~278 chunks/doc — đây là **worst case** vì 6 docs toàn bộ luật lớn; Luật/Nghị định/Thông tư trung bình nhỏ hơn nhiều):

| Kịch bản | Chunks (600 docs) | RAM Qdrant (int8 + HNSW + sparse) | Disk (fp32 on-disk + payload) |
|---|---|---|---|
| Worst-case (toàn luật lớn) | ~167K | ~264MB / 1024MB | ~1.2GB / 4GB |
| Trung bình thực tế | ~72K | ~114MB | ~0.5GB |

**Kết luận**: 600 docs khả thi trên free tier với điều kiện bật **scalar quantization int8 + original vectors on-disk ngay từ khi tạo collection mới** (đổi sau = re-index, §6.1 #6 roadmap). Gate xác nhận bằng số thật ở M2 (đo bytes/chunk thực tế tại bậc 100 docs → ngoại suy).

---

## 3. Milestones

> Mỗi milestone có: **việc làm** (tham chiếu roadmap §), **gate** (điều kiện đo lường để coi là xong — không đạt gate thì không mở milestone sau), và **artefact** (thứ để lại cho portfolio).
> Thứ tự M0→M2 là cứng (phụ thuộc dữ liệu). M3↔M4 có thể hoán đổi/xen kẽ.

### M0 — Nền sạch: trả nợ tín hiệu + recalibrate + blocker ingest

*Mục đích: mọi số đo từ M1 trở đi phải đứng trên nền tin được. Làm trên corpus 6 docs hiện tại.*

> **Đã đối chiếu code 2026-07-10** — mọi giả định của plan xác nhận đúng với HEAD (`f0a9299`), anchor cụ thể:
> `context[:2000]` cắt ở **2 chỗ**: NLI premise `src/rag/verifier.py:458` VÀ self-eval `src/rag/confidence.py:262` (0.1a và 0.1b chung gốc — sửa cùng nhau);
> grounding chưa có trong `DEFAULT_WEIGHTS` (`confidence.py:36-42`, đúng 5 signal);
> point ID `uuid.uuid4()` tại `src/db/qdrant_client.py:116`; chưa có payload index/quantization/on-disk (grep 0 kết quả);
> lookup scroll toàn collection client-side (`src/rag/citation_lookup.py:177-185`, TODO(scale) ghi sẵn trong code);
> `session_id` chỉ được khai báo (`src/api/main.py:34`), không dùng; Redis 0 tham chiếu trong `src/`;
> CORS `allow_origins=["*"]` (`main.py:93`); `/api/pipeline/enrich` (`main.py:310`) không auth.

| # | Việc | Tham chiếu |
|---|---|---|
| 0.1 | **Sửa 2 follow-up tín hiệu đã biết**: (a) NLI premise cắt `context[:2000]` → per-claim premise selection (chọn khối `[n]` được cite làm premise cho claim đó — grounded citations đã cho sẵn mapping); (b) self-eval ceiling 0.85–1.00 → thử rubric prompt chấm theo tiêu chí + thang thô hơn, đo lại độ phân biệt | Roadmap "2 follow-up còn lại" |
| 0.2 | **Đưa grounding_ratio thành confidence signal có trọng số** (đã siết bằng `_groundable_claims`, đang observability-only) — thêm weight, renormalize | Roadmap §3.2 follow-up |
| 0.3 | **Recalibrate REFUSAL_THRESHOLD** sau 0.1+0.2 + parent-doc: full Tier-2 hai set → `calibrate_threshold.py` → chốt threshold mới + ghi lai lịch | Nợ từ 2026-07-08 |
| 0.4 | **3 blocker ingest** (§6.1): #1 point ID deterministic (`uuid5(doc_id:chunk_index)`), #3 fix chunker metadata misalignment (gate: strict-lenient gap ≈ 0 trên Tier-1), #4 `article_number` số hóa vào payload + payload index (`document_id`, `article_number`, `legal_type`, `issuance_date`, `is_in_effect`) | §6.1 #1/#3/#4 |
| 0.5 | **Re-ingest 6 docs lần cuối** vào collection mới `luat_law_ctx_v2` (đặt tên theo version, §6.4 #22) với quantization int8 + on-disk bật sẵn; giữ collection cũ đến khi Tier-1 xác nhận ≥ baseline | §6.1 #6, §6.4 #22 |

**Gate M0**: strict-lenient gap ≈ 0; Tier-1 trên collection mới ≥ baseline (recall@5 routed ≥ 0.92); threshold mới có bảng calibration + margin ghi rõ; toàn bộ test pass.
**Artefact**: bảng trước/sau per-signal (self-eval distribution, NLI premise hit-rate) — chất liệu cho phần "confidence engineering" của portfolio.

### M1 — Bậc 100 docs: đa lĩnh vực đầu tiên

*Bậc quan trọng nhất theo roadmap §6.0 — kích hoạt eval general làm answer-test, lộ mọi vấn đề đa lĩnh vực khi còn đo được.*

| # | Việc | Tham chiếu |
|---|---|---|
| 1.1 | Chọn ~100 docs: hình sự (giữ 6) + dân sự, lao động, doanh nghiệp, đất đai, hôn nhân gia đình — ưu tiên bộ luật/luật gốc + nghị định hướng dẫn chính của mỗi lĩnh vực | §6.0 |
| 1.2 | **Checkpoint/resume cho ingest** (SQLite tiến độ, `--resume`) + script embed standalone chạy Kaggle/Colab (đọc parquet → upsert thẳng Qdrant Cloud) | §6.1 #2, #5 |
| 1.3 | **Doc-alias resolution nâng cấp**: số hiệu văn bản (`59/2020/QH14`) làm khóa chính, bảng alias viết tắt (BLDS, BLLĐ, LDN, LĐĐ...), disambiguation theo năm | §6.2 #8 |
| 1.4 | **Prompts đa lĩnh vực**: classifier few-shot + corpus_summary (đã động, kiểm tra lại ở quy mô mới), hierarchy văn bản (luật > nghị định > thông tư) vào ranking/context | §6.2 #12 |
| 1.5 | **Eval mở rộng**: flip các câu general có `required_laws` đã ingest sang answer-test (#14); thêm ≥10 câu grounded/lĩnh vực mới (#16); validate ground-truth với corpus thật; giữ nhóm out-of-scope |  §6.3 #14/#16 |
| 1.6 | **Out-of-scope làm lại bản chất** (#17): corpus đa lĩnh vực → câu "ngoài phạm vi" giờ là hành chính/quốc tế/tư vấn cá nhân; đo lại scope gate + theo dõi gen-802 pattern | §6.3 #17 |
| 1.7 | **Recalibrate threshold** trên eval set mở rộng (#15) | §6.3 #15 |
| 1.8 | **Enrichment chạy lại** cho 100 docs (graph/timeline/amendment) — đo thời gian chạy để quyết #13 (batch/incremental) cần gấp chưa | §6.2 #13 |

**Gate M1**: recall@5 tụt ≤5 điểm so với M0; refusal recall = 1.0 trên out-of-scope set mới; citation accuracy ≥ 0.85; per-sector recall được báo cáo riêng (không để 1 lĩnh vực gánh trung bình).
**Artefact**: eval report đa lĩnh vực đầu tiên — con số "coverage 6 lĩnh vực" cho README.

### M2 — Bậc 600 docs: corpus đích

| # | Việc | Tham chiếu |
|---|---|---|
| 2.1 | **Capacity gate bằng số thật**: đo bytes/chunk thực tế (RAM + disk Qdrant) tại 100 docs → ngoại suy 600; nếu vượt trần free tier → cắt corpus theo ưu tiên lĩnh vực (quyết định của user: 600 co giãn theo hạ tầng) | §2.2 plan này |
| 2.2 | Ingest 600 docs qua pipeline đã có checkpoint (Kaggle GPU) | — |
| 2.3 | **Citation lookup → server-side filter** (`document_id + article_number` trên payload index từ M0) thay scroll client-side | §6.2 #7 |
| 2.4 | **Metadata store JSON → SQLite** (documents/graph/timelines đang load hết vào RAM, `_enrich_result` O(N)/result) | §6.2 #9 |
| 2.5 | **`is_in_effect` thành payload filter trong Qdrant** (đang là post-filter client-side) — point-in-time (`at_date`) là option | §6.2 #10 |
| 2.6 | **Đo reranker latency P95** với candidate pool lớn; nếu quá chậm trên CPU: giảm candidates (top-10) trước khi nghĩ đến API ngoài | §6.2 #11 |
| 2.7 | Eval mở rộng tiếp + recalibrate (mỗi bậc đều làm — nguyên tắc 4) | §6.3 |

**Gate M2**: Tier-1 giữ gate như M1 trên eval set lớn hơn; P95 end-to-end đo được và ghi lại (chưa cần đạt mục tiêu — HF Spaces CPU đo ở M5); Qdrant dùng <70% trần free tier (chừa đường mở rộng).
**Artefact**: bảng scale journey 6 → 100 → 600 kèm recall từng bậc — câu chuyện "scale có kỷ luật" cho portfolio.

**✅ M2 HOÀN THÀNH (2026-07-15, Phase E)** — collection `luat_law_ctx_v4` (600 docs, 60,958 points) thành default; REFUSAL_THRESHOLD 0.650 → **0.700**. Gates:
- **Capacity**: 9.5% RAM / 6.0% disk trần free tier (<70%) ✅ — thực tế còn thấp hơn ước tính ~26% ở §2.2.
- **Tier-1 recall@5 (routed, v1-subset apples-to-apples 100→600)**: mọi sector tụt ≤5pt ✅; overall −3.0pt (0.875→0.845); doanh_nghiep watch-item giữ nguyên.
- **Bug thật lộ ra ở scale-up & đã fix**: `VersionResolver.find_replacement` coi Văn bản hợp nhất (VBHN) là văn bản thay thế → Luật Nuôi con nuôi 2010 bị đánh `is_in_effect=False` → routed `exclude_superseded` drop → hngd tụt −25pt. Fix consolidation guard (commit f55dc9e) + re-enrichment → hngd hồi 0.60→0.80.
- **Refusal**: gate "recall=1.0" KHÔNG đạt được ở ngưỡng dùng được trên OOS set 17 câu (M1 chỉ test 1 câu) — gen-903 confident-OOS ceiling; chọn t=0.700 balanced (recall 0.765, false_refusal 0.136), Youden 0.775 bị loại vì false-refuse 27%. Known gap ghi rõ, đẩy sang M3.
- Artefact: `docs/notes/scale-journey-6-100-600.md`. Chi tiết: `.superpowers/sdd/progress.md` (Phase E log).
- P95 end-to-end + reranker latency (2.6): deferred sang M5 (đo trên HF Spaces CPU) theo gate gốc.

### M3 — SOTA theo số đo: benchmark ngoài + kỹ thuật nâng cao

*Chỉ mở sau M2 (các kỹ thuật này cần corpus thật mới đo có ý nghĩa). Từng mục độc lập — làm theo thứ tự ROI, mục nào không cải thiện số → bỏ có ghi chép.*

| # | Việc | Kỳ vọng | Tham chiếu |
|---|---|---|---|
| 3.1 | **Benchmark ngoài: ALQAC + Zalo Legal Text Retrieval** — chạy retrieval của mình trên bộ đo công khai (recall@k, MRR, nDCG); đây là "chứng chỉ khách quan" giá trị nhất với thị trường VN | Số so sánh được với leaderboard/paper | §3.5 roadmap |
| 3.2 | **Fine-tune bi-encoder legal tiếng Việt** trên cặp (câu hỏi, điều luật) từ ALQAC/Zalo + synthetic queries từ corpus (Kaggle GPU). **Gate nghiêm**: chỉ thay BGE-M3 nếu thắng trên benchmark 3.1 VÀ eval nội bộ; thua thì giữ BGE-M3 và ghi lại thí nghiệm | Nếu thắng: điểm nhấn SOTA lớn nhất của portfolio | §3.8 |
| 3.3 | **Corrective RAG nhẹ**: rerank score thấp → rewrite query + retrieve lại 1 lần trước khi refuse. Đo bằng false-refusal rate (kỳ vọng cứu nhóm crim-201/202-pattern) | Giảm false-refusal, giữ refusal recall 1.0 | §3.7 |
| 3.4 | **Reranker-on-comparison fix** (crim-203 pattern): câu so sánh cần tổng hợp 2+ điều → max single-chunk relevance thấp giả tạo. Hướng: aggregate rerank score theo nhóm document, hoặc decompose câu so sánh thành 2 sub-query (multi-hop nhẹ, đúng phạm vi `_complex_pipeline`) | Cứu nhóm câu so sánh khỏi refuse oan | Roadmap crim-203 |
| 3.5 | **Cross-reference graph vào chat**: graph expansion (`expand_context=True` của `EnhancedRetriever`) cho route complex — mảnh Phase 2 cuối chưa wire | Multi-hop recall tăng | Gap #6 roadmap |
| 3.6 | **Verbatim-span quotes**: mỗi `[n]` kèm span nguyên văn, verify bằng string-match — verifier rẻ và chặt hơn NLI cho claim trích dẫn | Hallucination check mạnh nhất chuỗi | §3.2 follow-up |
| 3.7 | **Confident-OOS ceiling (gen-903)** — deferred từ M2: OOS được trả lời tự tin vượt mọi threshold dùng được; mục tiêu NÂNG refusal recall lên trên baseline 0.765 | Thu hẹp known gap lớn nhất của refusal | M2 Phase E deferral |

**Gate M3 (restated 2026-07-15 — gate cũ "recall 1.0" unreachable theo kết quả M2)**: mỗi mục có báo cáo trước/sau; mọi mục giữ **refusal recall ≥ 0.765 VÀ false_refusal ≤ 0.136** trên OOS set, **không câu OOS nào leak mới**; riêng 3.7 nhắm nâng recall > 0.765.
**Thứ tự thực thi (chốt 2026-07-15)**: 3.1 → (3.7 → 3.3 → 3.4) → 3.2 → 3.5 → 3.6; chore `store.db`→JSON export ghép mục sớm. Spec kickoff: [`docs/superpowers/specs/2026-07-15-m3-external-benchmark-design.md`](superpowers/specs/2026-07-15-m3-external-benchmark-design.md).
**Artefact**: bảng benchmark ALQAC/Zalo + writeup thí nghiệm fine-tune (kể cả khi kết quả là "BGE-M3 zero-shot vẫn tốt hơn" — đó vẫn là kết quả).

### M4 — Production polish: memory, cache, observability

| # | Việc | Chi tiết |
|---|---|---|
| 4.1 | **Multi-turn memory**: `session_id` (đang bị vứt) → conversation buffer (Redis, N=10) + **query condensation** (follow-up "vậy còn tội cướp giật thì sao?" → LLM viết lại thành câu độc lập TRƯỚC routing — mọi route phía sau không cần biết đến history) | Gap #2 roadmap |
| 4.2 | **Semantic cache Redis**: cache theo embedding similarity của query đã condensed; TTL + invalidation khi đổi collection | §3.10 |
| 4.3 | **Langfuse Cloud (Hobby free)**: trace per-stage (route, retrieval, rerank, generate, verify), token + cost + latency per query; thay dần `print()` bằng structured logging | §3.9, Gap #3 |
| 4.4 | **Eval multi-turn**: thêm ~10 kịch bản hội thoại 2-3 lượt vào eval set (condensation đúng/sai đo được) | mới |

**Gate M4**: kịch bản follow-up trong eval pass ≥ 80%; trace Langfuse xem được đủ 6 stage; cache hit trả lời <500ms.
**Artefact**: screenshot trace Langfuse + demo multi-turn — bằng chứng "production-grade" trực quan nhất.

### M5 — Security + Deploy public

| # | Việc | Chi tiết |
|---|---|---|
| 5.1 | **Security tối thiểu đúng mức portfolio**: CORS whitelist, rate limit per-IP (slowapi/middleware), API key đơn giản cho admin endpoints (`/api/pipeline/enrich`...), không nhận secret từ client | Gap #5 roadmap |
| 5.2 | **Deploy HF Spaces (Docker Space)**: FastAPI + Streamlit cùng container (hoặc 2 Space), model lazy-load sẵn có giúp cold start; Qdrant Cloud + Upstash + Groq qua Secrets | Thay Render 512MB (đã kết luận bất khả thi §5.2 roadmap) |
| 5.3 | **Chịu đựng free tier**: warm-up endpoint + thông báo cold start trên UI; graceful degradation khi Groq hết quota (fallback message tử tế, không 500) | mới |
| 5.4 | **Smoke test từ ngoài**: script e2e đánh vào URL public (1 câu answer, 1 câu refuse, 1 citation lookup, 1 follow-up) — chạy trong CI hàng tuần | mới |

**Gate M5**: URL public sống; P95 warm đo và ghi vào README (số thật, kể cả nếu CPU chậm — trung thực hơn là hứa <3s); smoke test xanh; quét lại: không secret nào trong repo/logs.
**Artefact**: URL demo public — mặt tiền số 1 của portfolio.

### M6 — Mặt tiền portfolio

| # | Việc | Chi tiết |
|---|---|---|
| 6.1 | **README final**: số đo thật từ M0-M5 (scale journey, benchmark ngoài, refusal metrics), badges CI/eval, GIF demo, link URL public | Template §14 plan gốc |
| 6.2 | **Demo video ~5 phút**: kịch bản kể chuyện — câu trả lời có citation → câu bị từ chối (điểm bán chính!) → citation lookup → follow-up multi-turn → trace Langfuse | Plan gốc W8 |
| 6.3 | **Writeup kỹ thuật** (blog/docs): 3-4 câu chuyện có số — (a) citation route 0.125→1.0, (b) contextual embedding 0.79→0.92, (c) calibrate threshold leak-safe, (d) fine-tune embedding thắng/thua BGE-M3 | Chất liệu đã có sẵn trong roadmap doc |
| 6.4 | **Đồng bộ tài liệu**: ARCHITECTURE.md + KNOWLEDGE_GRAPH_VERSIONING.md cập nhật theo hiện trạng cuối; roadmap doc đóng băng thành "lịch sử quyết định" | — |

**Gate M6 (= Definition of Done toàn giai đoạn)**: người lạ mở README hiểu dự án trong 5 phút, bấm URL dùng được ngay, mọi con số trong README đều trace được về eval report trong repo.

---

## 4. Những thứ cố tình KHÔNG làm (giữ nguyên kết luận roadmap + bổ sung theo định vị)

- **GraphRAG full / ColBERT / agentic multi-hop phức tạp** — lý do như roadmap §3.
- **Auth/JWT/RBAC, multi-tenant, billing** — ngoài định vị portfolio; ghi vào "Future Work" README là đủ.
- **Corpus full 10-15K docs** — 600 đủ chứng minh scale có kỷ luật; nhảy tiếp là việc của giai đoạn sản phẩm thật (nếu có).
- **LangChain refactor theo cấu trúc plan gốc §6** — orchestrator hiện tại chạy đúng và test tốt; chỉ tách file khi dedupe stream/non-stream (cơ hội tự nhiên, không phải mục tiêu riêng).
- **Mua GPU serving / rerank API trả phí** — vi phạm ràng buộc $0; nếu reranker CPU quá chậm thì giảm candidates.

## 5. Rủi ro chính

| Rủi ro | Khả năng | Giảm thiểu |
|---|---|---|
| HF Spaces CPU làm P95 quá xấu (reranker + NLI) | Trung bình | Đo sớm ở M2 gate; giảm candidates; NLI đã chạy sau stream nên không chặn token đầu; ghi số thật vào README thay vì hứa |
| Groq đổi chính sách free tier | Trung bình | Gateway đã multi-key + cross-provider fallback; eval chạy đêm với sleep; nếu mất hẳn → OpenRouter free models làm fallback chain mới |
| Chất lượng đa lĩnh vực tụt không cứu được ở bậc 100 | Thấp-TB | Chính là lý do có bậc 100 trước 600; gate chặn nhảy bậc; per-sector metrics chỉ đúng chỗ đau |
| Fine-tune embedding thua BGE-M3 | Trung bình | Đã thiết kế là thí nghiệm có gate — thua vẫn ra writeup; không block milestone khác |
| Dataset HF thiếu văn bản mới/sửa đổi sau 2024 | Trung bình | Kiểm tra coverage khi chọn 100 docs (M1.1); nếu thiếu → ghi rõ giới hạn thời điểm dữ liệu vào UI disclaimer (quan trọng với legal bot) |
| Eval set phình to làm Tier-2 tốn quota | Cao | Tier-2 chỉ chạy tại milestone gate (không nightly); Tier-1 không LLM vẫn nightly |

## 6. Nhịp làm việc đề xuất

- Mỗi phiên làm việc: đọc mục "Trạng thái" của doc này + roadmap doc → chọn việc theo milestone đang mở → Sonnet implement, Fable verify → cập nhật trạng thái.
- Sau mỗi milestone: cập nhật bảng dưới + memory dự án.

## 7. Trạng thái milestone

| Milestone | Trạng thái | Ngày xong | Ghi chú |
|---|---|---|---|
| M0 — Nền sạch | ✅ Hoàn thành | 2026-07-10 | 6 commits (`5dd8f43`…). Nổi bật: 178 Điều khôi phục vào index; phân bố confidence lần đầu tách sạch (gap +0.0139); threshold 0.725→0.700 (refusal_recall 1.0, false_refusal 0.0). Chi tiết: roadmap doc mục "Trạng thái M0". |
| M1 — 100 docs | ✅ Hoàn thành | 2026-07-12 | **Gate 4/4**: recall@5 −2.5/−2.9 điểm (≤5) ✓; refusal recall 21/21 ✓; **citation accuracy 0.8533** (pooled 2 run × 51 câu, reports `20260712T014035Z`+`043708Z`; criminal 0.8922) ✓; per-sector ✓. **M1.7**: threshold **0.700→0.650** (calibration `20260712T004208Z`) — vai trò threshold ĐỔI ở 100 docs: scope gate từ chối 20/21 OOS, confidence gate chỉ bắt crim-406; t=0.650 refusal_recall 1.0, false_refusal 0.028. Đường tới gate citation (0.6667→0.8533): metric fix (refused → None, không double-count), 2 bug temporal (NĐ hướng dẫn "thay" luật mẹ + fail-closed unknown effective_date — 3 docs sống lại), 3 ground truth relabel sang luật kế nhiệm (phá sản: 03→06 tháng). Còn lại có chủ: 5 zero-cite ổn định (dn-105 retrieval miss, dn-106 cite NĐ thay luật, hg-105/ld-103/ld-106) — theo dõi cùng root-cause B (5 over-refusals scope gate) và comparison-type (M3.4). Học phí ghi nhớ: label OOS crim-401/402 stale LẦN 2 — mọi label phải verify parquet. |
| M2 — 600 docs | ✅ Hoàn thành | 2026-07-15 | **Scale-prep (branch `m2-scale-prep`, 07-13)**: 2.3 citation server-side filter (drop chunk cache), 2.4 SQLite MetadataStore, 2.5 `is_in_effect` payload + server-side `exclude_superseded`, 2.6 reranker P95 bench (CPU repeats=20: P95 @pool 10/20/50/100 = 482/894/2180/**7582** ms → **cap ~top-10 trước rerank ở M5**). **Phase E — ingest + close-out (branch `m2-phase-e-closeout`, 07-15)**: ingest 600 docs (Kaggle GPU) + enrichment → `luat_law_ctx_v4` (60,958 points) thành **default**; REFUSAL_THRESHOLD **0.650→0.700**. **Gates**: 2.1 capacity 9.5% RAM / 6.0% disk (<70%) ✓; Tier-1 routed recall@5 (v1-subset apples-to-apples 100→600) mọi sector tụt ≤5pt ✓ (overall −3.0pt 0.875→0.845, doanh_nghiep watch-item giữ); 2.7 refusal_recall=1.0 **KHÔNG đạt được** trên OOS 17 câu (M1 chỉ test 1) — gen-903 confident-OOS ceiling → chọn **t=0.700 balanced** (recall 0.765, false_refusal 0.136; Youden 0.775 loại vì false-refuse 27%). **Bug thật lộ ra & fix**: `VersionResolver.find_replacement` coi Văn bản hợp nhất (VBHN) là văn bản thay thế → Luật Nuôi con nuôi 2010 `is_in_effect=False` → routed `exclude_superseded` drop → hngd −25pt; **consolidation guard** (`f55dc9e`) + 2 TDD test + re-enrichment → hngd 0.60→0.80. Suite 349. Commits `64c51fc`→`7693f09` (đã push origin/main). Artefact `docs/notes/scale-journey-6-100-600.md`; ledger `.superpowers/sdd/progress.md`. **Deferred→M3**: confident-OOS ceiling; P95 end-to-end + reranker latency (đo ở M5 CPU); store.db→JSON export cho fresh-clone. |
| M3 — SOTA theo số đo | 🔄 Đang làm | | Kickoff 2026-07-15: gate restated (recall ≥0.765, false_refusal ≤0.136, no new leak), thêm 3.7 (gen-903), thứ tự 3.1→(3.7→3.3→3.4)→3.2→3.5→3.6. Đang chạy: **3.1 Zalo benchmark** (spec `2026-07-15-m3-external-benchmark-design.md`). |
| M4 — Memory + observability | 🔲 | | |
| M5 — Security + deploy | 🔲 | | |
| M6 — Portfolio final | 🔲 | | |
