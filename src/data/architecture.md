🏗️ Phase 2 Architecture: Từ Document phẳng → Versioned Knowledge Graph

Vấn đề cốt lõi

Hiện tại mỗi document được chunk thành text phẳng, không có liên kết. Phase 2 biến 595 Luật thành một đồ thị phiên bản hóa:

HIỆN TẠI:  Doc → Chunks → Vectors → Search → Generate

TƯƠNG LAI: Doc → Parse Structure → Knowledge Graph
                                         ├── Nodes: Luật, Điều, Khoản
                                         ├── Edges: REFERENCES, AMENDS, REPLACES
                                         └── Timeline: effective_date → expiry_date
              → Version-aware Chunks → Vectors → Search → Graph Traverse → Generate

Data Model đề xuất

┌──────────────────────────────────────────────────────┐
│                    KNOWLEDGE GRAPH                     │
├──────────────┬───────────────────────────────────────┤
│    NODES     │                EDGES                  │
├──────────────┼───────────────────────────────────────┤
│ LawDocument  │  CONTAINS    Law → Article → Clause   │
│   ├─ id      │  REFERENCES  "theo Điều 123 BLDS..."  │
│   ├─ number  │  AMENDS      Luật sửa đổi → Luật gốc  │
│   ├─ status  │  REPLACES    Điều mới → Điều cũ       │
│   └─ dates   │  CITES       Dẫn chiếu luật khác      │
│              │                                       │
│ Article      │                                       │
│   ├─ number  │                                       │
│   ├─ content │                                       │
│   └─ version │                                       │
└──────────────┴───────────────────────────────────────┘

3 trụ cột — cách chúng liên kết

                     ┌──────────────────────┐
                     │   Amendments Tracking │
                     │   "Luật sửa đổi..."   │
                     └──────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │  Temporal Versioning   │
                    │  "Luật nào đang có     │
                    │   hiệu lực?"          │
                    └───────────┬───────────┘
                                │
    ┌───────────────────────────┼───────────────────────────┐
    │                           │                           │
    ▼                           ▼                           ▼
┌───────────┐           ┌───────────────┐          ┌───────────────┐
│  Article  │───REFERENCES──→│  Article B    │          │  External Law │
│  Điều 117 │           │  Điều 27 L.HN  │          │  Hiến pháp    │
└───────────┘           └───────────────┘          └───────────────┘
                    Cross-referencing

---
Để mình kiểm tra ingestion progress rồi mình sẽ vẽ thiết kế chi tiết cho từng trụ cột:

⏺ Task Output bkjqdtpxc
pre tokenize:   0%|          | 0/7 [00:00<?, ?it/s]pre tokenize: 100%|██████████| 7/7 [00:00<00:00, 384.74it/s]
Inference Embeddings:   0%|          | 0/7 [00:00<?, ?it/s]Inference Embeddings:  14%|█▍        | 1/7 [00:01<00:11,  1.85s/it]Inference Embeddings:
29%|██▊       | 2/7 [00:02<00:06,  1.40s/it]Inference Embeddings:  43%|████▎     | 3/7 [00:03<00:04,  1.22s/it]Inference Embeddings:  57%|█████▋    |
… +31 lines

⏺ Background command "Retry ingestion without recreate (resume)" completed (exit code 0)

  Thought for 7s

✅ Ingestion thành công!

📦 Collection: luat_law
📄 Documents:  60 (1/10 của 595)
📝 Chunks:     6,268 (~104 chunks/doc — Luật thường rất dài)
🔢 Vectors:    8,503 (có 1 ít duplicate từ lần chạy đầu bị timeout)

Pipeline hoạt động hoàn hảo với filter Luật + không giới hạn năm. Giờ chuyển sang Phase 2!

---
🏗️ Phase 2: Thiết kế chi tiết

Mình đề xuất chia làm 3 module, xây dựng tuần tự vì chúng phụ thuộc lẫn nhau:

📐 Tổng quan kiến trúc

scripts/ingest.py (hiện tại)
    │
    ▼
┌─────────────────────────────────────────────────────┐
│                 PHASE 2 PIPELINE                     │
│                                                      │
│  raw doc ──→ [Structure Parser] ──→ Articles/Clauses │
│                  │                                    │
│                  ├──→ [Ref Extractor] ──→ Graph edges │
│                  │                                    │
│                  ├──→ [Amendment Detector] ──→ diffs  │
│                  │                                    │
│                  └──→ [Version Resolver] ──→ timeline │
│                                                      │
│  output: VersionedArticle[] + ReferenceGraph         │
│     │                                                │
│     └──→ [Context-Aware Chunker] ──→ Qdrant          │
│                                                      │
└─────────────────────────────────────────────────────┘

Module 1: Cross-Referencing Engine

src/data/graph/
├── __init__.py
├── ref_extractor.py    # Parse reference patterns
├── graph_builder.py    # Build NetworkX graph
└── graph_traverser.py  # Query-time traversal

Pattern cần detect (ưu tiên regex):

┌──────────────────────────┬─────────────────────────────────────────┬─────────────┐
│         Pattern          │                  Ví dụ                  │ Độ phổ biến │
├──────────────────────────┼─────────────────────────────────────────┼─────────────┤
│ Điều X của Luật Y        │ "Điều 27 của Luật Hôn nhân và gia đình" │ Rất cao     │
├──────────────────────────┼─────────────────────────────────────────┼─────────────┤
│ Điều X Bộ luật Y         │ "Điều 116 Bộ luật Dân sự"               │ Cao         │
├──────────────────────────┼─────────────────────────────────────────┼─────────────┤
│ theo quy định tại Điều X │ Internal ref trong cùng luật            │ Rất cao     │
├──────────────────────────┼─────────────────────────────────────────┼─────────────┤
│ khoản X Điều Y           │ "khoản 1 Điều 117"                      │ Rất cao     │
├──────────────────────────┼─────────────────────────────────────────┼─────────────┤
│ tại Chương X             │ "tại Chương III của Luật này"           │ Trung bình  │
└──────────────────────────┴─────────────────────────────────────────┴─────────────┘

Output: ReferenceGraph — NetworkX directed graph, mỗi node là 1 Article, mỗi edge là 1 reference kèm context.

Module 2: Temporal Versioning

src/data/timeline/
├── __init__.py
├── date_extractor.py   # Extract effective/expiry dates
├── version_resolver.py  # Resolve "luật nào đang hiệu lực?"
└── timeline_index.py   # Time-aware chunk indexing

Cơ chế:
Với mỗi Điều luật:
  effective_period = (start_date, end_date)
    start_date = ngày luật có hiệu lực
    end_date   = ngày bị sửa đổi/thay thế (hoặc None nếu đang hiệu lực)

Query: "hiện nay quy định X như thế nào?"
  → Filter: effective_period.end_date IS NULL
  → Trả về version mới nhất

Query: "năm 2018 quy định X như thế nào?"
  → Filter: start_date <= 2018-01-01 AND (end_date IS NULL OR end_date > 2018-01-01)
  → Trả về version có hiệu lực tại thời điểm đó

Module 3: Amendments Tracking

src/data/amendments/
├── __init__.py
├── amendment_detector.py  # Detect "Luật sửa đổi" docs
├── instruction_parser.py  # Parse amendment instructions
└── consolidation.py       # Merge amendments into original

Pipeline:
1. Detect: Title chứa "sửa đổi, bổ sung" → amendment doc
2. Link: Tìm luật gốc qua tên luật được nhắc đến
3. Parse: Dùng LLM extract amendment instructions:
   [
     {action: "replace", article: "45", clause: "2", new_text: "..."},
     {action: "insert",  article: "123a", after: "123", new_text: "..."},
     {action: "delete",  article: "67", clause: "3"},
   ]
4. Apply: Merge amendments vào text gốc → consolidated version
5. Version: Gắn version chain
   Điều 45 (v1, 2015-2022) → Điều 45 (v2, 2022-nay)

---
🗺️ Lộ trình đề xuất

Week 1: Module 1 — Cross-Referencing
  └── ref_extractor.py + graph_builder.py
  └── Test: extract all refs from 60 docs → visualize graph

Week 2: Module 3 — Amendments Detection
  └── amendment_detector.py + instruction_parser.py
  └── Test: parse 5 amendment laws → verify merge quality

Week 3: Module 2 — Temporal Versioning
  └── version_resolver.py + timeline_index.py
  └── Test: time-aware queries against versioned chunks

Week 4: Integration + Re-ingest
  └── Full pipeline run on all 595 docs
  └── Eval: compare retrieval quality vs baseline
