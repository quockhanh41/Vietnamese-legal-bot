---
title: Vietnamese Legal Knowledge Bot
emoji: 🏛️
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.20.0
app_file: src/ui/app_gradio.py
python_version: "3.12"
startup_duration_timeout: 1h
pinned: false
short_description: Adaptive RAG tra cứu 600 văn bản pháp luật Việt Nam
---

# Vietnamese Legal Knowledge Bot

Adaptive RAG over 600 Vietnamese legal documents, running on HF's free
ZeroGPU tier. First question after a cold start may take longer while models
load onto GPU. See the repo README for architecture, eval numbers, and the
self-hosted Docker deploy path (full security hardening: CORS whitelist,
per-session rate limit, admin key).
