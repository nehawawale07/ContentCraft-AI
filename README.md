# 🎬 ContentCraft AI

> **Turn any topic into a fully narrated, AI-generated video — in one prompt.**

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-7F77DD)](https://www.langchain.com/langgraph)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Gradio](https://img.shields.io/badge/UI-Gradio-FF6B6B?logo=gradio&logoColor=white)](https://gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

ContentCraft AI is an autonomous multimodal agent that takes a single topic and produces a complete narrated video — it writes the script, generates scene artwork, synthesizes voiceover, **self-evaluates its own visuals**, and assembles everything into a final video. No manual editing required.

> 🖥️ Runs locally via Docker / FastAPI + Gradio — see [Quick start](#-quick-start) below.

---



## 🤖 What makes this an *agent* (not just a pipeline)

| Feature | How it works |
|---|---|
| 🧠 **Dynamic tool dispatch** | LangGraph graph decides which tools to call and in what order |
| 🔁 **Self-evaluation loop** | BLIP-2 captions each generated image; cosine similarity vs. the original prompt triggers a retry if the score falls below `0.35` |
| 💾 **Memory** | ChromaDB persists past topic→script pairs; similar topics get seeded context for variety |
| ⚡ **Parallel execution** | Image generation and audio synthesis run concurrently across scenes |
| 📡 **Async job queue** | FastAPI background tasks + polling — the UI never blocks |

---

## 🗺️ Architecture

```
                         ┌─────────────────────┐
                         │      Topic input      │
                         └──────────┬───────────┘
                                    │
                                    ▼
                  ┌──────────────────────────────────┐
                  │      LangGraph Orchestrator        │◄──────┐
                  └──────────────────┬─────────────────┘       │
                                     │                          │
        ┌────────────────┬──────────┴──────────┬───────────┐   │
        ▼                ▼                     ▼           │   │
┌───────────────┐ ┌─────────────────┐  ┌───────────────┐    │   │
│ Script Writer │ │ Image Generator  │  │      TTS       │    │   │
│ (Gemini 1.5    │ │ (FLUX.1-schnell │  │  (Coqui TTS)   │    │   │
│   Flash)       │ │   via HF API)   │  │                │    │   │
└───────┬───────┘ └────────┬─────────┘  └───────┬────────┘    │   │
        │                  │  (parallel)        │             │   │
        │                  ▼                    │             │   │
        │        ┌───────────────────────┐      │             │   │
        │        │     Self-Evaluator      │      │             │   │
        │        │  (BLIP-2 + cosine sim)  │      │             │   │
        │        └──────────┬──────────────┘      │             │   │
        │                   │                     │             │   │
        │           score < 0.35?                 │             │   │
        │              │         │                │             │   │
        │            yes        no                │             │   │
        │              │         │                │             │   │
        │              ▼         ▼                │             │   │
        │         ┌─────────┐  (continue)         │             │   │
        │         │  Retry   │                    │             │   │
        │         │ image gen│                    │             │   │
        │         └────┬────┘                     │             │   │
        │              └──────────────┐           │             │   │
        │                             ▼           ▼             │   │
        │                  ┌─────────────────────────┐          │   │
        └─────────────────►│     Video Assembler       │         │   │
                            │        (MoviePy)          │        │   │
                            └────────────┬─────────────┘         │   │
                                         │                        │   │
                                         ▼                        │   │
                            ┌─────────────────────────┐          │   │
                            │     FastAPI backend       │──────────┘   │
                            │ /generate /status         │              │
                            │ /download /health          │             │
                            └────────────┬─────────────┘               │
                                         │                              │
                                         ▼                              │
                            ┌─────────────────────────┐                │
                            │       Gradio UI           │               │
                            └────────────┬─────────────┘                │
                                         │                               │
                                         ▼                               │
                            ┌─────────────────────────┐                 │
                            │   HuggingFace Spaces       │◄───────────────┘
                            │       (Docker)             │
                            └─────────────────────────┘

     ChromaDB Memory ◄──────► LangGraph Orchestrator
     (persists topic → script pairs for context reuse)
```

---

## ⚙️ How it works

1. Enter a **topic** — e.g. *"How does a heart attack occur?"*
2. Choose a **style**:
   - 🎓 Clear & Engaging — great for explainers
   - 🎬 Dramatic & Visually Rich — think Netflix doc
   - 📋 Factual & Authoritative — neutral tone
3. Pick the **number of scenes**
4. Hit **Generate Video** — the LangGraph orchestrator takes over:
   - 📝 Script Writer breaks the topic into per-scene narration
   - 🖼️ Image Generator + 🎙️ TTS run in **parallel** per scene
   - 🔍 Self-Evaluator checks each image against its prompt, **retrying low-scoring ones**
   - 🎞️ Video Assembler stitches narration, images, and audio into a final MP4
5. Watch live progress via the async status bar, then preview & download

---

## 🚀 Quick start

### 1. Clone

```bash
git clone https://github.com/nehawawale07/ContentCraft-AI
cd ContentCraft-AI
```

### 2. Set environment variables

Create a `.env` file:

```env
GEMINI_API_KEY=your_gemini_api_key_here
HF_API_KEY=your_huggingface_api_key_here
```

Get free keys at:
- Gemini → https://aistudio.google.com/app/apikey
- HuggingFace → https://huggingface.co/settings/tokens

### 3. Install & run locally

```bash
pip install -r requirements.txt
bash start.sh
```

Then open **http://localhost:7860** 🎉

---

## 📁 Project structure

```
contentcraft-ai/
├── app/
│   ├── agent/
│   │   ├── graph.py      # LangGraph nodes + conditional edges
│   │   ├── tools.py      # script_writer, image_gen, tts, evaluator, assembler
│   │   └── memory.py     # ChromaDB vector store wrapper
│   ├── api/
│   │   └── main.py       # FastAPI: /generate /status /download /health
│   └── ui/
│       └── gradio_app.py # Gradio interface with live scene gallery
├── docs/                  # Screenshots, GIFs, diagrams for README
├── Dockerfile
├── start.sh
├── requirements.txt
└── README.md
```

---

## 🧰 Tech stack

| Component | Library | Why |
|---|---|---|
| Orchestration | LangGraph 0.2 | Stateful graph with conditional edges — real agent, not a chain |
| Script writing | Gemini 1.5 Flash | Fast, free tier, structured JSON output |
| Image generation | FLUX.1-schnell (HF API) | No local GPU needed, high quality |
| Text-to-speech | Coqui TTS | Lightweight, CPU-friendly, open source |
| Self-evaluation | BLIP-2 + sentence-transformers | Caption → cosine similarity retry loop |
| Memory | ChromaDB | Persistent vector store, zero setup |
| Video assembly | MoviePy | Battle-tested FFmpeg wrapper |
| Backend | FastAPI + uvicorn | Async, auto-generated OpenAPI docs |
| UI | Gradio 4 | Local web UI, multimodal widgets |
| Containerization | Docker | Reproducible local setup |

---

## 🎯 Example outputs

| Topic | Style | Scenes |
|---|---|---|
| How does a heart attack occur? | Dramatic & visually rich | 2 |
| The French Revolution | Cinematic | 6 |
| Photosynthesis explained | Educational | 4 |

---



---

## 🛣️ Roadmap / Contributing

PRs welcome! Some ideas:

- [ ] Add Whisper ASR for voice topic input
- [ ] Support image-to-video with AnimateDiff
- [ ] Add subtitle overlay with word-level timestamps
- [ ] MLflow experiment tracking per generation
- [ ] Improve narration text overlay accuracy

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
