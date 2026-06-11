# ContentCraft AI 🎬

> **Turn any topic into a narrated AI-generated video — in one prompt.**

A production-grade multimodal AI agent built with LangGraph orchestration, FLUX image generation, Coqui TTS, BLIP-2 self-evaluation, and MoviePy video assembly. Deployed on HuggingFace Spaces.

---

## What makes this an *agent* (not just a pipeline)

| Feature | How it works |
|---|---|
| **Dynamic tool dispatch** | LangGraph graph decides which tools to call and in what order |
| **Self-evaluation loop** | BLIP-2 captions each generated image; cosine similarity vs. original prompt triggers a retry if score < 0.35 |
| **Memory** | ChromaDB persists past topic→script pairs; similar topics get seeded context for variety |
| **Parallel execution** | Image gen and audio run concurrently across scenes |
| **Async job queue** | FastAPI background tasks + polling — UI never blocks |

---

## Architecture

```
Topic input
    │
    ▼
LangGraph Orchestrator  ←──→  ChromaDB Memory
    │
    ├──► Script Writer (Gemini 1.5 Flash)
    ├──► Image Generator (FLUX.1-schnell / HF Inference API)  ← parallel
    ├──► TTS (Coqui TTS)                                       ← parallel
    │
    ▼
Self-Evaluator (BLIP-2 + cosine similarity)
    │
    ├── score < 0.35 → Retry image gen
    │
    ▼
Video Assembler (MoviePy)
    │
    ▼
FastAPI backend  →  Gradio UI  →  HuggingFace Spaces
```

---

## Quick start

### 1. Clone

```bash
git clone https://github.com/YOUR_USERNAME/contentcraft-ai
cd contentcraft-ai
```

### 2. Set environment variables

Create a `.env` file:

```env
GEMINI_API_KEY=your_gemini_api_key_here
HF_API_KEY=your_huggingface_api_key_here
```

Get them free at:
- Gemini: https://aistudio.google.com/app/apikey
- HuggingFace: https://huggingface.co/settings/tokens

### 3. Install & run locally

```bash
pip install -r requirements.txt
bash start.sh
```

Then open http://localhost:7860

### 4. Deploy to HuggingFace Spaces

1. Create a new Space at https://huggingface.co/spaces
2. Select **Docker** as the SDK
3. Push this repo to the Space:
   ```bash
   git remote add space https://huggingface.co/spaces/YOUR_USERNAME/contentcraft-ai
   git push space main
   ```
4. Add your secrets in Space Settings → Repository secrets:
   - `GEMINI_API_KEY`
   - `HF_API_KEY`

---

## Project structure

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
├── Dockerfile
├── start.sh
├── requirements.txt
└── README.md
```

---

## Tech stack

| Component | Library | Why |
|---|---|---|
| Orchestration | LangGraph 0.2 | Stateful graph with conditional edges — real agent, not a chain |
| Script writing | Gemini 1.5 Flash | Fast, free tier, structured JSON output |
| Image generation | FLUX.1-schnell (HF API) | No local GPU needed, high quality |
| Text-to-speech | Coqui TTS | Lightweight, CPU-friendly, open source |
| Self-evaluation | BLIP-2 + sentence-transformers | Caption → cosine similarity retry loop |
| Memory | ChromaDB | Persistent vector store, zero setup |
| Video assembly | MoviePy | Battle-tested, good FFmpeg wrapper |
| Backend | FastAPI + uvicorn | Async, OpenAPI docs auto-generated |
| UI | Gradio 4 | HF Spaces native, multimodal widgets |
| Deployment | HuggingFace Spaces (Docker) | Free, public URL, GPU optional |

---

## Example outputs

| Topic | Style | Scenes |
|---|---|---|
| How black holes form | Documentary | 5 |
| The French Revolution | Cinematic | 6 |
| Photosynthesis explained | Educational | 4 |

---

## Resume bullet

> **ContentCraft AI** — Agentic multimodal pipeline using LangGraph orchestration, FLUX.1-schnell image generation, Coqui TTS narration, and MoviePy video assembly. Features a BLIP-2 + cosine similarity self-evaluation retry loop, ChromaDB persistent memory, FastAPI async job backend, and Gradio UI deployed on HuggingFace Spaces.

---

## Contributing

PRs welcome. Ideas:
- Add Whisper ASR for voice topic input
- Support image-to-video with AnimateDiff
- Add subtitle overlay with word-level timestamps
- MLflow experiment tracking per generation
