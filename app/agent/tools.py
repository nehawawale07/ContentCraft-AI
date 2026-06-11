"""
ContentCraft AI — Agent Tools
Each tool is an async function called by a LangGraph node.
"""

import os
import json
import asyncio
import uuid
import textwrap
from pathlib import Path
from typing import List, Optional

import httpx
import torch
from PIL import Image
import numpy as np

from dotenv import load_dotenv
load_dotenv()

# ── Output dirs ───────────────────────────────────────────────────────────────
OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)

for sub in ["images", "audio", "video"]:
    (OUTPUTS_DIR / sub).mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. SCRIPT WRITER  (Gemini 1.5 Flash via REST)
# ─────────────────────────────────────────────────────────────────────────────

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"

STYLE_INSTRUCTIONS = {
    "educational": "Clear, engaging, suited for a general audience. Each scene should teach one concept.",
    "cinematic":   "Dramatic, visually rich, with vivid scene descriptions. Think Netflix documentary.",
    "documentary": "Factual, authoritative tone. Neutral narration with striking visual cues.",
}

SCRIPT_PROMPT = textwrap.dedent("""
    You are a professional video scriptwriter.
    
    Topic: {topic}
    Style: {style} — {style_instruction}
    Number of scenes: {num_scenes}
    {memory_hint}
    
    Return ONLY a JSON array with exactly {num_scenes} objects. No markdown fences.
    Each object must have:
    - "index": integer starting at 0
    - "narration": 1–3 sentence spoken narration for this scene (will be read aloud by TTS)
    - "image_prompt": detailed visual description for an AI image generator (style, mood, composition, lighting)
    
    Make each scene distinct. The narration should flow naturally from one scene to the next.
""")



async def write_script(
    topic: str,
    style: str,
    num_scenes: int,
    memory_context: list,
) -> list:
    api_key = os.environ["GEMINI_API_KEY"]

    memory_hint = ""
    if memory_context:
        hint_topics = ", ".join([m["topic"] for m in memory_context])
        memory_hint = (
            f"Previously generated similar topics for variety reference: {hint_topics}"
        )

    prompt = SCRIPT_PROMPT.format(
        topic=topic,
        style=style,
        style_instruction=STYLE_INSTRUCTIONS.get(style, ""),
        num_scenes=num_scenes,
        memory_hint=memory_hint,
    )

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.8,
            "maxOutputTokens": 2048,
        },
    }

    timeout = httpx.Timeout(180.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        print("CALLING GEMINI...")
        resp = await client.post(
            f"{GEMINI_URL}?key={api_key}",
            json=payload,
        )

        print("GEMINI STATUS:", resp.status_code)

        if resp.status_code != 200:
            print("GEMINI RESPONSE:", resp.text[:1000])

        resp.raise_for_status()

        raw = resp.json()["candidates"][0]["content"]["parts"][0]["text"]

    raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()

    scenes = json.loads(raw)

    for s in scenes:
        s.setdefault("image_path", None)
        s.setdefault("audio_path", None)
        s.setdefault("eval_score", None)
        s.setdefault("retried", False)

    return scenes




# ─────────────────────────────────────────────────────────────────────────────
# 2. IMAGE GENERATOR  (FLUX.1-schnell via HuggingFace Inference API)
# ─────────────────────────────────────────────────────────────────────────────

HF_IMAGE_URL = "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell"


async def _generate_single_image(
    scene: dict,
    client: httpx.AsyncClient,
    retry: bool,
) -> dict:
    api_key = os.environ["HF_API_KEY"]

    prompt = scene["image_prompt"]

    if retry:
        prompt = f"High quality, detailed: {prompt}"

    payload = {
        "inputs": prompt,
        "parameters": {
            "num_inference_steps": 4
        }
    }

    resp = await client.post(
        HF_IMAGE_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json=payload,
        timeout=90,
    )

    print("HF STATUS:", resp.status_code)
    print("HF RESPONSE:", resp.text[:500])

    resp.raise_for_status()

    img_path = (
        OUTPUTS_DIR
        / "images"
        / f"scene_{scene['index']}_{uuid.uuid4().hex[:6]}.png"
    )

    img_path.write_bytes(resp.content)

    scene["image_path"] = str(img_path)

    return scene




async def generate_images(scenes: list, retry: bool = False) -> list:
    async with httpx.AsyncClient() as client:
        tasks = [_generate_single_image(s, client, retry) for s in scenes]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    out = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            print(f"[IMAGE ERROR] Scene {i}: {repr(r)}")
            scenes[i]["image_path"] = None
            out.append(scenes[i])
        else:
            out.append(r)

    return out




# ─────────────────────────────────────────────────────────────────────────────
# 3. TEXT-TO-SPEECH  (Coqui TTS — runs locally)
# ─────────────────────────────────────────────────────────────────────────────

async def _generate_single_audio(scene: dict) -> dict:
    audio_path = OUTPUTS_DIR / "audio" / f"scene_{scene['index']}.mp3"
    loop = asyncio.get_event_loop()

    def _synth():
        from gtts import gTTS
        tts = gTTS(text=scene["narration"], lang="en", slow=False)
        tts.save(str(audio_path))

    await loop.run_in_executor(None, _synth)
    scene["audio_path"] = str(audio_path)
    return scene


async def generate_audio(scenes: list) -> list:
    # TTS is CPU-bound — run sequentially to avoid OOM on free tier
    results = []
    for scene in scenes:
        try:
            s = await _generate_single_audio(scene)
        except Exception as e:
            print(f"[TTS] Scene {scene['index']} failed: {e}")
            scene["audio_path"] = None
        results.append(scene)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 4. SELF-EVALUATOR  (BLIP-2 captioning + cosine similarity)
# ─────────────────────────────────────────────────────────────────────────────

_blip_processor = None
_blip_model = None
_embedder = None


def _get_blip():
    global _blip_processor, _blip_model
    if _blip_model is None:
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        _blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        _blip_model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _blip_model.to(device)
    return _blip_processor, _blip_model


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def _cosine(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


async def evaluate_scene(scene: dict) -> float:
    if not scene.get("image_path"):
        return 1.0   # no image → skip penalty

    loop = asyncio.get_event_loop()

    def _run():
        processor, model = _get_blip()
        image = Image.open(scene["image_path"]).convert("RGB")
        device = next(model.parameters()).device
        inputs = processor(image, return_tensors="pt").to(device)
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=50)
        caption = processor.decode(generated[0], skip_special_tokens=True)

        embedder = _get_embedder()
        emb_caption = embedder.encode(caption)
        emb_prompt  = embedder.encode(scene["image_prompt"])
        score = _cosine(emb_caption, emb_prompt)
        return score

    score = await loop.run_in_executor(None, _run)
    return score


# ─────────────────────────────────────────────────────────────────────────────
# 5. VIDEO ASSEMBLER  (MoviePy)
# ─────────────────────────────────────────────────────────────────────────────

FALLBACK_COLOR = (15, 15, 30)   # dark navy for missing images


async def assemble_video(scenes: list, job_id: str, style: str) -> str:
    loop = asyncio.get_event_loop()

    def _build():
        from moviepy.editor import (
            ImageClip,
            AudioFileClip,
            ColorClip,
            concatenate_videoclips,
        )

        clips = []

        for scene in sorted(scenes, key=lambda s: s["index"]):

            # Audio
            if scene.get("audio_path"):
                audio = AudioFileClip(scene["audio_path"])
                duration = audio.duration
            else:
                audio = None
                duration = 4.0

            print(f"Scene {scene['index']}")
            print("Image:", scene.get("image_path"))
            print("Audio:", scene.get("audio_path"))
            print("Duration:", duration)

            # Image
            if scene.get("image_path") and os.path.exists(scene["image_path"]):
                img_clip = (
                    ImageClip(scene["image_path"])
                    .set_duration(duration)
                    .resize(height=720)
                )
            else:
                img_clip = ColorClip(
                    size=(1280, 720),
                    color=FALLBACK_COLOR,
                    duration=duration,
                )

            if audio:
                img_clip = img_clip.set_audio(audio)

            clips.append(img_clip)

        print("TOTAL CLIPS:", len(clips))

        final = concatenate_videoclips(
            clips,
            method="compose"
        )

        out_path = str(
            OUTPUTS_DIR / "video" / f"{job_id}.mp4"
        )

        final.write_videofile(
            out_path,
            fps=24,
            codec="libx264",
            audio_codec="aac",
            preset="medium",
            logger="bar",
        )

        return out_path

    return await loop.run_in_executor(None, _build)

    path = await loop.run_in_executor(None, _build)
    return path
