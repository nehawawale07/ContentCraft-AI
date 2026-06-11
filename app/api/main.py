"""
ContentCraft AI — FastAPI Backend
Exposes: POST /generate  GET /status/{job_id}  GET /download/{job_id}
"""

import asyncio
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.agent.graph import run_agent

app = FastAPI(title="ContentCraft AI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job store (swap for Redis in production)
jobs: dict[str, dict] = {}


# ── Request / Response models ─────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    topic: str = Field(..., min_length=3, max_length=200, example="The life cycle of a star")
    style: str = Field("educational", pattern="^(educational|cinematic|documentary)$")
    num_scenes: int = Field(4, ge=2, le=8)


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    video_path: Optional[str] = None
    scene_previews: Optional[list] = None
    error: Optional[str] = None


# ── Background task ───────────────────────────────────────────────────────────

async def run_job(job_id: str, topic: str, style: str, num_scenes: int):
    jobs[job_id]["status"] = "Running..."
    try:
        result = await run_agent(
            topic=topic,
            style=style,
            num_scenes=num_scenes,
            job_id=job_id,
        )
        jobs[job_id].update({
            "status": "Done",
            "video_path": result.get("video_path"),
            "scene_previews": [
                {
                    "index": s["index"],
                    "narration": s["narration"],
                    "image_path": s.get("image_path"),
                    "eval_score": s.get("eval_score"),
                }
                for s in result.get("scenes", [])
            ],
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        jobs[job_id].update({"status": "Error", "error": str(e)})


# ── Routes ────────────────────────────────────────────────────────────────────

@app.post("/generate", response_model=JobStatusResponse, status_code=202)
async def generate(req: GenerateRequest, background_tasks: BackgroundTasks):
    job_id = uuid.uuid4().hex
    jobs[job_id] = {"status": "Queued", "video_path": None, "error": None}
    background_tasks.add_task(run_job, job_id, req.topic, req.style, req.num_scenes)
    return JobStatusResponse(job_id=job_id, status="Queued")


@app.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, detail="Job not found")
    return JobStatusResponse(job_id=job_id, **job)


@app.get("/download/{job_id}")
async def download_video(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, detail="Job not found")
    if job["status"] != "Done":
        raise HTTPException(400, detail=f"Job not ready: {job['status']}")
    video_path = job.get("video_path")
    if not video_path or not Path(video_path).exists():
        raise HTTPException(404, detail="Video file not found")
    return FileResponse(video_path, media_type="video/mp4", filename=f"contentcraft_{job_id}.mp4")


@app.get("/health")
async def health():
    return {"status": "ok"}
