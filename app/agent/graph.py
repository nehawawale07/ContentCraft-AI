"""
ContentCraft AI — LangGraph Agent Graph
Orchestrates: script writing → image gen → TTS → self-eval → video assembly
"""

import os
import json
import asyncio
from typing import TypedDict, Annotated, List, Optional
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from app.agent.tools import (
    write_script,
    generate_images,
    generate_audio,
    assemble_video,
)
from app.agent.memory import memory_store


# ── State schema ──────────────────────────────────────────────────────────────

class Scene(TypedDict):
    index: int
    narration: str        # text for TTS
    image_prompt: str     # prompt for image gen
    image_path: Optional[str]
    audio_path: Optional[str]
    eval_score: Optional[float]
    retried: bool


class AgentState(TypedDict):
    topic: str
    style: str            # "educational" | "cinematic" | "documentary"
    num_scenes: int
    script: Optional[List[Scene]]
    scenes: Annotated[List[Scene], lambda a, b: b]   # always overwrite
    video_path: Optional[str]
    status: str           # current stage description
    error: Optional[str]
    job_id: str


# ── Nodes ─────────────────────────────────────────────────────────────────────

async def node_check_memory(state: AgentState) -> AgentState:
    """Check ChromaDB for similar past topics to seed the script."""
    similar = await memory_store.search(state["topic"], top_k=2)
    state["status"] = "Checking memory for similar topics..."
    if similar:
        state["_memory_context"] = similar
    return state


async def node_write_script(state: AgentState) -> AgentState:
    """Call LLM to produce a structured scene-by-scene script."""
    state["status"] = "Writing script..."
    memory_ctx = state.get("_memory_context", [])
    script = await write_script(
        topic=state["topic"],
        style=state["style"],
        num_scenes=state["num_scenes"],
        memory_context=memory_ctx,
    )
    state["script"] = script
    state["scenes"] = script   # initialise scenes list
    return state


async def node_generate_images(state: AgentState) -> AgentState:
    """Generate one image per scene in parallel."""
    state["status"] = "Generating images..."
    scenes = await generate_images(state["scenes"])
    state["scenes"] = scenes
    return state


async def node_generate_audio(state: AgentState) -> AgentState:
    """Generate TTS audio per scene in parallel."""
    state["status"] = "Generating audio narration..."
    scenes = await generate_audio(state["scenes"])
    state["scenes"] = scenes
    return state


async def node_self_evaluate(state: AgentState) -> AgentState:
    """
    Run BLIP-2 caption on each generated image and compare to the
    original image_prompt. Flag scenes with low cosine similarity.
    """
    state["status"] = "Evaluating output quality..."
    scenes = state["scenes"]
    evaluated = []
    for scene in scenes:
        if scene.get("image_path") and not scene.get("retried"):
            score = await evaluate_scene(scene)
            scene["eval_score"] = score
        evaluated.append(scene)
    state["scenes"] = evaluated
    return state


async def node_retry_failed(state: AgentState) -> AgentState:
    """Re-generate images for scenes that scored below threshold."""
    state["status"] = "Retrying low-quality scenes..."
    THRESHOLD = 0.35
    to_retry = [s for s in state["scenes"] if (s.get("eval_score") or 1.0) < THRESHOLD]
    if to_retry:
        retried = await generate_images(to_retry, retry=True)
        retried_map = {s["index"]: s for s in retried}
        new_scenes = []
        for s in state["scenes"]:
            if s["index"] in retried_map:
                r = retried_map[s["index"]]
                r["retried"] = True
                new_scenes.append(r)
            else:
                new_scenes.append(s)
        state["scenes"] = new_scenes
    return state


async def node_assemble_video(state: AgentState) -> AgentState:
    """Stitch scenes into final MP4 using MoviePy."""
    state["status"] = "Assembling video..."
    video_path = await assemble_video(
        scenes=state["scenes"],
        job_id=state["job_id"],
        style=state["style"],
    )
    state["video_path"] = video_path
    state["status"] = "Done!"

    # persist to memory so future similar topics benefit
    await memory_store.save(
        topic=state["topic"],
        script=state["script"],
    )
    return state


# ── Conditional edges ─────────────────────────────────────────────────────────

def should_retry(state: AgentState) -> str:
    THRESHOLD = 0.35
    failed = [s for s in state["scenes"] if (s.get("eval_score") or 1.0) < THRESHOLD and not s.get("retried")]
    return "retry" if failed else "assemble"


# ── Build graph ───────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    g = StateGraph(AgentState)

    g.add_node("check_memory",    node_check_memory)
    g.add_node("write_script",    node_write_script)
    g.add_node("generate_images", node_generate_images)
    g.add_node("generate_audio",  node_generate_audio)
    #g.add_node("self_evaluate",   node_self_evaluate)
    #g.add_node("retry_failed",    node_retry_failed)
    g.add_node("assemble_video",  node_assemble_video)

    g.set_entry_point("check_memory")

    g.add_edge("check_memory",    "write_script")
    g.add_edge("write_script",    "generate_images")
    g.add_edge("generate_images", "generate_audio")
    g.add_edge("generate_audio", "assemble_video")
    g.add_edge("assemble_video", END)

    return g.compile()


# Singleton compiled graph
agent_graph = build_graph()


async def run_agent(topic: str, style: str, num_scenes: int, job_id: str) -> AgentState:
    initial_state: AgentState = {
        "topic": topic,
        "style": style,
        "num_scenes": num_scenes,
        "script": None,
        "scenes": [],
        "video_path": None,
        "status": "Starting...",
        "error": None,
        "job_id": job_id,
    }
    result = await agent_graph.ainvoke(initial_state)
    return result
