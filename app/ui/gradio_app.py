"""
ContentCraft AI — Gradio Interface
Polls the FastAPI backend and streams scene previews as they arrive.
"""

import os
import time
import httpx
import gradio as gr

API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8000")
POLL_INTERVAL = 2   # seconds


STYLE_DESCRIPTIONS = {
    "educational": "🎓 Clear & engaging — great for explainers",
    "cinematic":   "🎬 Dramatic & visually rich — think Netflix doc",
    "documentary": "📽️ Factual & authoritative — neutral tone",
}


def generate_video(topic: str, style: str, num_scenes: int):
    """
    Generator function — yields UI updates as the job progresses.
    Gradio's streaming mode calls this repeatedly.
    """
    if not topic.strip():
        yield (
            gr.update(value="⚠️ Please enter a topic.", visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            [],
            gr.update(visible=False),
        )
        return

    # Map display label back to key
    style_key = style.split(" ")[0].lower().strip("🎓🎬📽️").strip()
    style_key_map = {"clear": "educational", "dramatic": "cinematic", "factual": "documentary"}
    style_key = style_key_map.get(style_key, "educational")

    # Submit job
    try:
        resp = httpx.post(
            f"{API_BASE}/generate",
            json={"topic": topic, "style": style_key, "num_scenes": int(num_scenes)},
            timeout=15,
        )
        resp.raise_for_status()
        job_id = resp.json()["job_id"]
    except Exception as e:
        yield (
            gr.update(value=f"❌ Failed to start job: {e}", visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            [],
            gr.update(visible=False),
        )
        return

    # Poll for status
    last_status = ""
    while True:
        time.sleep(POLL_INTERVAL)
        try:
            status_resp = httpx.get(f"{API_BASE}/status/{job_id}", timeout=10)
            data = status_resp.json()
        except Exception:
            continue

        current_status = data.get("status", "")

        if current_status != last_status:
            last_status = current_status
            previews = data.get("scene_previews") or []

            # Build gallery items from scene image paths
            gallery_items = []
            for scene in previews:
                img_path = scene.get("image_path")
                label = f"Scene {scene['index']+1}: {scene['narration'][:60]}..."
                if img_path and os.path.exists(img_path):
                    gallery_items.append((img_path, label))

            video_visible = current_status == "Done"
            video_path = data.get("video_path") if video_visible else None

            yield (
                gr.update(value=f"**Status:** {current_status}", visible=True),
                gr.update(value=video_path, visible=video_visible),
                gr.update(
                    value=f"{API_BASE}/download/{job_id}" if video_visible else None,
                    visible=video_visible,
                ),
                gallery_items,
                gr.update(visible=not video_visible),
            )

        if current_status in ("Done", "Error"):
            break


# ── Layout ────────────────────────────────────────────────────────────────────

with gr.Blocks(
    title="ContentCraft AI",
    theme=gr.themes.Base(
        primary_hue="violet",
        secondary_hue="slate",
        font=[gr.themes.GoogleFont("Space Grotesk"), "sans-serif"],
    ),
    css="""
    #hero { text-align: center; padding: 2rem 0 1rem; }
    #hero h1 { font-size: 2.4rem; font-weight: 700; margin-bottom: 0.25rem; }
    #hero p  { color: #888; font-size: 1rem; }
    #gen-btn { min-height: 52px; font-size: 1rem; }
    .scene-gallery img { border-radius: 8px; }
    """,
) as demo:

    # Header
    gr.HTML("""
    <div id="hero">
      <h1>🎬 ContentCraft AI</h1>
      <p>Turn any topic into a narrated video — powered by an agentic AI pipeline</p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=2):
            topic_input = gr.Textbox(
                label="Topic",
                placeholder="e.g.  How black holes form,  The French Revolution,  Photosynthesis explained",
                lines=2,
            )
            with gr.Row():
                style_input = gr.Radio(
                    choices=list(STYLE_DESCRIPTIONS.values()),
                    value=list(STYLE_DESCRIPTIONS.values())[0],
                    label="Style",
                )
            num_scenes_input = gr.Slider(
                minimum=2, maximum=8, value=4, step=1,
                label="Number of scenes",
            )
            generate_btn = gr.Button("✨ Generate Video", variant="primary", elem_id="gen-btn")

        with gr.Column(scale=3):
            status_label = gr.Markdown("Ready.", visible=True)
            progress_spinner = gr.HTML(
                "<p style='color:#888;font-size:0.9rem'>⏳ Generating…</p>",
                visible=False,
            )
            video_output = gr.Video(label="Final video", visible=False)
            download_btn  = gr.Button("⬇️ Download MP4", visible=False)

    gr.Markdown("### Scene previews")
    scene_gallery = gr.Gallery(
        label="Generated scenes",
        elem_classes=["scene-gallery"],
        columns=4,
        height=220,
    )

    gr.Markdown(
        "_Built with LangGraph · FLUX.1-schnell · Coqui TTS · BLIP-2 self-eval · MoviePy · FastAPI_",
        elem_id="footer",
    )

    # ── Events ────────────────────────────────────────────────────────────────
    generate_btn.click(
        fn=generate_video,
        inputs=[topic_input, style_input, num_scenes_input],
        outputs=[status_label, video_output, download_btn, scene_gallery, progress_spinner],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
