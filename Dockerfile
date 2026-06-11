# ContentCraft AI — Dockerfile
# Optimised for HuggingFace Spaces (CPU tier, 16 GB RAM)
# HF Spaces requires app to listen on port 7860

FROM python:3.11-slim

# System deps for MoviePy / audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    espeak-ng \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python deps first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Create output dirs
RUN mkdir -p outputs/images outputs/audio outputs/video chroma_db

# HF Spaces runs as non-root user 1000
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

# Expose Gradio port (HF Spaces requirement)
EXPOSE 7860

# Start FastAPI in the background, then Gradio in the foreground
# The startup script handles both processes
CMD ["bash", "start.sh"]
