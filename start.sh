#!/bin/bash
# Start FastAPI backend in background, then launch Gradio UI on port 7860

set -e

echo "Starting ContentCraft AI..."

# Start FastAPI on port 8000
uvicorn app.api.main:app --host 0.0.0.0 --port 8000 &
FASTAPI_PID=$!

# Wait for FastAPI to be ready
echo "Waiting for API to be ready..."
for i in {1..20}; do
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        echo "API is up!"
        break
    fi
    sleep 1
done

# Start Gradio UI (foreground — HF Spaces needs this to stay alive)
export API_BASE_URL="http://localhost:8000"
python app/ui/gradio_app.py

# If Gradio exits, also kill FastAPI
kill $FASTAPI_PID
