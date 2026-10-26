# Red Light Violation Detection – Web UI

This adds a simple Flask web app over your existing YOLO-based detector.

## Quick start (Windows PowerShell)

1. Create and activate a virtual environment (recommended):
   - python -m venv .venv
   - .venv\Scripts\Activate.ps1

2. Install dependencies:
   - pip install -r requirements.txt

3. Ensure `best.pt` is in this folder.

4. Run the web app:
   - python app.py

5. Open the browser at http://localhost:5000 and upload a video.

Notes:
- Processing runs server-side and can take time depending on video length and hardware.
- Annotated videos are saved under `static/outputs/`. Snapshots under `violations/`.
- Live Capture is a placeholder on the home page.
- Optional: To enable no-helmet detection, place a trained YOLO model file named `helmet.pt` in this folder or set env var `HELMET_MODEL` to its path. The web UI will show no-helmet counts and snapshots.

## CLI script
Your original `main.py` is still present for local runs.