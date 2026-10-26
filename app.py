import os
import importlib
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
import traceback

from detector import process_video


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, 'static', 'uploads')
OUTPUT_DIR = os.path.join(BASE_DIR, 'static', 'outputs')
VIOLATIONS_DIR = os.path.join(BASE_DIR, 'violations')

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VIOLATIONS_DIR, exist_ok=True)

# Ensure we can always generate output videos on Windows even without H.264 codecs
os.environ.setdefault('FORCE_FRAMES_FALLBACK', '1')


def ensure_video_ok(mp4_path: str) -> None:
    """If mp4_path doesn't exist or is tiny, and a frames folder exists, try assembling to MP4.
    This prevents 0:00 videos when encoder is missing.
    """
    try:
        ok = os.path.exists(mp4_path) and os.path.getsize(mp4_path) > 1024
        if ok:
            return
        frames_dir = mp4_path + '.frames'
        if not os.path.isdir(frames_dir):
            return
    # Try to assemble frames into an MP4
        try:
            # Prefer moviepy.editor for v1; fallback to top-level exports for v2
            try:
                mpe = importlib.import_module('moviepy.editor')
            except ModuleNotFoundError:
                mpe = importlib.import_module('moviepy')
            ImageSequenceClip = getattr(mpe, 'ImageSequenceClip', None)
            if ImageSequenceClip is None:
                print('[app] moviepy ImageSequenceClip not available; cannot assemble frames')
                return
            frame_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.lower().endswith('.jpg')])
            if not frame_files:
                print('[app] No frames found to assemble')
                return
            clip = ImageSequenceClip(frame_files, fps=24)
            temp_out = mp4_path + '.h264.mp4'
            try:
                clip.write_videofile(
                    temp_out,
                    codec='libx264',
                    audio=False,
                    preset='medium',
                    ffmpeg_params=['-movflags', '+faststart'],
                )
            except TypeError:
                # MoviePy v2 may not accept preset or ffmpeg_params; try minimal args
                clip.write_videofile(
                    temp_out,
                    codec='libx264',
                    audio=False,
                )
            clip.close()
            try:
                os.replace(temp_out, mp4_path)
            except Exception:
                import shutil
                shutil.copyfile(temp_out, mp4_path)
                os.remove(temp_out)
            print('[app] Assembled MP4 from frames at', mp4_path)
        except Exception as e:
            print('[app] Failed to assemble MP4 from frames:', e)
            # Fallback: try to assemble an AVI with MJPG using OpenCV
            try:
                import cv2 as cv
                frame_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.lower().endswith('.jpg')])
                if frame_files:
                    first = cv.imread(frame_files[0])
                    h, w = first.shape[:2]
                    avi_path = os.path.splitext(mp4_path)[0] + '.avi'
                    fourcc = cv.VideoWriter_fourcc(*'MJPG')
                    vw = cv.VideoWriter(avi_path, fourcc, 24, (w, h))
                    for f in frame_files:
                        img = cv.imread(f)
                        if img is not None and img.shape[1] == w and img.shape[0] == h:
                            vw.write(img)
                    vw.release()
                    if os.path.exists(avi_path) and os.path.getsize(avi_path) > 1024:
                        print('[app] Assembled AVI from frames at', avi_path)
            except Exception as e2:
                print('[app] AVI fallback assembly failed:', e2)
    except Exception:
        pass


def allowed_file(filename: str) -> bool:
    allowed = {'.mp4', '.avi', '.mov', '.mkv'}
    ext = os.path.splitext(filename)[1].lower()
    return ext in allowed


app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev-secret')


@app.get('/')
def index():
    return render_template('index.html')


@app.get('/upload')
def upload_get():
    # If someone browses directly to /upload via GET, send them to the form
    return redirect(url_for('index'))


@app.post('/upload')
def upload():
    if 'video' not in request.files:
        flash('No file part')
        return redirect(url_for('index'))

    file = request.files['video']
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))

    if not allowed_file(file.filename):
        flash('Unsupported file type. Please upload MP4/AVI/MOV/MKV')
        return redirect(url_for('index'))

    # Save upload
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    in_name = f"upload_{ts}{os.path.splitext(file.filename)[1].lower()}"
    in_path = os.path.join(UPLOAD_DIR, in_name)
    file.save(in_path)

    # Produce output video path
    out_name = f"annotated_{ts}.mp4"
    out_path = os.path.join(OUTPUT_DIR, out_name)

    # Run detection (blocking). For large videos, this can take time.
    try:
        summary = process_video(input_path=in_path, output_path=out_path)
    except Exception as e:
        print('[app] ERROR during processing upload:', e)
        traceback.print_exc()
        flash(f'Processing failed: {e}')
        return redirect(url_for('index'))

    # Ensure MP4 exists even if frames fallback left only images
    ensure_video_ok(out_path)

    # Render result page with the generated video (add cache-bust param)
    video_url = url_for('outputs_file', filename=out_name)
    video_url += f"?v={ts}"
    return render_template(
        'result.html',
        video_src=video_url,
        violations=summary.get('violations', 0),
        snapshots=summary.get('snapshots', []),
        helmet_violations=summary.get('helmet_violations', 0),
        helmet_snapshots=summary.get('helmet_snapshots', []),
        triple_violations=summary.get('triple_rider_violations', 0),
        triple_snapshots=summary.get('triple_rider_snapshots', []),
    )


@app.get('/violations/<path:filename>')
def violations_file(filename):
    return send_from_directory(VIOLATIONS_DIR, filename)


@app.get('/outputs/<path:filename>')
def outputs_file(filename):
    # Serve video output with best-effort format handling
    path_mp4 = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(path_mp4) and os.path.getsize(path_mp4) > 1024:
        return send_from_directory(OUTPUT_DIR, filename, mimetype='video/mp4')

    # If MP4 missing or tiny, try .avi variant
    base, ext = os.path.splitext(filename)
    alt_avi = base + '.avi'
    alt_tmp_avi = filename + '.tmp.avi'
    if os.path.exists(os.path.join(OUTPUT_DIR, alt_avi)):
        return send_from_directory(OUTPUT_DIR, alt_avi, mimetype='video/x-msvideo')
    if os.path.exists(os.path.join(OUTPUT_DIR, alt_tmp_avi)):
        return send_from_directory(OUTPUT_DIR, alt_tmp_avi, mimetype='video/x-msvideo')

    # Fallback to original even if small; browser may still attempt playback
    return send_from_directory(OUTPUT_DIR, filename, mimetype='video/mp4')


@app.get('/run-sample')
def run_sample():
    sample_path = os.path.join(BASE_DIR, 'Traffic_video.mp4')
    if not os.path.exists(sample_path):
        flash('Sample video "Traffic_video.mp4" not found in project folder.')
        return redirect(url_for('index'))

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_name = f"annotated_{ts}.mp4"
    out_path = os.path.join(OUTPUT_DIR, out_name)

    try:
        summary = process_video(input_path=sample_path, output_path=out_path)
    except Exception as e:
        print('[app] ERROR during run-sample:', e)
        traceback.print_exc()
        flash(f'Sample processing failed: {e}')
        return redirect(url_for('index'))

    # Ensure MP4 exists even if frames fallback left only images
    ensure_video_ok(out_path)
    video_url = url_for('outputs_file', filename=out_name) + f"?v={ts}"

    return render_template(
        'result.html',
        video_src=video_url,
        violations=summary.get('violations', 0),
        snapshots=summary.get('snapshots', []),
        helmet_violations=summary.get('helmet_violations', 0),
        helmet_snapshots=summary.get('helmet_snapshots', []),
        triple_violations=summary.get('triple_rider_violations', 0),
        triple_snapshots=summary.get('triple_rider_snapshots', []),
    )


if __name__ == '__main__':
    # Run locally (disable reloader to prevent resets). Allow overriding host/port via env.
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', '5000'))
    app.run(host=host, port=port, debug=False, use_reloader=False)


def ensure_video_ok(mp4_path: str) -> None:
    """If mp4_path doesn't exist or is tiny, and a frames folder exists, try assembling to MP4.
    This prevents 0:00 videos when encoder is missing.
    """
    try:
        ok = os.path.exists(mp4_path) and os.path.getsize(mp4_path) > 1024
        if ok:
            return
        frames_dir = mp4_path + '.frames'
        if not os.path.isdir(frames_dir):
            return
        # Try to assemble frames into an MP4
        try:
            try:
                mpe = importlib.import_module('moviepy.editor')
            except ModuleNotFoundError:
                mpe = importlib.import_module('moviepy')
            ImageSequenceClip = getattr(mpe, 'ImageSequenceClip', None)
            if ImageSequenceClip is None:
                print('[app] moviepy ImageSequenceClip not available; cannot assemble frames')
                return
            frame_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.lower().endswith('.jpg')])
            if not frame_files:
                print('[app] No frames found to assemble')
                return
            clip = ImageSequenceClip(frame_files, fps=24)
            temp_out = mp4_path + '.h264.mp4'
            try:
                clip.write_videofile(
                    temp_out,
                    codec='libx264',
                    audio=False,
                    preset='medium',
                    ffmpeg_params=['-movflags', '+faststart'],
                )
            except TypeError:
                clip.write_videofile(
                    temp_out,
                    codec='libx264',
                    audio=False,
                )
            clip.close()
            try:
                os.replace(temp_out, mp4_path)
            except Exception:
                import shutil
                shutil.copyfile(temp_out, mp4_path)
                os.remove(temp_out)
            print('[app] Assembled MP4 from frames at', mp4_path)
        except Exception as e:
            print('[app] Failed to assemble MP4 from frames:', e)
    except Exception:
        pass
