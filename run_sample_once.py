import os
from detector import process_video
from app import ensure_video_ok

# Force frames fallback to avoid OpenCV VideoWriter codec issues on Windows
os.environ['FORCE_FRAMES_FALLBACK'] = '1'
os.environ['DETECTOR_DEBUG'] = '1'
os.environ['DETECTOR_MAX_FRAMES'] = '150'

BASE = os.path.dirname(os.path.abspath(__file__))
INPUT = os.path.join(BASE, 'Traffic_video.mp4')
OUTPUT_DIR = os.path.join(BASE, 'static', 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT = os.path.join(OUTPUT_DIR, 'sample_annotated.mp4')

try:
	summary = process_video(INPUT, OUTPUT, save_snapshots=True, frame_skip=5)
	# Ensure an MP4 exists even if frames fallback was used
	ensure_video_ok(OUTPUT)
	print('\n[run_sample_once] Summary:', summary)
	print('[run_sample_once] Output video:', OUTPUT, 'exists:', os.path.exists(OUTPUT), 'size:', os.path.getsize(OUTPUT) if os.path.exists(OUTPUT) else 0)
except Exception as e:
	import traceback
	print('[run_sample_once] ERROR:', e)
	traceback.print_exc()
