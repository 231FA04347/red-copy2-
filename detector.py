import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from ultralytics import YOLO
import cv2 as cv
from datetime import datetime
from collections import defaultdict
import time
import importlib


MODEL_PATH = os.environ.get('DETECTOR_MODEL', 'best.pt')
HELMET_MODEL_PATH = os.environ.get('HELMET_MODEL', 'helmet.pt')  # optional, place a trained helmet model here
# Secondary detector for persons/motorcycles (COCO pretrained), used only for helmet logic
PERSON_MODEL_PATH = os.environ.get('PERSON_MODEL', 'yolov8n.pt')
_model = None
_helmet_model = None
_person_model = None


def get_model():
    global _model
    if _model is None:
        _model = YOLO(MODEL_PATH)
    return _model


def get_helmet_model():
    """Load optional helmet model if present; return None if not available."""
    global _helmet_model
    if _helmet_model is not None:
        return _helmet_model
    if HELMET_MODEL_PATH and os.path.exists(HELMET_MODEL_PATH):
        try:
            _helmet_model = YOLO(HELMET_MODEL_PATH)
            return _helmet_model
        except Exception:
            return None
    return None


def get_person_model():
    """Load a COCO-pretrained model (e.g., yolov8n.pt) for person/motorcycle detection.
    This is separate from the main traffic model, which may not include person class.
    """
    global _person_model
    if _person_model is not None:
        return _person_model
    try:
        if PERSON_MODEL_PATH and os.path.exists(PERSON_MODEL_PATH):
            _person_model = YOLO(PERSON_MODEL_PATH)
        else:
            # Try to load by name if file not present
            _person_model = YOLO('yolov8n.pt')
        return _person_model
    except Exception:
        return None


def process_video(input_path: str, output_path: str, save_snapshots: bool = True, frame_skip: int = 5) -> dict:
    """Run detection/tracking on a video and write an annotated MP4.

    Returns a summary dict: { 'violations': int, 'snapshots': [relative paths under /violations] }
    """
    assert os.path.exists(input_path), f"Input video not found: {input_path}"

    base_dir = os.path.dirname(os.path.abspath(__file__))
    violations_dir = os.path.join(base_dir, 'violations')
    os.makedirs(violations_dir, exist_ok=True)

    cap = cv.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {input_path}")

    fps = cap.get(cv.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH) or 854)
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT) or 480)

    # Choose output size; allow env override for speed
    out_w_env = os.environ.get('DETECTOR_OUT_W')
    out_h_env = os.environ.get('DETECTOR_OUT_H')
    if out_w_env and out_h_env and out_w_env.isdigit() and out_h_env.isdigit():
        out_size = (int(out_w_env), int(out_h_env))
    else:
        # default for consistency with original script
        out_size = (854, 480)

    # Frame skip can be overridden via env for speed
    frame_skip_env = os.environ.get('DETECTOR_FRAME_SKIP')
    effective_skip = int(frame_skip_env) if (frame_skip_env and frame_skip_env.isdigit()) else frame_skip

    # Prefer MP4 encoders; if unavailable (e.g., OpenH264 missing on Windows), fall back to AVI (MJPG/XVID)
    output_fps = max(fps / max(effective_skip, 1), 1)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out = None
    written_path = output_path
    use_frames_fallback = False
    frames_dir = None
    force_frames_env = os.environ.get('FORCE_FRAMES_FALLBACK', '0') == '1'
    writer_candidates = [
        ("avc1", ".mp4"),
        ("H264", ".mp4"),
        ("mp4v", ".mp4"),
        ("XVID", ".avi"),
        ("MJPG", ".avi"),
    ]
    if not force_frames_env:
        for codec, ext in writer_candidates:
            try:
                # If the selected encoder prefers AVI, write to a temp AVI and convert later
                target_path = output_path if ext == ".mp4" else output_path + ".tmp.avi"
                fourcc = cv.VideoWriter_fourcc(*codec)
                tmp = cv.VideoWriter(target_path, fourcc, output_fps, out_size)
                if tmp is not None and tmp.isOpened():
                    out = tmp
                    written_path = target_path
                    break
            except Exception:
                continue
    if out is None:
        # As a last resort, fall back to saving frames and assembling with MoviePy later
        frames_dir = output_path + ".frames"
        try:
            os.makedirs(frames_dir, exist_ok=True)
            use_frames_fallback = True
        except Exception as e:
            raise RuntimeError("Failed to create VideoWriter and frames fallback directory: " + str(e))

    object_y_hist = defaultdict(list)
    saved_ids = set()
    violation_timers = {}  # Track frames since red-light violation of the vehicle
    helmet_violation_timers = {}  # Track frames to flash for no-helmet riders
    seen_nohelmet_keys = set()    # Dedupe counting for no-helmet when track id is missing
    # Triple rider detection state
    triple_violation_timers = {}
    seen_triple_keys = set()
    number_of_triple_violations = 0
    flash_duration = int((fps or 25) / max(effective_skip, 1) * 2)  # Flash for ~2 seconds
    helmet_every_env = os.environ.get('DETECTOR_HELMET_EVERY')
    helmet_every = int(helmet_every_env) if (helmet_every_env and helmet_every_env.isdigit()) else 1
    # Triple rider configurable params
    triple_min_env = os.environ.get('TRIPLE_MIN_PERSONS')
    triple_min_persons = int(triple_min_env) if (triple_min_env and triple_min_env.isdigit()) else 3
    triple_iou_env = os.environ.get('TRIPLE_IOU')
    try:
        triple_iou_thresh = float(triple_iou_env) if triple_iou_env else 0.10
    except Exception:
        triple_iou_thresh = 0.10
    # Expanded zone paddings (fractions of box w/h)
    def _float_env(name, default):
        v = os.environ.get(name)
        try:
            return float(v) if v is not None else default
        except Exception:
            return default
    triple_pad_w = _float_env('TRIPLE_PAD_W', 0.35)
    triple_pad_top = _float_env('TRIPLE_PAD_TOP', 0.35)
    triple_pad_bot = _float_env('TRIPLE_PAD_BOT', 0.50)
    triple_person_conf = _float_env('TRIPLE_PERSON_CONF', 0.20)

    def is_red_light():
        current_pos_ms = cap.get(cv.CAP_PROP_POS_MSEC)
        current_pos_seconds = (current_pos_ms or 0) / 1000.0
        return current_pos_seconds > 12

    def draw_traffic_light(frame, is_red):
        cv.rectangle(frame, (800, 10), (850, 110), (50, 50, 50), -1)
        cv.rectangle(frame, (800, 10), (850, 110), (255, 255, 255), 2)
        color = (0, 0, 255) if is_red else (0, 0, 50)
        cv.circle(frame, (825, 35), 15, color, -1)
        color = (0, 255, 0) if not is_red else (0, 50, 0)
        cv.circle(frame, (825, 85), 15, color, -1)

    def flash_vehicle(vehicle_id):
        if vehicle_id in violation_timers:
            time_since_violation = violation_timers[vehicle_id]
            if time_since_violation > flash_duration:
                return False
            flash_pattern = (time_since_violation // 2) % 2 == 0
            violation_timers[vehicle_id] += 1
            return flash_pattern
        else:
            return False

    model = get_model()
    helmet_model = get_helmet_model()  # may be None
    person_model = get_person_model()  # may be None
    debug = os.environ.get('DETECTOR_DEBUG', '0') == '1'
    max_frames_env = os.environ.get('DETECTOR_MAX_FRAMES')
    max_frames = int(max_frames_env) if (max_frames_env and max_frames_env.isdigit()) else None
    if debug:
        print(f"[detector] model loaded: {MODEL_PATH}")
        print(f"[detector] helmet model: {'ON' if helmet_model is not None else 'OFF'} ({HELMET_MODEL_PATH})")
        print(f"[detector] person model: {'ON' if person_model is not None else 'OFF'} ({PERSON_MODEL_PATH})")
    number_of_violations = 0
    number_of_helmet_violations = 0
    frame_count = 0

    snapshots_rel = []
    helmet_snapshots_rel = []
    triple_snapshots_rel = []

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if debug and frame_count % 50 == 0:
                print(f"[detector] processed frames: {frame_count}")
            if frame_count % max(effective_skip, 1) != 0:
                continue

            frame_resized = cv.resize(frame, out_size)

            # Track for red-light detection using the main model (no class filter to support custom models)
            results = model.track(frame_resized, persist=True)

            annotated_frame = results[0].plot()

            # Draw wait lines
            cv.line(annotated_frame, (10, 300), (844, 315), (0, 0, 255), thickness=2)
            cv.line(annotated_frame, (844, 0), (844, 315), (0, 0, 255), thickness=2)
            cv.line(annotated_frame, (10, 0), (10, 300), (0, 0, 255), thickness=2)

            if is_red_light():
                if results[0].boxes.id is not None:
                    for box in results[0].boxes:
                        vid = int(box.id)
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        start_y = y2 - 20
                        object_y_hist[vid].append(start_y)
                        if len(object_y_hist[vid]) >= 2:
                            prev_y = object_y_hist[vid][-2]
                            curr_y = object_y_hist[vid][-1]
                            line_y = 310
                            if prev_y < line_y and curr_y >= line_y and vid not in saved_ids:
                                number_of_violations += 1
                                violation_timers[vid] = 0
                                # Crop safely
                                y1c = max(y1 - 5, 0)
                                y2c = min(y2 + 5, frame_resized.shape[0])
                                x1c = max(x1 - 5, 0)
                                x2c = min(x2 + 5, frame_resized.shape[1])
                                cropped = frame_resized[y1c:y2c, x1c:x2c]
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                filename = f"vehicle_{vid}_{timestamp}.jpg"
                                abs_snapshot = os.path.join(violations_dir, filename)
                                if save_snapshots and cropped.size > 0:
                                    cv.imwrite(abs_snapshot, cropped)
                                    # Store relative path for web usage
                                    snapshots_rel.append(os.path.join('violations', filename).replace('\\', '/'))
                                saved_ids.add(vid)

                        if flash_vehicle(vid):
                            cv.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
                            cv.putText(annotated_frame, "VIOLATION!", (x2 - 80, y2 + 25), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Helmet detection: use a dedicated COCO model for persons/motorcycles
            # Strategy: find persons (cls=0) and nearby motorcycles (cls=3), crop head and run helmet model
            person_results = None
            # Throttle person/motorcycle detection to once every N processed frames for speed
            if person_model is not None and (frame_count % max(helmet_every, 1) == 0):
                try:
                    # COCO: 0=person, 1=bicycle, 3=motorcycle
                    person_results = person_model(frame_resized, verbose=False, classes=[0, 1, 3], conf=triple_person_conf)
                except Exception:
                    person_results = None
            elif debug and person_model is None:
                print("[detector] person model unavailable; skipping person/motorcycle logic")

            if person_results is not None and len(person_results) > 0 and person_results[0].boxes is not None:
                boxes = person_results[0].boxes
                # Collect persons and motorcycles
                persons = []
                motos = []
                for b in boxes:
                    cls_id = int(b.cls[0]) if hasattr(b, 'cls') else -1
                    if cls_id == 0:  # person
                        persons.append(b)
                    elif cls_id in (1, 3):  # bicycle or motorcycle
                        motos.append(b)

                # Helper: IoU
                def iou(a, b):
                    ax1, ay1, ax2, ay2 = map(int, a.xyxy[0])
                    bx1, by1, bx2, by2 = map(int, b.xyxy[0])
                    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
                    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
                    inter = iw * ih
                    if inter == 0:
                        return 0.0
                    area_a = (ax2 - ax1) * (ay2 - ay1)
                    area_b = (bx2 - bx1) * (by2 - by1)
                    union = area_a + area_b - inter
                    return inter / max(union, 1)

                def bottom_center(b):
                    x1, y1, x2, y2 = map(int, b.xyxy[0])
                    cx = x1 + (x2 - x1) // 2
                    return cx, y2

                def expand_zone(x1, y1, x2, y2, pw=triple_pad_w, ptop=triple_pad_top, pbot=triple_pad_bot):
                    w = x2 - x1
                    h = y2 - y1
                    ex = int(w * pw)
                    et = int(h * ptop)
                    eb = int(h * pbot)
                    nx1 = max(0, x1 - ex)
                    ny1 = max(0, y1 - et)
                    nx2 = min(frame_resized.shape[1], x2 + ex)
                    ny2 = min(frame_resized.shape[0], y2 + eb)
                    return nx1, ny1, nx2, ny2

                def point_in_rect(px, py, rect):
                    rx1, ry1, rx2, ry2 = rect
                    return (px >= rx1 and px <= rx2 and py >= ry1 and py <= ry2)

                # Helper to create a stable key if track ID is missing
                def pid_key_for(b):
                    if getattr(b, 'id', None) is not None:
                        try:
                            return int(b.id)
                        except Exception:
                            pass
                    # Fallback: quantize bbox to reduce duplicates
                    x1, y1, x2, y2 = map(int, b.xyxy[0])
                    return f"bbox_{x1//10}_{y1//10}_{x2//10}_{y2//10}"

                # Evaluate each person; prefer those overlapping with a motorcycle, but don't require it
                for p in persons:
                    overlap = 0.0
                    for m in motos:
                        overlap = max(overlap, iou(p, m))

                    pidk = pid_key_for(p)

                    # Define head region (top 60% of person bbox) and pad horizontally by 10%
                    px1, py1, px2, py2 = map(int, p.xyxy[0])
                    pw, ph = max(px2 - px1, 1), max(py2 - py1, 1)
                    head_y2 = py1 + max(int(ph * 0.6), 1)
                    pad_x = max(int(pw * 0.1), 2)
                    hx1, hy1, hx2, hy2 = px1 - pad_x, py1, px2 + pad_x, head_y2
                    # Bound checks
                    hx1 = max(hx1, 0); hy1 = max(hy1, 0)
                    hx2 = min(hx2, frame_resized.shape[1]); hy2 = min(hy2, frame_resized.shape[0])
                    head_crop = frame_resized[hy1:hy2, hx1:hx2]

                    has_helmet = False
                    if helmet_model is not None and head_crop.size > 0:
                        try:
                            hr = helmet_model.predict(head_crop, verbose=False)
                            # Consider any detection as helmet present (assuming model trained to detect 'helmet')
                            # Filter by reasonable confidence threshold
                            dets = hr[0].boxes
                            if dets is not None and len(dets) > 0:
                                # Check if any detection has conf >= 0.25 (lower threshold for light models)
                                for hb in dets:
                                    conf = float(hb.conf[0]) if hasattr(hb, 'conf') else 0.0
                                    if conf >= 0.25:
                                        has_helmet = True
                                        break
                        except Exception:
                            has_helmet = False

                    # Consider rider-like only if overlapping a motorcycle sufficiently
                    is_rider_like = overlap >= 0.2

                    if not has_helmet and is_rider_like:
                        # Mark as no-helmet violation (dedupe by track id or bbox key)
                        if pidk not in helmet_violation_timers and pidk not in seen_nohelmet_keys:
                            helmet_violation_timers[pidk] = 0
                            seen_nohelmet_keys.add(pidk)
                            number_of_helmet_violations += 1
                            # Save snapshot once per key
                            y1c = max(py1 - 5, 0); y2c = min(py2 + 5, frame_resized.shape[0])
                            x1c = max(px1 - 5, 0); x2c = min(px2 + 5, frame_resized.shape[1])
                            person_crop = frame_resized[y1c:y2c, x1c:x2c]
                            if save_snapshots and person_crop.size > 0:
                                ts2 = datetime.now().strftime("%Y%m%d_%H%M%S")
                                fname = f"nohelmet_{pidk}_{ts2}.jpg"
                                abs_p = os.path.join(violations_dir, fname)
                                cv.imwrite(abs_p, person_crop)
                                helmet_snapshots_rel.append(os.path.join('violations', fname).replace('\\', '/'))

                        # Draw overlay box and label
                        cv.rectangle(annotated_frame, (px1, py1), (px2, py2), (0, 165, 255), 3)  # orange
                        cv.putText(annotated_frame, "NO HELMET", (px1, max(py1-8, 10)), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

                # Triple rider detection: for each motorcycle, count overlapping persons (IoU >= 0.2)
                # Gather additional two-wheeler zones from the main model detections if available
                extra_moto_zones = []
                try:
                    main_boxes = results[0].boxes
                    names_map = getattr(results[0], 'names', None)
                    if main_boxes is not None and names_map is not None:
                        for mb in main_boxes:
                            cls_id = int(mb.cls[0]) if hasattr(mb, 'cls') else -1
                            name = names_map.get(cls_id, '') if isinstance(names_map, dict) else ''
                            name_l = str(name).lower()
                            if ('two' in name_l and 'wheel' in name_l) or ('bike' in name_l) or ('motor' in name_l) or ('scooter' in name_l):
                                x1, y1, x2, y2 = map(int, mb.xyxy[0])
                                extra_moto_zones.append((x1, y1, x2, y2))
                except Exception:
                    pass

                # Build a unified list of motorcycle zones: from COCO motos and from main model two-wheeler-like boxes
                moto_zones = []
                # From COCO
                for m in motos:
                    mx1, my1, mx2, my2 = map(int, m.xyxy[0])
                    moto_zones.append((mx1, my1, mx2, my2))
                # From main model
                moto_zones.extend(extra_moto_zones)

                # Evaluate persons per moto zone
                for (mx1, my1, mx2, my2) in moto_zones:
                    zone = expand_zone(mx1, my1, mx2, my2)
                    persons_on_bike = []
                    for p in persons:
                        bcx, bcy = bottom_center(p)
                        if point_in_rect(bcx, bcy, zone) or iou(p, type('B', (), {'xyxy': [[mx1, my1, mx2, my2]]})) >= triple_iou_thresh:
                            persons_on_bike.append(p)
                    if debug:
                        print(f"[detector] moto zone {zone} persons_on_bike={len(persons_on_bike)}")
                    if len(persons_on_bike) >= triple_min_persons:
                        # Build a stable key using track id if available else bbox quantization
                        mkey = f"moto_bbox_{mx1//10}_{my1//10}_{mx2//10}_{my2//10}"

                        if mkey not in seen_triple_keys:
                            seen_triple_keys.add(mkey)
                            triple_violation_timers[mkey] = 0
                            number_of_triple_violations += 1
                            # Save snapshot of the motorcycle region
                            y1c = max(my1 - 5, 0); y2c = min(my2 + 5, frame_resized.shape[0])
                            x1c = max(mx1 - 5, 0); x2c = min(mx2 + 5, frame_resized.shape[1])
                            moto_crop = frame_resized[y1c:y2c, x1c:x2c]
                            if save_snapshots and moto_crop.size > 0:
                                ts3 = datetime.now().strftime("%Y%m%d_%H%M%S")
                                fname = f"triplerider_{mkey}_{ts3}.jpg"
                                abs_p = os.path.join(violations_dir, fname)
                                cv.imwrite(abs_p, moto_crop)
                                triple_snapshots_rel.append(os.path.join('violations', fname).replace('\\', '/'))

                        # Draw label on the motorcycle box
                        cv.rectangle(annotated_frame, (mx1, my1), (mx2, my2), (255, 0, 255), 3)  # magenta
                        cv.putText(annotated_frame, f"TRIPLE RIDER ({len(persons_on_bike)})", (mx1, max(my1-8, 10)), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

            # Overlay counters
            draw_traffic_light(annotated_frame, is_red_light())
            cv.rectangle(annotated_frame, (5, 4), (520, 100), (255, 255, 255), -1)
            cv.rectangle(annotated_frame, (5, 4), (520, 100), (0, 0, 0), 2)
            active_count = len(results[0].boxes) if results[0].boxes.id is not None else 0
            cv.putText(annotated_frame, f"Active Objects: {active_count}", (15, 28), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv.putText(annotated_frame, f"Red-light Violations: {number_of_violations}", (15, 50), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv.putText(annotated_frame, f"No-Helmet: {number_of_helmet_violations}", (15, 72), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv.putText(annotated_frame, f"Triple Riders: {number_of_triple_violations}", (260, 72), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            if out is not None:
                out.write(annotated_frame)
            elif use_frames_fallback:
                # Save annotated frame as an image
                if frames_dir:
                    fpath = os.path.join(frames_dir, f"frame_{frame_count:06d}.jpg")
                    ok = cv.imwrite(fpath, annotated_frame)
                    if debug and not ok:
                        print(f"[detector] WARN failed to write frame to {fpath}")

            # Optional early stop for quick tests
            if max_frames is not None and frame_count >= max_frames:
                if debug:
                    print(f"[detector] stopping early at {frame_count} frames due to DETECTOR_MAX_FRAMES")
                break

    finally:
        cap.release()
        if out is not None:
            out.release()
        cv.destroyAllWindows()

    # If we used frames fallback, assemble video directly from frames
    if use_frames_fallback and frames_dir and os.path.isdir(frames_dir):
        try:
            try:
                mpe = importlib.import_module('moviepy.editor')
                ImageSequenceClip = getattr(mpe, 'ImageSequenceClip', None)
            except ModuleNotFoundError:
                mpe = importlib.import_module('moviepy')
                ImageSequenceClip = getattr(mpe, 'ImageSequenceClip', None)
            # Collect and sort frames
            frame_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith('.jpg')])
            if len(frame_files) == 0:
                raise RuntimeError("No frames saved for assembly.")
            if ImageSequenceClip is None:
                raise RuntimeError('moviepy ImageSequenceClip not available')
            clip = ImageSequenceClip(frame_files, fps=int(round(output_fps)) or 1)
            temp_out = output_path + ".h264.mp4"
            try:
                clip.write_videofile(
                    temp_out,
                    codec="libx264",
                    audio=False,
                    preset="medium",
                    ffmpeg_params=["-movflags", "+faststart"],
                )
            except TypeError:
                clip.write_videofile(
                    temp_out,
                    codec="libx264",
                    audio=False,
                )
            clip.close()
            try:
                os.replace(temp_out, output_path)
            except Exception:
                import shutil
                shutil.copyfile(temp_out, output_path)
                os.remove(temp_out)
        except Exception:
                # Could not assemble MP4 (moviepy missing or failed); try to assemble an AVI using OpenCV
                try:
                    import cv2
                    frame_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith('.jpg')])
                    if frame_files:
                        first = cv2.imread(frame_files[0])
                        h, w = first.shape[:2]
                        avi_path = os.path.splitext(output_path)[0] + '.avi'
                        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                        vw = cv2.VideoWriter(avi_path, fourcc, int(round(output_fps)) or 1, (w, h))
                        for f in frame_files:
                            img = cv2.imread(f)
                            if img is not None and img.shape[1] == w and img.shape[0] == h:
                                vw.write(img)
                        vw.release()
                        # If avi created, we can clean frames dir as well
                        if os.path.exists(avi_path) and os.path.getsize(avi_path) > 1024:
                            output_path = avi_path
                except Exception:
                    pass
        finally:
            # Best-effort clean up frames directory only if MP4 exists
            try:
                if os.path.exists(output_path):
                    import shutil
                    shutil.rmtree(frames_dir, ignore_errors=True)
            except Exception:
                pass

        return {
            "violations": number_of_violations,
            "snapshots": snapshots_rel,
            "helmet_violations": number_of_helmet_violations,
            "helmet_snapshots": helmet_snapshots_rel,
            "helmet_model_loaded": helmet_model is not None,
            "triple_rider_violations": number_of_triple_violations,
            "triple_rider_snapshots": triple_snapshots_rel,
        }

    # Post-process to ensure browser-friendly mp4 (H.264 + moov at start)
    try:
        try:
            mpe2 = importlib.import_module('moviepy.editor')
        except ModuleNotFoundError:
            mpe2 = importlib.import_module('moviepy')
        VideoFileClip = getattr(mpe2, 'VideoFileClip', None)
        # Re-encode to H.264 with faststart from the actual written file
        temp_out = output_path + ".h264.mp4"
        if VideoFileClip is None:
            raise RuntimeError('moviepy VideoFileClip not available')
        clip = VideoFileClip(written_path)
        try:
            clip.write_videofile(
                temp_out,
                codec="libx264",
                audio=False,
                preset="medium",
                fps=int(round(output_fps)) or 1,
                ffmpeg_params=["-movflags", "+faststart"],
            )
        except TypeError:
            clip.write_videofile(
                temp_out,
                codec="libx264",
                audio=False,
                fps=int(round(output_fps)) or 1,
            )
        clip.close()
        try:
            os.replace(temp_out, output_path)
        except Exception:
            # Fallback copy+remove
            import shutil
            shutil.copyfile(temp_out, output_path)
            os.remove(temp_out)
        # Clean up temp AVI if used
        if written_path.endswith('.tmp.avi') and os.path.exists(written_path):
            try:
                os.remove(written_path)
            except Exception:
                pass
    except Exception:
        # If moviepy is not available or fails, we keep the original file
        # If we wrote to a temp AVI but couldn't convert, ensure the caller at least gets that file
        if written_path.endswith('.tmp.avi') and os.path.exists(written_path):
            try:
                # Provide the AVI as the output by renaming
                os.replace(written_path, output_path)
            except Exception:
                pass

    return {
        "violations": number_of_violations,
        "snapshots": snapshots_rel,
        "helmet_violations": number_of_helmet_violations,
        "helmet_snapshots": helmet_snapshots_rel,
        "helmet_model_loaded": helmet_model is not None,
        "triple_rider_violations": number_of_triple_violations,
        "triple_rider_snapshots": triple_snapshots_rel,
    }
