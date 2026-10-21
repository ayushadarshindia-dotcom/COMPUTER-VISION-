"""
Drone safety monitor v3: fight/force detection, lying / person-down / fall
recognition, and group caution. Alerts show on screen, print in the console,
save a snapshot, and beep (Windows). No phone call or extra accounts needed.

Run (inside your venv):
    python drone_safety_monitor.py --source http://192.168.1.2:8080/video
    python drone_safety_monitor.py --source video.mp4
    python drone_safety_monitor.py --source 0

What you will see:
    green box   person standing / moving normally
    yellow box  person lying down (timer counts up)
    red box     ALERT: PERSON DOWN (lying longer than --down-seconds)
    red box     ALERT: POSSIBLE FIGHT (close people, strong sustained arm movement)
    orange box  CAUTION: group of N

Options:
    --down-seconds 5     seconds lying before PERSON DOWN alert
    --fall-drop 0.35     how fast a drop to the ground must be to count as a FALL
                         (in body-sizes within 1 second; lower = more sensitive)
    --fight-seconds 2    seconds of aggressive arm movement before FIGHT alert
    --fight-speed 2.0    arm-speed threshold (higher = fewer false alarms)
    --fight-dist 1.0     people must be within this many body-sizes to count
    --group-size 5       minimum people for the group caution
    --no-beep            turn off the alert sound
    --no-show            no video window (use on the drone itself)
    --debug              show torso angle and box shape under each person (for tuning)
    --skip-frames 0      run the AI on 1 of every N+1 frames (0 = every frame);
                         2 = about 10 AI frames/s from a 30 FPS camera
    --imgsz 640          AI input size (416 or 320 is faster, a bit less accurate)
    --model NAME         model file; on a Raspberry Pi use yolov8n-pose_ncnn_model

The AI speed (frames analysed per second) shows in the corner of the video
and prints in the console every 5 seconds.

Press q in the window to quit. Alert snapshots go to the 'alerts' folder.
"""
import argparse
import os
import time
from collections import defaultdict, deque

import cv2
import numpy as np
from ultralytics import YOLO

try:
    import winsound  # Windows only
except ImportError:
    winsound = None

KP_CONF = 0.5          # minimum keypoint confidence to trust a body point
WRISTS = [9, 10]       # COCO keypoint indices for left and right wrist


def is_lying(box, kpts):
    """Decide if a person is lying down. Returns (lying, info_text).

    Needs a clearly visible torso (a shoulder AND a hip). Sitting at a desk or
    behind a table hides the hips, so then we cannot tell and say "not lying".
    A lying person also has a box that is not taller than it is wide.
    """
    x1, y1, x2, y2 = box
    w, h = max(x2 - x1, 1), max(y2 - y1, 1)
    aspect = w / h
    shoulders = [k[:2] for k in (kpts[5], kpts[6]) if k[2] > KP_CONF]
    hips = [k[:2] for k in (kpts[11], kpts[12]) if k[2] > KP_CONF]

    if shoulders and hips:
        sh = np.mean(shoulders, axis=0)
        hp = np.mean(hips, axis=0)
        dx, dy = abs(sh[0] - hp[0]), abs(sh[1] - hp[1])
        angle = np.degrees(np.arctan2(dx, dy))  # 0 = upright, 90 = flat
        lying = dx > dy * 1.2 and aspect > 0.9
        return lying, f"torso {angle:.0f}deg  box w/h {aspect:.2f}"

    # torso not clearly visible: only trust a very wide, flat box
    lying = aspect > 1.8
    return lying, f"torso hidden  box w/h {aspect:.2f}"


def find_groups(centers, sizes, min_size, dist_factor):
    """Group people who stand close together (union-find). Returns lists of indices."""
    n = len(centers)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            limit = dist_factor * max(sizes[i], sizes[j])
            if np.linalg.norm(centers[i] - centers[j]) < limit:
                parent[find(i)] = find(j)

    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return [g for g in groups.values() if len(g) >= min_size]


def run(args):
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")

    model = YOLO(args.model, task="pose")  # yolov8n-pose.pt downloads on first run
    os.makedirs("alerts", exist_ok=True)

    lying_since, last_seen, alerted = {}, {}, set()
    y_hist = defaultdict(list)            # tid -> [(time, centre_y, size)] for last 1 s
    prev_rel, arm_speed = {}, defaultdict(float)
    fight_since, fight_alerted = {}, set()
    group_alert_time = 0.0
    frame_times = deque(maxlen=30)   # for the AI speed counter
    last_fps_print = 0.0
    frame_count = 0

    def trigger(kind, text, frame):
        stamp = time.strftime("%Y%m%d_%H%M%S")
        print(f"[{stamp}] ALERT: {text}")
        cv2.imwrite(f"alerts/{kind}_{stamp}.jpg", frame)
        if winsound and not args.no_beep:
            winsound.PlaySound("SystemExclamation",
                               winsound.SND_ALIAS | winsound.SND_ASYNC)

    while True:
        # grab() reads a frame cheaply; retrieve() decodes it. Skipped frames are
        # only grabbed, so the camera stream never falls behind.
        if not cap.grab():
            break
        frame_count += 1
        if args.skip_frames and frame_count % (args.skip_frames + 1) != 0:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            break
        scale = args.width / frame.shape[1]
        frame = cv2.resize(frame, None, fx=scale, fy=scale)
        now = time.time()

        frame_times.append(now)
        ai_fps = 0.0
        if len(frame_times) > 1 and frame_times[-1] > frame_times[0]:
            ai_fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0])
        if ai_fps and now - last_fps_print > 5:
            last_fps_print = now
            print(f"[{time.strftime('%H:%M:%S')}] AI speed: {ai_fps:.1f} frames/s")

        result = model.track(frame, persist=True, tracker="bytetrack.yaml",
                             imgsz=args.imgsz, verbose=False)[0]

        ids, boxes_i, centers, sizes = [], [], [], []
        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            track_ids = result.boxes.id.int().cpu().tolist()
            kpts_all = result.keypoints.data.cpu().numpy()

            for box, tid, kpts in zip(boxes, track_ids, kpts_all):
                x1, y1, x2, y2 = box.astype(int)
                center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
                size = max(x2 - x1, y2 - y1, 1)
                ids.append(tid)
                boxes_i.append((x1, y1, x2, y2))
                centers.append(center)
                sizes.append(size)
                last_seen[tid] = now

                # keep the last second of vertical position (for fall detection)
                hist = y_hist[tid]
                hist.append((now, center[1], size))
                while hist and now - hist[0][0] > 1.0:
                    hist.pop(0)

                # arm speed: wrist movement relative to the body centre, per second,
                # in body sizes. Using the relative position cancels camera shake.
                wr = kpts[WRISTS]
                rel = np.column_stack([wr[:, :2] - center, wr[:, 2]])
                speed = 0.0
                if tid in prev_rel:
                    prev_t, prev = prev_rel[tid]
                    dt = max(now - prev_t, 1e-3)
                    for k in range(2):
                        if rel[k, 2] > KP_CONF and prev[k, 2] > KP_CONF:
                            d = np.linalg.norm(rel[k, :2] - prev[k, :2])
                            speed = max(speed, d / dt / size)
                prev_rel[tid] = (now, rel)
                arm_speed[tid] = 0.6 * arm_speed[tid] + 0.4 * speed

                # lying / fall / person down
                color, label = (0, 200, 0), f"person #{tid}"
                lying, info = is_lying(box, kpts)
                if lying:
                    if tid not in lying_since:
                        lying_since[tid] = now
                        # a fall = the body centre dropped fast just before lying
                        t0, y0, s0 = hist[0]
                        if (center[1] - y0) / s0 > args.fall_drop and now - t0 > 0.1:
                            trigger("fall", f"Possible fall detected, person {tid}", frame)
                    secs = now - lying_since[tid]
                    if secs >= args.down_seconds:
                        color = (0, 0, 255)
                        label = f"ALERT: PERSON DOWN #{tid} ({secs:.0f}s)"
                        if tid not in alerted:
                            alerted.add(tid)
                            trigger("down", f"Person down, lying for {secs:.0f} seconds", frame)
                    else:
                        color = (0, 255, 255)
                        label = f"lying #{tid} ({secs:.0f}s)"
                else:
                    lying_since.pop(tid, None)
                    alerted.discard(tid)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, max(y1 - 6, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                if args.debug:
                    cv2.putText(frame, info, (x1, min(y2 + 16, frame.shape[0] - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # forget people who left the view for more than 2 seconds
        for tid in [t for t, s in last_seen.items() if now - s > 2.0]:
            last_seen.pop(tid)
            lying_since.pop(tid, None)
            alerted.discard(tid)
            prev_rel.pop(tid, None)
            arm_speed.pop(tid, None)
            y_hist.pop(tid, None)

        # fight / force: two people close together, strong sustained arm movement
        active_pairs = set()
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                close = (np.linalg.norm(centers[i] - centers[j])
                         < args.fight_dist * max(sizes[i], sizes[j]))
                active = max(arm_speed[ids[i]], arm_speed[ids[j]]) > args.fight_speed
                if not (close and active):
                    continue
                pair = tuple(sorted((ids[i], ids[j])))
                active_pairs.add(pair)
                fight_since.setdefault(pair, now)
                if now - fight_since[pair] >= args.fight_seconds:
                    bx = np.array([boxes_i[i], boxes_i[j]])
                    cv2.rectangle(frame, (bx[:, 0].min(), bx[:, 1].min()),
                                  (bx[:, 2].max(), bx[:, 3].max()), (0, 0, 255), 3)
                    cv2.putText(frame, "ALERT: POSSIBLE FIGHT",
                                (int(bx[:, 0].min()), max(int(bx[:, 1].min()) - 24, 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    if pair not in fight_alerted:
                        fight_alerted.add(pair)
                        trigger("fight", "Possible fight detected", frame)
        for pair in list(fight_since):
            if pair not in active_pairs:
                fight_since.pop(pair)
                fight_alerted.discard(pair)

        # group gathering (caution only)
        if len(centers) >= args.group_size:
            for group in find_groups(centers, sizes, args.group_size, args.group_dist):
                pts = np.array([centers[i] for i in group])
                gx1, gy1 = pts.min(axis=0).astype(int) - 30
                gx2, gy2 = pts.max(axis=0).astype(int) + 30
                cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), (0, 140, 255), 3)
                cv2.putText(frame, f"CAUTION: group of {len(group)}", (gx1, max(gy1 - 8, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)
                if now - group_alert_time > 10:
                    group_alert_time = now
                    print(f"[{time.strftime('%H:%M:%S')}] CAUTION: group of {len(group)} people")

        cv2.putText(frame, f"people: {len(ids)}   AI FPS: {ai_fps:.1f}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if not args.no_show:
            cv2.imshow("Drone safety monitor", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="0")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--down-seconds", type=float, default=5.0)
    p.add_argument("--fall-drop", type=float, default=0.35)
    p.add_argument("--fight-seconds", type=float, default=2.0)
    p.add_argument("--fight-speed", type=float, default=2.0)
    p.add_argument("--fight-dist", type=float, default=1.0)
    p.add_argument("--group-size", type=int, default=5)
    p.add_argument("--group-dist", type=float, default=1.5)
    p.add_argument("--no-beep", action="store_true")
    p.add_argument("--no-show", action="store_true")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--skip-frames", type=int, default=0)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--model", default="yolov8n-pose.pt")
    run(p.parse_args())
