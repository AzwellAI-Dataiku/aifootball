# -*- coding: utf-8 -*-
"""
Local Step2 Script
MP4 input -> YOLO(person) detect -> ByteTrack(person) tracking
-> CSV (+ optional annotated mp4)

OUTPUT CSV (standardized):
frame_idx,track_id,object_type,conf,cx,cy,x1,y1,x2,y2,cx_norm,cy_norm,video_w,video_h
"""


import os
import json
import time
from pathlib import Path
from datetime import datetime


import cv2
import numpy as np
from ultralytics import YOLO


# ============================================================
# 0) 로컬 설정 (여기만 수정)
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

VIDEO_PATH = BASE_DIR / "inputs" / "videos" / "tactical_view_test.mp4"
BEST_PT_PATH = BASE_DIR / "inputs" / "models" / "best_person.pt"


# 출력 폴더
OUT_ROOT = BASE_DIR / "tracking_results" / "person_tracking"

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

# YOLO / ByteTrack 파라미터
DEVICE = "cpu"            # GPU: 0,1,... / CPU: "cpu" GPU: 0
IMGSZ  = 960
CONF   = 0.1
IOU    = 0.5
TRACKER_YAML = "bytetrack.yaml"
PERSIST = True
VID_STRIDE = 1
USE_HALF = True

# 후처리 옵션
MIN_BOX_AREA = 1200
SAVE_ANNOTATED_MP4 = True

# person 클래스명 (단일 클래스면 자동 override)
PERSON_NAME = "person"

# 시각화 옵션
CLASS_COLORS = {"person": (255, 144, 30)}
BOX_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
TEXT_THICKNESS = 2


# ============================================================
# 1) 유틸
# ============================================================

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def _is_gpu(device):
    return not (isinstance(device, str) and str(device).lower() == "cpu")

def get_video_meta(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"영상 열기 실패: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return fps, w, h, total

def draw_box_and_label(img, x1, y1, x2, y2, color, text):
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    cv2.rectangle(img, (x1, y1), (x2, y2), color, BOX_THICKNESS)
    cv2.putText(
        img, text, (x1, max(20, y1 - 6)),
        FONT, FONT_SCALE, color, TEXT_THICKNESS, cv2.LINE_AA
    )


# ============================================================
# 2) Step2 실행
# ============================================================

def main():
    run_dir = OUT_ROOT / run_id
    ensure_dir(str(run_dir))

    if not VIDEO_PATH.is_file():
        raise RuntimeError(f"VIDEO_PATH 없음: {VIDEO_PATH}")
    if not BEST_PT_PATH.is_file():
        raise RuntimeError(f"BEST_PT_PATH 없음: {BEST_PT_PATH}")

    fps, W, H, total_frames = get_video_meta(str(VIDEO_PATH))

    print("[MODEL] load:", BEST_PT_PATH)
    model = YOLO(str(BEST_PT_PATH))

    half_flag = bool(USE_HALF and _is_gpu(DEVICE))

    print("[TRACK] YOLO + ByteTrack start...")
    stream = model.track(
        source=str(VIDEO_PATH),
        stream=True,
        tracker=TRACKER_YAML,
        imgsz=IMGSZ,
        conf=CONF,
        iou=IOU,
        device=DEVICE,
        persist=PERSIST,
        half=half_flag,
        vid_stride=VID_STRIDE,
        verbose=False
    )

    rows = []

    # annotated mp4
    writer = None
    out_mp4 = None
    if SAVE_ANNOTATED_MP4:
        out_mp4 = run_dir / "person_overlay.mp4"
        writer = cv2.VideoWriter(
            out_mp4,
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(fps),
            (W, H)
        )

    frame_idx = -1
    inferred_person_name = PERSON_NAME
    debug_names_printed = False

    for r in stream:
        frame_idx += 1
        frame = r.orig_img
        names = r.names

        # 클래스명 1회 출력 + 자동 보정
        if not debug_names_printed:
            print("[DEBUG] model class names:", names)
            debug_names_printed = True
            if isinstance(names, dict) and len(names) == 1:
                inferred_person_name = str(names.get(0))
                if inferred_person_name != PERSON_NAME:
                    print(f"[AUTO] person class -> '{inferred_person_name}'")

        if r.boxes is None or len(r.boxes) == 0:
            if writer:
                writer.write(frame)
            continue

        xyxy = r.boxes.xyxy.cpu().numpy()
        cls  = r.boxes.cls.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()
        ids = r.boxes.id.cpu().numpy().astype(int) if r.boxes.id is not None else None

        render = frame.copy() if writer else None

        for i in range(len(xyxy)):
            cls_name = str(names[int(cls[i])])
            if cls_name != inferred_person_name:
                continue

            x1, y1, x2, y2 = xyxy[i]
            area = (x2 - x1) * (y2 - y1)
            if area < MIN_BOX_AREA:
                continue

            track_id = int(ids[i]) if ids is not None else -1
            score = float(confs[i])

            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)

            rows.append({
                "frame_idx": frame_idx,
                "track_id": track_id,
                "object_type": "person",
                "conf": score,
                "cx": cx,
                "cy": cy,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "cx_norm": cx / W,
                "cy_norm": cy / H,
                "video_w": W,
                "video_h": H
            })

            if render is not None:
                draw_box_and_label(
                    render, x1, y1, x2, y2,
                    CLASS_COLORS["person"],
                    f"person id{track_id} {score:.2f}"
                )

        if writer:
            writer.write(render)

    if writer:
        writer.release()

    if not rows:
        raise RuntimeError("rows 비어있음 → CONF / 클래스명 / 모델 확인 필요")

    # CSV 저장
    csv_path = run_dir / "person_tracking.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("frame_idx,track_id,object_type,conf,cx,cy,x1,y1,x2,y2,cx_norm,cy_norm,video_w,video_h\n")
        for r in rows:
            f.write(
                f"{r['frame_idx']},{r['track_id']},{r['object_type']},{r['conf']:.6f},"
                f"{r['cx']:.2f},{r['cy']:.2f},"
                f"{r['x1']:.2f},{r['y1']:.2f},{r['x2']:.2f},{r['y2']:.2f},"
                f"{r['cx_norm']:.6f},{r['cy_norm']:.6f},"
                f"{r['video_w']},{r['video_h']}\n"
            )

    # meta 저장
    meta_path = run_dir / "run_meta.json"

    meta = {
        "run_id": run_id,
        "best_pt": str(BEST_PT_PATH),
        "video": str(VIDEO_PATH),
        "rows": len(rows),
        "imgsz": IMGSZ,
        "conf": CONF,
        "device": DEVICE,
        "class_name": inferred_person_name,
        "outputs": {
            "csv": str(csv_path),
            "overlay_mp4": str(out_mp4) if SAVE_ANNOTATED_MP4 else None,
            "meta": str(meta_path)
        }
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("[DONE]")
    print("CSV :", str(csv_path))
    if out_mp4:
        print("MP4 :", str(out_mp4))


if __name__ == "__main__":
    main()
