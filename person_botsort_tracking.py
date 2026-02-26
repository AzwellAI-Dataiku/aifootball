# -*- coding: utf-8 -*-
"""
Local Python Script - YOLO person tracking with BoT-SORT (offline tactical view)
- Dataiku 의존성 제거, 로컬 파일 경로로 변환
- 설정값 및 로직은 Dataiku 버전과 동일

산출물 구조:
  tracking_results/person_tracking/
    └── YYYYMMDD_HHMMSS/
          ├── {video_stem}_tracked.mp4   ← 트래킹 영상
          ├── tracks_frame_level.csv     ← 프레임 단위 트랙 데이터
          ├── meta.yaml                  ← 실행 파라미터 메타정보
          ├── botsort_custom.yaml        ← 트래커 설정 (재현용)
          └── labels/                    ← 프레임별 YOLO txt (파일 다수)
                ├── 000001.txt
                └── ...
"""

import os
import shutil
import time
import glob
from pathlib import Path
from collections import defaultdict

import yaml


# ============================================================
# 0) 로컬 경로 설정
# ============================================================

BASE_DIR = Path(__file__).parent

INPUT_VIDEO_DIR  = BASE_DIR / "inputs" / "videos"
INPUT_MODEL_PATH = BASE_DIR / "inputs" / "models" / "best_person_nano.pt"
OUTPUT_BASE_DIR  = BASE_DIR / "tracking_results" / "person_tracking"


# ============================================================
# 1) 트래킹 파라미터 (Dataiku 버전과 동일하게 유지)
# ============================================================

DEVICE  = "cpu"
IMGSZ   = 1280
CONF    = 0.25
IOU     = 0.7
HALF    = False

# ===== (1) 시각화: 박스/텍스트 얇게 =====
LINE_WIDTH  = 1
SHOW_LABELS = True
SHOW_CONF   = False

# ===== (2) BoT-SORT 커스텀 설정 =====
USE_CUSTOM_BOTSORT       = True
BOTSORT_TRACK_BUFFER     = 90
BOTSORT_MATCH_THRESH     = 0.8
BOTSORT_NEW_TRACK_THRESH = 0.25

# ===== (3) 오프라인 stitching 설정 =====
ENABLE_BORDER_STITCHING = True
STITCH_MAX_GAP_FRAMES   = 45
STITCH_MAX_DIST_PX      = 120
BORDER_PX               = 40
MIN_TRACKLET_LEN        = 8


# ============================================================
# 2) 유틸
# ============================================================

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def try_parse_line(parts):
    """
    Ultralytics tracking save_txt 포맷 파싱.
    7컬럼: cls  track_id  cx  cy  w  h  conf
    6컬럼: cls  cx  cy  w  h  conf  (track 없는 경우, track_id=0)
    """
    nums = [float(p) for p in parts]

    if len(nums) >= 7:
        # cls track_id cx cy w h conf
        cls      = int(nums[0])
        track_id = int(nums[1])
        cx, cy, w, h = nums[2], nums[3], nums[4], nums[5]
        conf = nums[6]
        return (track_id, cls, cx, cy, w, h, conf)

    if len(nums) == 6:
        # cls cx cy w h conf
        cls      = int(nums[0])
        track_id = 0
        cx, cy, w, h = nums[1], nums[2], nums[3], nums[4]
        conf = nums[5]
        return (track_id, cls, cx, cy, w, h, conf)

    return None

def xywhn_to_xyxy(x, y, w, h, img_w, img_h):
    x1 = (x - w / 2.0) * img_w
    y1 = (y - h / 2.0) * img_h
    x2 = (x + w / 2.0) * img_w
    y2 = (y + h / 2.0) * img_h
    return x1, y1, x2, y2


# ============================================================
# 3) 출력 폴더 생성 (날짜_시간 단일 폴더)
# ============================================================

RUN_TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")
RUN_DIR       = OUTPUT_BASE_DIR / RUN_TIMESTAMP
LABELS_DIR    = RUN_DIR / "labels"

ensure_dir(RUN_DIR)
ensure_dir(LABELS_DIR)

print(f"[INFO] 출력 폴더: {RUN_DIR}")


# ============================================================
# 4) 입력 파일 확인
# ============================================================

video_exts = [".mp4", ".mov", ".mkv", ".avi"]
video_paths = [str(p) for p in INPUT_VIDEO_DIR.rglob("*") if p.suffix.lower() in video_exts]

if not video_paths:
    raise RuntimeError(f"[ERROR] 입력 영상이 없습니다: {INPUT_VIDEO_DIR}")

print(f"[INFO] 입력 영상 수: {len(video_paths)}")
for vp in video_paths:
    print(f"  - {vp}")

local_best = str(INPUT_MODEL_PATH)
if not os.path.exists(local_best):
    raise RuntimeError(f"[ERROR] 모델 파일이 없습니다: {local_best}")

print(f"[MODEL] {local_best}")


# ============================================================
# 5) BoT-SORT 커스텀 yaml 생성 (run 폴더에 저장 → 재현 가능)
# ============================================================

tracker_path = "botsort.yaml"  # 기본값 (USE_CUSTOM_BOTSORT=False 시)

if USE_CUSTOM_BOTSORT:
    tracker_yaml_path = str(RUN_DIR / "botsort_custom.yaml")
    botsort_cfg = {
        "tracker_type": "botsort",
        "track_high_thresh": 0.5,
        "track_low_thresh": 0.1,
        "new_track_thresh": float(BOTSORT_NEW_TRACK_THRESH),
        "track_buffer": int(BOTSORT_TRACK_BUFFER),
        "match_thresh": float(BOTSORT_MATCH_THRESH),
        "fuse_score": True,
        "gmc_method": "sparseOptFlow",
        "proximity_thresh": 0.5,
        "appearance_thresh": 0.25,
        "with_reid": True,
        "model": "auto"
    }
    with open(tracker_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(botsort_cfg, f, sort_keys=False, allow_unicode=True)
    tracker_path = tracker_yaml_path
    print(f"[CFG] BoT-SORT yaml 저장: {tracker_yaml_path}")


# ============================================================
# 6) meta.yaml 저장 (실행 파라미터 기록)
# ============================================================

meta = {
    "run_timestamp": RUN_TIMESTAMP,
    "model": local_best,
    "videos": video_paths,
    "tracker": tracker_path,
    "params": {
        "device": DEVICE,
        "imgsz": IMGSZ,
        "conf": CONF,
        "iou": IOU,
        "half": HALF,
        "line_width": LINE_WIDTH,
        "show_labels": SHOW_LABELS,
        "show_conf": SHOW_CONF,
    },
    "botsort": {
        "use_custom": USE_CUSTOM_BOTSORT,
        "track_buffer": BOTSORT_TRACK_BUFFER,
        "match_thresh": BOTSORT_MATCH_THRESH,
        "new_track_thresh": BOTSORT_NEW_TRACK_THRESH,
    },
    "stitching": {
        "enabled": ENABLE_BORDER_STITCHING,
        "max_gap_frames": STITCH_MAX_GAP_FRAMES,
        "max_dist_px": STITCH_MAX_DIST_PX,
        "border_px": BORDER_PX,
        "min_tracklet_len": MIN_TRACKLET_LEN,
    }
}

meta_path = RUN_DIR / "meta.yaml"
with open(meta_path, "w", encoding="utf-8") as f:
    yaml.safe_dump(meta, f, sort_keys=False, allow_unicode=True, default_flow_style=False)
print(f"[META] 저장: {meta_path}")


# ============================================================
# 7) Tracking (BoT-SORT)
#    Ultralytics가 project/name 구조로 임시 저장하므로
#    트래킹 완료 후 영상과 labels를 RUN_DIR로 재배치
# ============================================================

from ultralytics import YOLO
model = YOLO(local_best)

YOLO_TMP_DIR = RUN_DIR / "_yolo_tmp"
ensure_dir(YOLO_TMP_DIR)

print("[TRACK] Starting tracking...")
print(f"[TRACK] tracker={tracker_path}, imgsz={IMGSZ}, conf={CONF}, iou={IOU}, half={HALF}, device={DEVICE}")
print(f"[TRACK] line_width={LINE_WIDTH}, show_labels={SHOW_LABELS}, show_conf={SHOW_CONF}")

for vp in video_paths:
    stem = Path(vp).stem
    print(f"[TRACK] video: {vp}")

    try:
        _ = model.track(
            source=vp,
            tracker=tracker_path,
            imgsz=IMGSZ,
            conf=CONF,
            iou=IOU,
            device=DEVICE,
            half=HALF,
            save=True,
            save_txt=True,
            save_conf=True,
            project=str(YOLO_TMP_DIR),
            name=stem,
            persist=True,
            line_width=LINE_WIDTH,
            show_labels=SHOW_LABELS,
            show_conf=SHOW_CONF
        )
    except TypeError:
        _ = model.track(
            source=vp,
            tracker=tracker_path,
            imgsz=IMGSZ,
            conf=CONF,
            iou=IOU,
            device=DEVICE,
            half=HALF,
            save=True,
            save_txt=True,
            save_conf=True,
            project=str(YOLO_TMP_DIR),
            name=stem,
            persist=True,
            line_thickness=LINE_WIDTH,
            show_labels=SHOW_LABELS,
            show_conf=SHOW_CONF
        )

    tmp_video_dir = YOLO_TMP_DIR / stem

    # 산출물 영상 → RUN_DIR/{stem}_tracked.{ext}
    for vf in tmp_video_dir.rglob("*"):
        if vf.suffix.lower() in video_exts and vf.is_file():
            dst = RUN_DIR / f"{stem}_tracked{vf.suffix}"
            shutil.move(str(vf), str(dst))
            print(f"[OUTPUT] 영상 저장: {dst.name}")

    # labels/*.txt → RUN_DIR/labels/
    # (Ultralytics 버전마다 저장 경로가 다를 수 있으므로 tmp 전체 재귀 탐색)
    moved = 0
    for lf in tmp_video_dir.rglob("*.txt"):
        dst_lf = LABELS_DIR / lf.name
        shutil.move(str(lf), str(dst_lf))
        moved += 1
    print(f"[OUTPUT] labels 이동: {moved}개 → labels/")

# 임시 폴더 정리
shutil.rmtree(str(YOLO_TMP_DIR), ignore_errors=True)
print("[TRACK] Tracking done.")


# ============================================================
# 8) Export to CSV + Border-aware stitching
# ============================================================

try:
    import cv2
except Exception:
    cv2 = None
    print("[WARN] opencv-python(cv2) 없음. 경계 stitching 품질이 떨어질 수 있습니다.")

def get_video_wh(video_stem: str):
    if cv2 is None:
        return None, None
    cand = glob.glob(str(INPUT_VIDEO_DIR / "**" / f"{video_stem}.*"), recursive=True)
    vid = None
    for c in cand:
        if os.path.splitext(c)[1].lower() in video_exts:
            vid = c
            break
    if not vid:
        return None, None
    cap = cv2.VideoCapture(vid)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return w, h

def is_near_border(x1, y1, x2, y2, W, H, border_px):
    if W is None or H is None:
        return False
    return (x1 <= border_px) or (y1 <= border_px) or (x2 >= W - border_px) or (y2 >= H - border_px)

# labels 수집 (이미 RUN_DIR/labels/ 에 정리됨)
label_txts = sorted(LABELS_DIR.glob("*.txt"))
print(f"[EXPORT] label 파일 수: {len(label_txts)}")

# 영상이 하나면 그 stem, 여럿이면 각 프레임 파일이 어느 영상 것인지
# Ultralytics는 영상별로 labels를 분리하지 않으므로 영상이 1개인 경우를 기본으로 처리
default_video_stem = Path(video_paths[0]).stem if len(video_paths) == 1 else "multi_video"

rows = []
for txt_path in label_txts:
    try:
        frame_idx = int(txt_path.stem)
    except:
        frame_idx = None

    W, H = get_video_wh(default_video_stem)

    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parsed = None
            try:
                parsed = try_parse_line(line.split())
            except:
                parsed = None
            if parsed is None:
                continue

            track_id, cls, x, y, w, h, conf = parsed

            if W is not None and H is not None:
                x1, y1, x2, y2 = xywhn_to_xyxy(x, y, w, h, W, H)
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                cx_norm = cx / W
                cy_norm = cy / H
            else:
                x1, y1, x2, y2 = x, y, w, h
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                cx_norm = None
                cy_norm = None

            rows.append({
                "frame_idx": frame_idx,
                "track_id": track_id,
                "object_type": "person",
                "conf": conf,
                "cx": cx,
                "cy": cy,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "cx_norm": cx_norm,
                "cy_norm": cy_norm,
                "video_w": W,
                "video_h": H,
            })

# ---- 경계 재등장 stitching ----
def border_stitch(rows):
    by_video = defaultdict(list)
    for r in rows:
        by_video[default_video_stem].append(r)

    stitched_all = []
    for video, items in by_video.items():
        items = [it for it in items if it["frame_idx"] is not None]
        items.sort(key=lambda x: (x["frame_idx"], x["track_id"]))

        W, H = get_video_wh(video)

        tracks = defaultdict(list)
        for it in items:
            tracks[it["track_id"]].append(it)

        tinfo = {}
        for tid, seq in tracks.items():
            seq.sort(key=lambda z: z["frame_idx"])
            s = seq[0]; e = seq[-1]
            len_ = len(seq)

            border_start = is_near_border(s["x1"], s["y1"], s["x2"], s["y2"], W, H, BORDER_PX)
            border_end   = is_near_border(e["x1"], e["y1"], e["x2"], e["y2"], W, H, BORDER_PX)

            tinfo[tid] = {
                "start": s["frame_idx"], "end": e["frame_idx"],
                "c_start": (s["cx"], s["cy"]), "c_end": (e["cx"], e["cy"]),
                "border_start": border_start,
                "border_end": border_end,
                "len": len_
            }

        remap = {}

        def gid(tid):
            while tid in remap:
                tid = remap[tid]
            return tid

        tids = list(tinfo.keys())
        tids.sort(key=lambda t: tinfo[t]["end"])

        for a in tids:
            a2 = gid(a)
            if a2 != a:
                continue
            ia = tinfo[a]
            if ia["len"] < MIN_TRACKLET_LEN:
                continue
            if not ia["border_end"]:
                continue

            best_b = None
            best_d = 1e18

            for b in tids:
                b2 = gid(b)
                if b2 != b:
                    continue
                if b == a:
                    continue
                ib = tinfo[b]
                if ib["len"] < MIN_TRACKLET_LEN:
                    continue
                if not ib["border_start"]:
                    continue

                gap = ib["start"] - ia["end"]
                if gap < 1 or gap > STITCH_MAX_GAP_FRAMES:
                    continue

                ax, ay = ia["c_end"]
                bx, by = ib["c_start"]
                d = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5

                if d < best_d and d <= STITCH_MAX_DIST_PX:
                    best_d = d
                    best_b = b

            if best_b is not None:
                remap[best_b] = a
                ia2 = tinfo[a]
                ib2 = tinfo[best_b]
                ia2["end"] = ib2["end"]
                ia2["c_end"] = ib2["c_end"]
                ia2["border_end"] = ib2["border_end"]

        for it in items:
            new_it = dict(it)
            new_it["track_id"] = gid(it["track_id"])
            stitched_all.append(new_it)

    return stitched_all

if ENABLE_BORDER_STITCHING and rows:
    print("[STITCH] Border-aware stitching 적용 중...")
    rows = border_stitch(rows)

# CSV 저장 (템플릿 스키마)
csv_path = RUN_DIR / "person_tracking.csv"
with open(csv_path, "w", encoding="utf-8") as f:
    f.write("frame_idx,track_id,object_type,conf,cx,cy,x1,y1,x2,y2,cx_norm,cy_norm,video_w,video_h\n")
    for rr in rows:
        f.write(
            f"{rr['frame_idx']},{rr['track_id']},{rr['object_type']},{rr['conf']:.6f},"
            f"{rr['cx']:.2f},{rr['cy']:.2f},"
            f"{rr['x1']:.2f},{rr['y1']:.2f},{rr['x2']:.2f},{rr['y2']:.2f},"
            f"{rr['cx_norm']:.6f},{rr['cy_norm']:.6f},"
            f"{rr['video_w']},{rr['video_h']}\n"
        )

print(f"[EXPORT] CSV 저장: {csv_path.name}")


# ============================================================
# 9) 완료 요약
# ============================================================

print("\n" + "=" * 60)
print(f"[DONE] 출력 폴더: {RUN_DIR}")
print(f"  ├── *_tracked.mp4          ← 트래킹 영상")
print(f"  ├── person_tracking.csv      ← 프레임 트랙 데이터")
print(f"  ├── meta.yaml              ← 실행 파라미터")
print(f"  ├── botsort_custom.yaml    ← 트래커 설정")
print(f"  └── labels/               ← 프레임별 txt ({len(label_txts)}개)")
print("=" * 60)