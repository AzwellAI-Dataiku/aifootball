# -*- coding: utf-8 -*-
"""
Local Python Script - FINAL Tactical Ball Tracking (Offline, Accuracy-First)
- Same algorithmic structure, but WITHOUT Dataiku I/O
- Inputs: local VIDEO_PATH, local BEST_PT_PATH
- Outputs: local OUT_DIR/{run_id}/ball_track.csv, run_meta.json, (optional) ball_overlay.mp4

CSV schema (standardized):
frame_idx,track_id,object_type,conf,cx,cy,cx_norm,cy_norm,video_w,video_h
"""

import os
import json
import math
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import cv2
from ultralytics import YOLO


# =========================
# 0) Inputs/Outputs
# =========================
BASE_DIR = Path(__file__).resolve().parent

VIDEO_PATH = BASE_DIR / "inputs" / "videos" / "tactical_view_test.mp4"
BEST_PT_PATH = BASE_DIR / "inputs" / "models" / "best_ball.pt"

RESULTS_ROOT = BASE_DIR / "tracking_results" / "ball_tracking"

# ✅ run_id를 날짜_시분초로 통일
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

SAVE_OVERLAY = True
OVERLAY_FPS = None  # None이면 원본 fps 사용


# =========================
# 1) YOLO params (Recall-first, offline)
# =========================
IMGSZ = 1280
CONF_THRES = 0.02          # ✅ 후보 풀 넓히기
IOU_THRES  = 0.5
MAX_DET    = 80


# =========================
# 2) Soft priors (NO hard filtering)
# =========================
# FG score (soft)
USE_FG_SOFT = True
DIFF_THRESH = 18
MORPH_KERNEL = (5, 5)
DILATE_ITER = 2
ERODE_ITER  = 1
W_FG = 6.0                  # node_cost에 -(W_FG * fg_ratio)

# Top ROI (soft penalty)
USE_TOP_SOFT = True
TOP_BAN_RATIO = 0.15
GREEN_RATIO_THRESH = 0.25
TOP_PENALTY = 60.0           # 상단 영역 후보면 비용 가산(continue X)


# =========================
# 3) Camera motion estimation (optional, cost-only)
# =========================
USE_CAM_MOTION = True
CAM_MOTION_INTERVAL = 2
ORB_FEATURES = 800
MIN_MATCHES = 25
RANSAC_THRESH = 3.0


# =========================
# 4) Optical flow 후보 추가 (pool widening)
# =========================
USE_OPT_FLOW_CAND = True
FLOW_WIN = 31
FLOW_MAX_LEVEL = 3
FLOW_ERR_MAX = 35.0
FLOW_CONF = 0.12


# =========================
# 5) Global optimization (MCF-equivalent shortest path) params
# =========================
MAX_GAP = 8
MAX_LINK_DIST = 320.0

# Node cost (lower is better)
NODE_BIAS = 1.5
W_CONF = 10.0                # -W_CONF * conf

# Edge cost
W_MOTION = 1.0
GAP_PENALTY = 18.0

# Miss cost per frame
MISS_PENALTY_PER_FRAME = 6.0

# size prior (soft)
USE_SIZE_PRIOR = True
SIZE_TOL_RATIO = 2.2
W_SIZE = 45.0


# =========================
# 6) RTS Kalman smoother (post-processing final)
# =========================
USE_RTS_SMOOTHER = True
Q_POS = 1.0
Q_VEL = 3.0
R_MEAS = 9.0

# fallback smoothing if RTS off
SMOOTH_WINDOW = 7
MAX_JUMP_GATE_PX = 280


# =========================
# Utilities
# =========================
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def box_center(x1, y1, x2, y2):
    return (0.5*(x1+x2), 0.5*(y1+y2))

def box_area(x1, y1, x2, y2):
    return float(max(1.0, x2-x1) * max(1.0, y2-y1))

def green_ratio(frame_bgr):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (35, 30, 30), (90, 255, 255))
    return float(np.count_nonzero(mask)) / float(mask.size)

def fg_mask_from_prev(curr_bgr, prev_gray, A_prev_to_curr=None):
    curr_gray = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)
    if prev_gray is None:
        return curr_gray, None

    if A_prev_to_curr is not None:
        warped_prev = cv2.warpAffine(
            prev_gray, A_prev_to_curr, (curr_gray.shape[1], curr_gray.shape[0]),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
        )
        diff = cv2.absdiff(curr_gray, warped_prev)
    else:
        diff = cv2.absdiff(curr_gray, prev_gray)

    _, mask = cv2.threshold(diff, DIFF_THRESH, 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_KERNEL)
    mask = cv2.dilate(mask, k, iterations=DILATE_ITER)
    mask = cv2.erode(mask, k, iterations=ERODE_ITER)
    return curr_gray, mask

def fg_ratio(mask, x1, y1, x2, y2):
    if mask is None:
        return 0.0
    H, W = mask.shape[:2]
    x1 = int(max(0, min(W-1, x1))); x2 = int(max(0, min(W, x2)))
    y1 = int(max(0, min(H-1, y1))); y2 = int(max(0, min(H, y2)))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    crop = mask[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0
    return float(np.count_nonzero(crop)) / float(crop.size)

def estimate_affine(prev_gray, curr_gray, orb=None):
    if prev_gray is None or curr_gray is None:
        return None, False
    if orb is None:
        orb = cv2.ORB_create(nfeatures=ORB_FEATURES)
    kp1, des1 = orb.detectAndCompute(prev_gray, None)
    kp2, des2 = orb.detectAndCompute(curr_gray, None)
    if des1 is None or des2 is None or len(kp1) < MIN_MATCHES or len(kp2) < MIN_MATCHES:
        return None, False
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    if matches is None or len(matches) < MIN_MATCHES:
        return None, False
    matches = sorted(matches, key=lambda m: m.distance)[:min(len(matches), 140)]
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
    A, _ = cv2.estimateAffinePartial2D(
        pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_THRESH
    )
    if A is None:
        return None, False
    return A.astype(np.float32), True

def apply_affine_to_point(A, x, y):
    if A is None:
        return float(x), float(y)
    nx = A[0,0]*x + A[0,1]*y + A[0,2]
    ny = A[1,0]*x + A[1,1]*y + A[1,2]
    return float(nx), float(ny)

def moving_average_1d(x, w):
    if w <= 1:
        return x
    pad = w // 2
    xpad = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(w)/w
    return np.convolve(xpad, kernel, mode="valid")

def gate_and_interpolate(xs, ys, max_jump):
    x = xs.copy()
    y = ys.copy()
    for i in range(1, len(x)):
        if np.isnan(x[i]) or np.isnan(y[i]) or np.isnan(x[i-1]) or np.isnan(y[i-1]):
            continue
        d = math.hypot(x[i]-x[i-1], y[i]-y[i-1])
        if d > max_jump:
            x[i] = np.nan
            y[i] = np.nan
    s = pd.Series(x).interpolate(limit_direction="both").to_numpy(dtype=float)
    t = pd.Series(y).interpolate(limit_direction="both").to_numpy(dtype=float)
    return s, t


# =========================
# RTS Kalman smoother (x,y,vx,vy)
# =========================
def rts_smoother_xy(meas_xy, dt=1.0, q_pos=1.0, q_vel=3.0, r_meas=9.0):
    T = len(meas_xy)
    F = np.array([[1,0,dt,0],
                  [0,1,0,dt],
                  [0,0,1,0 ],
                  [0,0,0,1 ]], dtype=np.float64)
    Hm = np.array([[1,0,0,0],
                   [0,1,0,0]], dtype=np.float64)
    Q = np.diag([q_pos, q_pos, q_vel, q_vel]).astype(np.float64)
    R = np.diag([r_meas, r_meas]).astype(np.float64)
    I = np.eye(4, dtype=np.float64)

    x = np.zeros((4,1), dtype=np.float64)
    first = None
    for (xx,yy) in meas_xy:
        if not (np.isnan(xx) or np.isnan(yy)):
            first = (xx,yy)
            break
    if first is None:
        return np.full(T, np.nan), np.full(T, np.nan)

    x[0,0], x[1,0] = first[0], first[1]
    P = np.diag([1000,1000,1000,1000]).astype(np.float64)

    xs_f, Ps_f = [None]*T, [None]*T
    xs_p, Ps_p = [None]*T, [None]*T

    for t in range(T):
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q
        xs_p[t], Ps_p[t] = x_pred, P_pred

        mx, my = meas_xy[t]
        if not (np.isnan(mx) or np.isnan(my)):
            z = np.array([[mx],[my]], dtype=np.float64)
            yk = z - (Hm @ x_pred)
            S = Hm @ P_pred @ Hm.T + R
            K = P_pred @ Hm.T @ np.linalg.inv(S)
            x = x_pred + K @ yk
            P = (I - K @ Hm) @ P_pred
        else:
            x, P = x_pred, P_pred

        xs_f[t], Ps_f[t] = x, P

    xs_s, Ps_s = [None]*T, [None]*T
    xs_s[-1], Ps_s[-1] = xs_f[-1], Ps_f[-1]

    for t in range(T-2, -1, -1):
        P_f = Ps_f[t]
        P_p = Ps_p[t+1]
        C = P_f @ F.T @ np.linalg.inv(P_p)
        xs_s[t] = xs_f[t] + C @ (xs_s[t+1] - xs_p[t+1])
        Ps_s[t] = P_f + C @ (Ps_s[t+1] - P_p) @ C.T

    xs = np.array([xs_s[t][0,0] for t in range(T)], dtype=float)
    ys = np.array([xs_s[t][1,0] for t in range(T)], dtype=float)
    return xs, ys


# =========================
# Global shortest path (MCF 1-object equivalent)
# =========================
def run_global_shortest_path(cands_per_t, T, affines, use_cam_motion, W, H):
    # per-frame area reference
    area_ref_t = []
    for t in range(T):
        cs = cands_per_t[t]
        if len(cs) >= 2:
            area_ref_t.append(float(np.median([c["area"] for c in cs])))
        elif len(cs) == 1:
            area_ref_t.append(float(cs[0]["area"]))
        else:
            area_ref_t.append(None)

    def det_node_cost(c, area_ref):
        cost = NODE_BIAS - (W_CONF * c["conf"]) - (W_FG * c["fg"]) + c["top_pen"]
        if USE_SIZE_PRIOR and (area_ref is not None) and area_ref > 0 and c["area"] > 0:
            r = max(c["area"]/area_ref, area_ref/c["area"])
            if r > SIZE_TOL_RATIO:
                cost += W_SIZE * (r - SIZE_TOL_RATIO)
        return float(cost)

    def det_edge_cost(t_from, c_from, t_to, c_to):
        dt = t_to - t_from
        if dt <= 0 or dt > MAX_GAP:
            return None

        if use_cam_motion and dt == 1:
            A = affines[t_to] if t_to < len(affines) else None
            px, py = apply_affine_to_point(A, c_from["cx"], c_from["cy"])
            dx = c_to["cx"] - px
            dy = c_to["cy"] - py
        else:
            dx = c_to["cx"] - c_from["cx"]
            dy = c_to["cy"] - c_from["cy"]

        dist = math.hypot(dx, dy)
        if dist > MAX_LINK_DIST:
            return None

        cost = (W_MOTION * dist) + (GAP_PENALTY * (dt - 1))

        if USE_SIZE_PRIOR and c_from["area"] > 0 and c_to["area"] > 0:
            rr = max(c_from["area"]/c_to["area"], c_to["area"]/c_from["area"])
            if rr > SIZE_TOL_RATIO:
                cost += 0.5 * W_SIZE * (rr - SIZE_TOL_RATIO)

        return float(cost)

    dp_det = [np.full(len(cands_per_t[t]), np.inf, dtype=np.float64) for t in range(T)]
    dp_miss = np.full(T, np.inf, dtype=np.float64)

    prev_det_t = [np.full(len(cands_per_t[t]), -1, dtype=np.int32) for t in range(T)]
    prev_det_j = [np.full(len(cands_per_t[t]), -2, dtype=np.int32) for t in range(T)]
    prev_miss_from_det = np.full(T, -1, dtype=np.int32)

    dp_miss[0] = MISS_PENALTY_PER_FRAME
    for j, c in enumerate(cands_per_t[0]):
        dp_det[0][j] = det_node_cost(c, area_ref_t[0])

    for t in range(1, T):
        best = dp_miss[t-1] + MISS_PENALTY_PER_FRAME
        best_from_det = -1
        if len(cands_per_t[t-1]) > 0:
            jbest = int(np.argmin(dp_det[t-1]))
            v = float(dp_det[t-1][jbest] + MISS_PENALTY_PER_FRAME)
            if v < best:
                best = v
                best_from_det = jbest
        dp_miss[t] = best
        prev_miss_from_det[t] = best_from_det

        for j, cj in enumerate(cands_per_t[t]):
            nodec = det_node_cost(cj, area_ref_t[t])

            best_cost = dp_miss[t-1] + nodec
            best_pt, best_pj = t-1, -1

            for dt in range(1, MAX_GAP+1):
                tp = t - dt
                if tp < 0:
                    break

                vmiss = dp_miss[tp] + nodec
                if vmiss < best_cost:
                    best_cost = vmiss
                    best_pt, best_pj = tp, -1

                for pj, cp in enumerate(cands_per_t[tp]):
                    ec = det_edge_cost(tp, cp, t, cj)
                    if ec is None:
                        continue
                    v = dp_det[tp][pj] + ec + nodec
                    if v < best_cost:
                        best_cost = v
                        best_pt, best_pj = tp, pj

            dp_det[t][j] = best_cost
            prev_det_t[t][j] = best_pt
            prev_det_j[t][j] = best_pj

    end_t = T-1
    end_is_miss = True
    end_j = -1
    end_cost = float(dp_miss[end_t])
    if len(cands_per_t[end_t]) > 0:
        jbest = int(np.argmin(dp_det[end_t]))
        if float(dp_det[end_t][jbest]) < end_cost:
            end_cost = float(dp_det[end_t][jbest])
            end_is_miss = False
            end_j = jbest

    traj = []
    t = end_t
    cur_is_miss = end_is_miss
    cur_j = end_j

    while t >= 0:
        if cur_is_miss:
            traj.append((t, np.nan, np.nan, np.nan, "miss"))
            if t == 0:
                break
            from_det = int(prev_miss_from_det[t])
            t = t - 1
            if from_det >= 0:
                cur_is_miss = False
                cur_j = from_det
            else:
                cur_is_miss = True
                cur_j = -1
        else:
            c = cands_per_t[t][cur_j]
            traj.append((t, c["cx"], c["cy"], c["conf"], c["src"]))
            pt = int(prev_det_t[t][cur_j])
            pj = int(prev_det_j[t][cur_j])
            if pt < 0:
                break
            t = pt
            if pj < 0:
                cur_is_miss = True
                cur_j = -1
            else:
                cur_is_miss = False
                cur_j = pj

    traj = traj[::-1]
    df = pd.DataFrame(traj, columns=["frame_id","cx","cy","conf","source"])
    if len(df) != T:
        base = pd.DataFrame({"frame_id": np.arange(T)})
        df = base.merge(df, on="frame_id", how="left")
        df["source"] = df["source"].fillna("miss")
    return df, end_cost


# =========================
# Main
# =========================
def main():
    if not os.path.isfile(str(VIDEO_PATH)):
        raise RuntimeError(f"VIDEO_PATH not found: {VIDEO_PATH}")
    if not os.path.isfile(str(BEST_PT_PATH)):
        raise RuntimeError(f"BEST_PT_PATH not found: {BEST_PT_PATH}")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_dir = RESULTS_ROOT / run_id
    ensure_dir(run_dir)

    meta_path = run_dir / "run_meta.json"
    csv_path = run_dir / "ball_track.csv"
    overlay_path = run_dir / "ball_overlay.mp4"



    print("[INFO] video:", VIDEO_PATH)
    print("[INFO] best.pt:", BEST_PT_PATH)

    model = YOLO(str(BEST_PT_PATH))

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError("Failed to open video.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    N   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    print("[INFO] fps:", fps, "W,H:", W, H, "frames:", N)

    writer = None
    if SAVE_OVERLAY:
        ofps = fps if OVERLAY_FPS is None else OVERLAY_FPS
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(overlay_path), fourcc, float(ofps), (W, H))
        if not writer.isOpened():
            raise RuntimeError("Failed to open VideoWriter for overlay mp4.")

    # ------------------------------------------------------------
    # A) Candidate extraction (YOLO-only + soft features)
    # ------------------------------------------------------------
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    prev_gray = None
    orb = cv2.ORB_create(nfeatures=ORB_FEATURES) if USE_CAM_MOTION else None
    affines = [None]   # A_{t-1->t}
    frames_gray_cache = []

    raw_cands = []     # per frame list of candidates

    t = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames_gray_cache.append(curr_gray)

        # camera motion A_{t-1->t}
        A = None
        if USE_CAM_MOTION and prev_gray is not None and (t % CAM_MOTION_INTERVAL == 0):
            A, okA = estimate_affine(prev_gray, curr_gray, orb=orb)
            if not okA:
                A = None
        if t > 0:
            affines.append(A)

        # fg mask (motion-compensated)
        fgmask = None
        if USE_FG_SOFT:
            _, fgmask = fg_mask_from_prev(frame, prev_gray, A_prev_to_curr=A)

        prev_gray = curr_gray

        # top soft enable only if green background
        apply_top_soft = False
        if USE_TOP_SOFT:
            gr = green_ratio(frame)
            if gr >= GREEN_RATIO_THRESH:
                apply_top_soft = True

        # YOLO detect
        res = model.predict(
            frame, imgsz=IMGSZ, conf=CONF_THRES, iou=IOU_THRES,
            max_det=MAX_DET, verbose=False
        )[0]

        cands = []
        if res.boxes is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            for (x1,y1,x2,y2), conf in zip(xyxy, confs):
                cx, cy = box_center(x1,y1,x2,y2)
                ar = box_area(x1,y1,x2,y2)

                fg = fg_ratio(fgmask, x1,y1,x2,y2) if USE_FG_SOFT else 0.0
                top_pen = 0.0
                if apply_top_soft and (cy < H * TOP_BAN_RATIO):
                    top_pen = TOP_PENALTY

                cands.append({
                    "cx": float(cx), "cy": float(cy),
                    "conf": float(conf),
                    "area": float(ar),
                    "fg": float(fg),
                    "top_pen": float(top_pen),
                    "src": "yolo"
                })

        raw_cands.append(cands)

        if t % 300 == 0 and t > 0:
            print(f"[CAND] t={t} candidates={len(cands)}")

        t += 1

    T = len(raw_cands)
    cap.release()

    print("[INFO] frames read:", T,
          "avg YOLO K:", float(np.mean([len(x) for x in raw_cands])) if T>0 else 0.0)

    if T <= 2:
        raise RuntimeError("Too few frames.")

    # ------------------------------------------------------------
    # B) 1st pass global optimization (YOLO only)
    # ------------------------------------------------------------
    df1, cost1 = run_global_shortest_path(raw_cands, T, affines, USE_CAM_MOTION, W, H)
    print("[INFO] pass1 cost:", cost1)

    # ------------------------------------------------------------
    # C) Optional: add optical flow candidates (2-pass)
    # ------------------------------------------------------------
    if USE_OPT_FLOW_CAND:
        cands2 = [list(lst) for lst in raw_cands]
        for t in range(1, T):
            px = df1.loc[t-1, "cx"]
            py = df1.loc[t-1, "cy"]
            if np.isnan(px) or np.isnan(py):
                continue

            prevg = frames_gray_cache[t-1]
            currg = frames_gray_cache[t]
            p0 = np.array([[[float(px), float(py)]]], dtype=np.float32)

            p1, st, err = cv2.calcOpticalFlowPyrLK(
                prevImg=prevg, nextImg=currg,
                prevPts=p0, nextPts=None,
                winSize=(FLOW_WIN, FLOW_WIN),
                maxLevel=FLOW_MAX_LEVEL,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
            )
            if p1 is None or st is None or st[0,0] != 1:
                continue

            e = float(err[0,0]) if err is not None else 999.0
            if e > FLOW_ERR_MAX:
                continue

            nx, ny = float(p1[0,0,0]), float(p1[0,0,1])
            if not (0 <= nx < W and 0 <= ny < H):
                continue

            med_area = float(np.median([cc["area"] for cc in cands2[t]])) if len(cands2[t]) > 0 else 400.0

            cands2[t].append({
                "cx": nx, "cy": ny,
                "conf": float(FLOW_CONF),
                "area": float(med_area),
                "fg": 0.0,
                "top_pen": 0.0,
                "src": "flow"
            })

        df2, cost2 = run_global_shortest_path(cands2, T, affines, USE_CAM_MOTION, W, H)
        print("[INFO] pass2 cost:", cost2)
        df_track = df2
    else:
        df_track = df1

    # ------------------------------------------------------------
    # D) Post: gate -> interpolate -> RTS smoother
    # ------------------------------------------------------------
    df_track["cx_i"] = df_track["cx"].interpolate(limit_direction="both")
    df_track["cy_i"] = df_track["cy"].interpolate(limit_direction="both")

    xs = df_track["cx_i"].to_numpy(dtype=float)
    ys = df_track["cy_i"].to_numpy(dtype=float)
    xs, ys = gate_and_interpolate(xs, ys, MAX_JUMP_GATE_PX)

    if USE_RTS_SMOOTHER:
        meas = [(float(x), float(y)) for x, y in zip(xs, ys)]
        xs_s, ys_s = rts_smoother_xy(meas, dt=1.0, q_pos=Q_POS, q_vel=Q_VEL, r_meas=R_MEAS)
        df_track["cx_final"] = xs_s
        df_track["cy_final"] = ys_s
    else:
        df_track["cx_final"] = moving_average_1d(xs, SMOOTH_WINDOW)
        df_track["cy_final"] = moving_average_1d(ys, SMOOTH_WINDOW)

    # ------------------------------------------------------------
    # E) Overlay render
    # ------------------------------------------------------------
    if SAVE_OVERLAY:
        cap2 = cv2.VideoCapture(VIDEO_PATH)
        if not cap2.isOpened():
            raise RuntimeError("Failed to reopen video for overlay rendering.")
        for t in range(T):
            ok, frame = cap2.read()
            if not ok:
                break
            cx = df_track.loc[t, "cx_final"]
            cy = df_track.loc[t, "cy_final"]
            if not (np.isnan(cx) or np.isnan(cy)):
                cv2.circle(frame, (int(cx), int(cy)), 8, (0,255,0), 2)
            writer.write(frame)
        cap2.release()
        writer.release()

    # ------------------------------------------------------------
    # F) Export standardized CSV
    # ------------------------------------------------------------
    df_std = pd.DataFrame({
        "frame_idx": df_track["frame_id"].astype(int),
        "track_id": 0,                     # ✅ ball 단일 트랙
        "object_type": "ball",
        "conf": df_track["conf"].fillna(0.0).astype(float),
        "cx": df_track["cx_final"].astype(float),
        "cy": df_track["cy_final"].astype(float),
        "cx_norm": (df_track["cx_final"] / float(W)).astype(float) if W > 0 else np.nan,
        "cy_norm": (df_track["cy_final"] / float(H)).astype(float) if H > 0 else np.nan,
        "video_w": int(W),
        "video_h": int(H),
    })
    df_std.to_csv(csv_path, index=False)

    # ------------------------------------------------------------
    # G) Meta
    # ------------------------------------------------------------
    meta = {
        "run_id": run_id,
        "video_path": str(VIDEO_PATH),
        "best_pt_path": str(BEST_PT_PATH),
        "fps": float(fps),
        "width": int(W),
        "height": int(H),
        "frames": int(T),
        "outputs": {
            "csv": str(csv_path),
            "overlay_mp4": str(overlay_path) if SAVE_OVERLAY else None
        },
        "csv_schema": "frame_idx,track_id,object_type,conf,cx,cy,cx_norm,cy_norm,video_w,video_h",
        "final_form": {
            "candidate_generation": "YOLO low-conf high-recall + optional optical flow candidates (2-pass)",
            "fg_usage": "soft prior only (cost term), no hard filtering",
            "top_roi_usage": "soft penalty only (cost term), no hard filtering",
            "global_optimization": "MCF(1-object) equivalent shortest path on time-layered DAG with gap edges",
            "post_smoothing": "RTS Kalman smoother"
        },
        "params": {
            "IMGSZ": IMGSZ, "CONF_THRES": CONF_THRES, "IOU_THRES": IOU_THRES, "MAX_DET": MAX_DET,
            "USE_FG_SOFT": USE_FG_SOFT, "DIFF_THRESH": DIFF_THRESH, "W_FG": W_FG,
            "USE_TOP_SOFT": USE_TOP_SOFT, "TOP_BAN_RATIO": TOP_BAN_RATIO, "TOP_PENALTY": TOP_PENALTY,
            "GREEN_RATIO_THRESH": GREEN_RATIO_THRESH,
            "USE_CAM_MOTION": USE_CAM_MOTION, "CAM_MOTION_INTERVAL": CAM_MOTION_INTERVAL,
            "USE_OPT_FLOW_CAND": USE_OPT_FLOW_CAND, "FLOW_WIN": FLOW_WIN, "FLOW_ERR_MAX": FLOW_ERR_MAX,
            "FLOW_CONF": FLOW_CONF,
            "MAX_GAP": MAX_GAP, "MAX_LINK_DIST": MAX_LINK_DIST,
            "NODE_BIAS": NODE_BIAS, "W_CONF": W_CONF, "W_MOTION": W_MOTION, "GAP_PENALTY": GAP_PENALTY,
            "MISS_PENALTY_PER_FRAME": MISS_PENALTY_PER_FRAME,
            "USE_SIZE_PRIOR": USE_SIZE_PRIOR, "SIZE_TOL_RATIO": SIZE_TOL_RATIO, "W_SIZE": W_SIZE,
            "USE_RTS_SMOOTHER": USE_RTS_SMOOTHER, "Q_POS": Q_POS, "Q_VEL": Q_VEL, "R_MEAS": R_MEAS,
            "MAX_JUMP_GATE_PX": MAX_JUMP_GATE_PX,
            "SAVE_OVERLAY": SAVE_OVERLAY
        }
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("[DONE] Local run finished.")
    print(" - CSV   :", csv_path)
    print(" - META  :", meta_path)
    if SAVE_OVERLAY:
        print(" - OVERLAY:", overlay_path)


if __name__ == "__main__":
    main()
