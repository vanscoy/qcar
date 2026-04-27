#!/usr/bin/env python3
# Real QCar route FSM for ground testing
#
# Uses:
# - QCar
# - Front Camera3D for yellow-line following (Y segments)
# - Right Camera2D for white-line following (S and R segments)
# - Encoder distance as guide / backup
# - Left turn behavior for L, with per-segment steer override
#
# Mission model:
#   HUB -> dropoff -> HUB
#
# Keys:
#   ESC / Q : quit
#   R       : re-plan from HUB
#   S       : emergency stop -> IDLE
#   SPACE   : manual advance current segment

from Quanser.product_QCar import QCar
from Quanser.q_essential import Camera3D, Camera2D

import cv2
import time
import math
import enum
import numpy as np
import threading
import os
import sys
from heapq import heappush, heappop

# ==========================================================
# Tunables
# ==========================================================
HEADLESS = False

# Route execution motor commands
SPEED_YELLOW   = 0.078
SPEED_STRAIGHT = 0.070
SPEED_TURN     = 0.065

STEER_CLIP = 0.5
STRAIGHT_STEER = 0.0

DWELL_SECONDS = 2.0

# Left turn fallback default only
TURN_LEFT_STEER   = 0.18
TURN_LEFT_TIME_S  = 1.00   # fallback only if distance is unset

# If yellow follower loses line too long, slow down
LOST_TIMEOUT_S = 1.0
LOST_SPEED_CMD = 0.035

# Encoder / odometry
TICKS_PER_REV = 31844.0
WHEEL_DIAM_M  = 0.066
WHEEL_CIRC_M  = math.pi * WHEEL_DIAM_M

# ==========================================================
# Yellow follower config (front camera)
# ==========================================================
bottom_frac = 0.40
band_frac   = 0.20
MIN_BAND_PTS = 30
MIN_CONTOUR_AREA = 50

target_offset_right = 1000
steering_gain = 0.0012
max_steering_angle = 28.0

FRAME_W, FRAME_H, FRAME_FPS = 1280, 720, 20.0

KERNEL = np.ones((5, 5), np.uint8)

# Grabber watchdogs
MAX_FRAME_AGE_S   = 0.25
STALE_MAX_FRAMES  = 8
MAX_LOOP_TIME_S   = 0.20
RESET_COOLDOWN_S  = 1.0
MAX_ESCALATION    = 3

WINDOW = "QCar Ground FSM"
WIN_MASK = "Yellow Mask"

# ==========================================================
# Right camera / white follower config
# Used for modes: S and R
# ==========================================================
RIGHT_CAM_ID    = "0"
RIGHT_FRAME_W   = 640
RIGHT_FRAME_H   = 480
RIGHT_FRAME_FPS = 30.0

WHITE_TARGET_OFFSET = 50
WHITE_STEERING_GAIN = 0.008
WHITE_STEER_CLIP    = 0.5

WHITE_SPEED_MAX = 0.078
WHITE_SPEED_MIN = 0.072
WHITE_SPEED_KP  = 2.96296e-05

WHITE_THRESH = 150
WIN_RIGHT = "Right Camera View"

# ==========================================================
# Arrival / exit logic tuning
# ==========================================================
# Start looking for visual exits only after this fraction of encoder distance.
EXIT_ARM_FRACTION = 0.80

# Hard fallback if vision never confirms exit.
EXIT_HARD_MAX_FRACTION = 1.20

# Find-yellow confirmation
FIND_YELLOW_CONFIRM_FRAMES = 3
FIND_YELLOW_MIN_AREA = 120

# Yellow-end confirmation
YELLOW_END_CONFIRM_FRAMES = 3

# A and G special case:
# consider yellow "run out" only in a small bottom strip, not whole bottom 40%
RUNOUT_BOTTOM_FRAC = 0.12
RUNOUT_MIN_AREA = 80
RUNOUT_CONFIRM_FRAMES = 3

# ==========================================================
# Exit rule names
# ==========================================================
EXIT_ENCODER = "encoder"
EXIT_YELLOW_END = "yellow_end"
EXIT_FIND_YELLOW = "find_yellow"
EXIT_YELLOW_RUNOUT_BOTTOM = "yellow_runout_bottom"

# ==========================================================
# Map / planner
# Each edge is:
#   (u, v, drive_mode, seg_dist_m, exit_mode, left_steer_override)
#
# Drive mode:
#   Y = front yellow follow
#   S = right white follow
#   R = right white follow
#   L = left turn behavior
#
# Exit rules:
#   EXIT_ENCODER
#   EXIT_YELLOW_END
#   EXIT_FIND_YELLOW
#   EXIT_YELLOW_RUNOUT_BOTTOM
#
# left_steer_override:
#   None for non-left-turn segments
#   float for left-turn segments that need custom steer
# ==========================================================
NODES = {
    "HUB": (0.0, 0.0, 0.0),
    "A":   (0.0, 0.0, 0.0),
    "B":   (0.0, 0.0, 0.0),
    "C_1": (0.0, 0.0, 0.0),
    "C_2": (0.0, 0.0, 0.0),
    "D_1": (0.0, 0.0, 0.0),
    "D_2": (0.0, 0.0, 0.0),
    "E_1": (0.0, 0.0, 0.0),
    "E_2": (0.0, 0.0, 0.0),
    "F_1": (0.0, 0.0, 0.0),
    "F_2": (0.0, 0.0, 0.0),
    "G":   (0.0, 0.0, 0.0),
    "X":   (0.0, 0.0, 0.0),
    "Y":   (0.0, 0.0, 0.0),
}

RAW_EDGES = [
    # Right / white / left segments:
    # most finish when yellow is found
    # exceptions ending at A or G finish when yellow runs out in bottom strip
    ("A",   "C_2", "R", 1.758, EXIT_FIND_YELLOW, None),
    ("A",   "D_2", "L", 2.248, EXIT_FIND_YELLOW, 0.183),

    ("C_2", "Y",   "Y", 2.171, EXIT_ENCODER, None),

    ("C_1", "G",   "L", 2.248, EXIT_YELLOW_RUNOUT_BOTTOM, 0.18),
    ("C_1", "D_2", "S", 2.157, EXIT_FIND_YELLOW, None),

    ("D_2", "X",   "Y", 2.376, EXIT_ENCODER, None),

    ("D_1", "G",   "R", 1.958, EXIT_YELLOW_RUNOUT_BOTTOM, None),
    ("D_1", "C_2", "S", 2.157, EXIT_FIND_YELLOW, None),

    ("G",   "E_2", "R", 1.758, EXIT_FIND_YELLOW, None),
    ("G",   "F_2", "L", 2.248, EXIT_FIND_YELLOW, 0.18),

    ("E_2", "D_1", "Y", 2.714, EXIT_YELLOW_END, None),

    ("E_1", "F_2", "S", 2.757, EXIT_FIND_YELLOW, None),
    ("E_1", "A",   "L", 2.248, EXIT_YELLOW_RUNOUT_BOTTOM, 0.22),

    ("F_1", "A",   "R", 1.758, EXIT_YELLOW_RUNOUT_BOTTOM, None),
    ("F_1", "E_2", "S", 2.157, EXIT_FIND_YELLOW, None),

    ("F_2", "B",   "Y", 1.038, EXIT_ENCODER, None),

    ("B",   "HUB", "Y", 3.534, EXIT_ENCODER, None),
    ("HUB", "C_1", "Y", 0.810, EXIT_YELLOW_END, None),

    ("X",   "E_1", "Y", 1.740, EXIT_YELLOW_END, None),
    ("Y",   "F_1", "Y", 2.063, EXIT_YELLOW_END, None),
]

# ==========================================================
# FSM
# ==========================================================
class State(enum.Enum):
    IDLE = 0
    ROUTE = 1
    YELLOW_FOLLOW = 2
    WHITE_FOLLOW = 3
    TURN_LEFT = 4
    TURN_RIGHT = 5
    STOP_WAIT = 6

# ==========================================================
# Camera helpers
# ==========================================================
def safe_terminate_camera3d(cam):
    if cam is None:
        return
    for name in ("terminate_RGB", "stop_RGB", "stop_rgb", "stop"):
        if hasattr(cam, name):
            try:
                getattr(cam, name)()
            except Exception:
                pass
    try:
        cam.terminate()
    except AttributeError as e:
        if "video3d" not in str(e):
            raise
    except Exception:
        pass

def safe_terminate_camera2d(cam):
    if cam is None:
        return
    for name in ("terminate", "stop"):
        if hasattr(cam, name):
            try:
                getattr(cam, name)()
            except Exception:
                pass

class FrameGrabber:
    def __init__(self):
        self.cam = None
        self.lock = threading.Lock()
        self.frame = None
        self.running = False
        self.last_ck = None
        self.stale_count = 0
        self.last_good_t = time.time()
        self.last_reset_attempt = 0.0
        self.thread = None

    def _open_cam(self):
        self.cam = Camera3D(
            mode='RGB',
            frame_width_RGB=FRAME_W,
            frame_height_RGB=FRAME_H,
            frame_rate_RGB=FRAME_FPS,
            device_id='0'
        )

    def start(self):
        self._open_cam()
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        try:
            if self.thread:
                self.thread.join(timeout=0.5)
        except Exception:
            pass
        try:
            safe_terminate_camera3d(self.cam)
        except Exception:
            pass
        self.cam = None

    def _checksum(self, img):
        thumb = cv2.resize(img, (64, 36), interpolation=cv2.INTER_AREA)
        return hash(thumb.tobytes())

    def _soft_reset_ok(self):
        return (time.time() - self.last_reset_attempt) >= RESET_COOLDOWN_S

    def _soft_reset(self):
        if not self._soft_reset_ok():
            return False
        self.last_reset_attempt = time.time()
        try:
            safe_terminate_camera3d(self.cam)
        except Exception:
            pass
        try:
            self._open_cam()
            self.last_ck = None
            self.stale_count = 0
            self.last_good_t = time.time()
            return True
        except Exception:
            return False

    def _loop(self):
        while self.running:
            try:
                self.cam.read_RGB()
                img = self.cam.image_buffer_RGB
                if img is not None and img.size != 0:
                    ck = self._checksum(img)
                    if self.last_ck is not None and ck == self.last_ck:
                        self.stale_count += 1
                    else:
                        self.stale_count = 0
                        self.last_good_t = time.time()
                    self.last_ck = ck
                    with self.lock:
                        self.frame = img.copy()
            except Exception:
                pass
            time.sleep(0.001)

    def get_frame(self):
        too_old = (time.time() - self.last_good_t) > MAX_FRAME_AGE_S
        too_stale = self.stale_count >= STALE_MAX_FRAMES
        if too_old or too_stale:
            self._soft_reset()
        with self.lock:
            return None if self.frame is None else self.frame.copy()

# ==========================================================
# QCar helpers
# ==========================================================
def neutral_brake(car):
    try:
        car.read_write_std(
            np.array([0.0, 0.0], dtype=np.float64),
            np.array([1, 0, 0, 0, 1, 0, 0, 0], dtype=np.float64)
        )
    except Exception:
        pass

def drive(car, speed_cmd, steering_cmd, brake=False):
    leds = np.array(
        [1, 0, 0, 0, 1, 0, 1 if not brake else 0, 1 if not brake else 0],
        dtype=np.float64
    )
    if brake:
        leds = np.array([1, 0, 0, 0, 1, 0, 0, 0], dtype=np.float64)

    try:
        car.read_write_std(
            np.array([float(speed_cmd), float(steering_cmd)], dtype=np.float64),
            leds
        )
    except Exception:
        pass

def reopen_qcar(car):
    try:
        car.terminate()
    except Exception:
        pass
    try:
        return QCar()
    except Exception:
        return None

def read_ticks(qcar) -> float:
    return float(qcar.read_encoder())

# ==========================================================
# Encoder odometry
# ==========================================================
class SegmentOdom:
    def __init__(self):
        self.prev_ticks = None
        self.segment_dist_m = 0.0
        self.total_dist_m = 0.0

    def reset_all(self, qcar):
        self.prev_ticks = read_ticks(qcar)
        self.segment_dist_m = 0.0
        self.total_dist_m = 0.0

    def reset_segment(self, qcar):
        self.prev_ticks = read_ticks(qcar)
        self.segment_dist_m = 0.0

    def update(self, qcar):
        ticks_now = read_ticks(qcar)

        if self.prev_ticks is None:
            self.prev_ticks = ticks_now
            return 0.0, self.segment_dist_m, self.total_dist_m

        d_ticks = ticks_now - self.prev_ticks
        self.prev_ticks = ticks_now

        d_m = (d_ticks / TICKS_PER_REV) * WHEEL_CIRC_M

        self.segment_dist_m += abs(d_m)
        self.total_dist_m += abs(d_m)

        return d_m, self.segment_dist_m, self.total_dist_m

# ==========================================================
# Planner
# ==========================================================
def build_graph(raw_edges):
    nodes = set()
    for u, v, _mode, _segd, _exit_mode, _steer in raw_edges:
        nodes.add(u)
        nodes.add(v)

    graph = {n: [] for n in nodes}
    for u, v, mode, seg_dist_m, exit_mode, steer_override in raw_edges:
        w = 1.0
        graph[u].append((v, w, mode, seg_dist_m, exit_mode, steer_override))
    return graph

def dijkstra(graph, start, goal):
    pq = [(0.0, start)]
    best = {start: 0.0}
    parent = {start: None}
    parent_edge = {start: None}

    while pq:
        g, u = heappop(pq)
        if g != best.get(u, float("inf")):
            continue
        if u == goal:
            break

        for v, w, mode, seg_dist_m, exit_mode, steer_override in graph.get(u, []):
            ng = g + w
            if ng < best.get(v, float("inf")):
                best[v] = ng
                parent[v] = u
                parent_edge[v] = (u, v, mode, seg_dist_m, exit_mode, steer_override, w)
                heappush(pq, (ng, v))

    if goal not in best:
        return None, float("inf"), None

    path, steps = [], []
    cur = goal
    while cur is not None:
        path.append(cur)
        e = parent_edge[cur]
        if e is not None:
            steps.append(e)
        cur = parent[cur]

    path.reverse()
    steps.reverse()
    return path, best[goal], steps

def plan_full_route(graph, dropoff):
    legs = [("HUB", dropoff), (dropoff, "HUB")]

    full_nodes = []
    full_modes = []
    full_dists = []
    full_exits = []
    full_left_steers = []

    for i, (s, t) in enumerate(legs):
        path, _cost, steps = dijkstra(graph, s, t)
        if path is None:
            raise RuntimeError(f"No path from {s} to {t}")

        if i == 0:
            full_nodes += path
        else:
            full_nodes += path[1:]

        for (_u, _v, mode, seg_dist_m, exit_mode, steer_override, _w) in steps:
            full_modes.append(mode)
            full_dists.append(seg_dist_m)
            full_exits.append(exit_mode)
            full_left_steers.append(steer_override)

    return full_nodes, full_modes, full_dists, full_exits, full_left_steers

def ask_dropoff(graph):
    valid = sorted(graph.keys())
    print("\nAvailable nodes:")
    print(" ".join(valid))
    d = input("\nDropoff node (example: B): ").strip().upper()
    if d not in graph or d == "HUB":
        print("Invalid dropoff. Using B.")
        d = "B"
    return d

# ==========================================================
# Yellow perception
# ==========================================================
def make_yellow_mask(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower1 = (15, 90, 80)
    upper1 = (45, 255, 255)
    lower2 = (15, 40, 60)
    upper2 = (45, 255, 200)

    m1 = cv2.inRange(hsv, lower1, upper1)
    m2 = cv2.inRange(hsv, lower2, upper2)
    mask_hsv = cv2.bitwise_or(m1, m2)

    white_glare = cv2.inRange(hsv, (0, 0, 220), (180, 60, 255))
    mask = cv2.bitwise_and(mask_hsv, cv2.bitwise_not(white_glare))

    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    _, b_bin = cv2.threshold(lab[:, :, 2], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.bitwise_and(mask, b_bin)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL, iterations=1)
    return mask

def get_line_info_bottom(image):
    h, w, _ = image.shape
    y0 = int(h * (1.0 - bottom_frac))
    roi = image[y0:h, 0:w]

    mask = make_yellow_mask(roi)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, mask

    largest = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(largest) < MIN_CONTOUR_AREA:
        return None, mask

    pts = largest.reshape(-1, 2)
    roi_h = roi.shape[0]
    band_y_start = int(roi_h * (1.0 - band_frac))
    band_pts = pts[pts[:, 1] >= band_y_start]

    if band_pts.shape[0] >= MIN_BAND_PTS:
        cx, cy = int(band_pts[:, 0].mean()), int(band_pts[:, 1].mean())
    else:
        N = min(50, pts.shape[0])
        idx = np.argsort(pts[:, 1])[-N:]
        sel = pts[idx]
        cx, cy = int(sel[:, 0].mean()), int(sel[:, 1].mean())

    contour_full = largest + np.array([0, y0])
    centroid_full = (cx, y0 + cy)

    return {
        "contour": contour_full,
        "centroid": centroid_full,
        "cx_full": cx,
        "roi_origin": (0, y0),
        "roi_size": (w, h - y0),
        "band_y_start_full": y0 + band_y_start,
        "area": cv2.contourArea(largest),
    }, mask

def yellow_follow_step(img):
    info, mask = get_line_info_bottom(img)
    h, w, _ = img.shape
    display = img.copy()

    steering = 0.0
    found = False
    status = "NO YELLOW DETECTED"

    if info is not None:
        found = True
        desired_x = w - target_offset_right
        error = desired_x - info["cx_full"]
        steering = float(np.clip(error * steering_gain, -STEER_CLIP, STEER_CLIP))
        status = f"FOUND | err={error:+.1f} steer={steering:+.3f}"

        if not HEADLESS:
            (rx0, ry0) = info["roi_origin"]
            (rw, rh) = info["roi_size"]
            cv2.rectangle(display, (rx0, ry0), (rx0 + rw - 1, ry0 + rh - 1), (0, 255, 0), 2)
            cv2.drawContours(display, [info["contour"]], -1, (255, 0, 0), 2)
            cv2.circle(display, info["centroid"], 8, (255, 0, 0), -1)
            cv2.circle(display, (desired_x, ry0 + rh // 2), 8, (0, 0, 255), -1)
            cv2.rectangle(display,
                          (rx0, info["band_y_start_full"]),
                          (rx0 + rw - 1, ry0 + rh - 1),
                          (0, 200, 200), 2)

    return steering, display, mask, status, found, info

# ==========================================================
# Yellow detection helpers used by exit logic
# ==========================================================
def get_front_yellow_area(image):
    info, _mask = get_line_info_bottom(image)
    if info is None:
        return 0.0
    return float(info.get("area", 0.0))

def get_bottom_strip_yellow_area(image):
    h, w, _ = image.shape
    strip_h = max(1, int(h * RUNOUT_BOTTOM_FRAC))
    roi = image[h - strip_h:h, 0:w]
    mask = make_yellow_mask(roi)
    return float(cv2.countNonZero(mask)), mask

# ==========================================================
# White-line perception (right camera)
# Used for modes S and R
# ==========================================================
def get_right_line_offset(image):
    h, w, _ = image.shape

    crop_x = int(w * 0.2)
    crop_y = h // 2
    crop = image[crop_y:h, crop_x:w]

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, WHITE_THRESH, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, thresh

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 50:
        return None, thresh

    M = cv2.moments(largest)
    if M["m00"] <= 0:
        return None, thresh

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    contour_full = largest + np.array([crop_x, crop_y])
    centroid_full = (crop_x + cx, crop_y + cy)

    info = {
        "contour": contour_full,
        "centroid": centroid_full,
        "crop_x": crop_x,
        "crop_y": crop_y,
        "crop_w": w - crop_x,
        "crop_h": h - crop_y,
    }
    return info, thresh

def white_follow_step(img):
    info, thresh = get_right_line_offset(img)
    h, w, _ = img.shape
    display = img.copy()

    steering = 0.0
    speed_cmd = WHITE_SPEED_MIN
    found = False
    status = "NO WHITE DETECTED"

    crop_x = int(w * 0.2)
    crop_y = h // 2
    crop_w = w - crop_x
    crop_h = h - crop_y

    if not HEADLESS:
        cv2.rectangle(display,
                      (crop_x, crop_y),
                      (crop_x + crop_w - 1, crop_y + crop_h - 1),
                      (0, 255, 255), 2)
        cv2.line(display, (crop_x, crop_y), (crop_x, crop_y + crop_h - 1), (0, 0, 255), 2)
        cv2.line(display, (crop_x + crop_w - 1, crop_y),
                 (crop_x + crop_w - 1, crop_y + crop_h - 1), (0, 255, 0), 2)

    if info is not None:
        found = True

        centroid_x, centroid_y = info["centroid"]

        # Preserving your working control logic
        target_x = int(w * 0.5) + WHITE_TARGET_OFFSET
        target_y = h // 2 + (h // 4) + 15

        dy = int(centroid_y) - int(target_y)
        steering = float(np.clip(dy * WHITE_STEERING_GAIN, -WHITE_STEER_CLIP, WHITE_STEER_CLIP))

        prop = abs(target_y - int(centroid_y)) + 1
        speed_cmd = float(np.clip(WHITE_SPEED_MAX - (WHITE_SPEED_KP * float(prop)),
                                  WHITE_SPEED_MIN, WHITE_SPEED_MAX))

        status = f"FOUND | dy={dy:+d} steer={steering:+.3f} speed={speed_cmd:.3f}"

        if not HEADLESS:
            cv2.drawContours(display, [info["contour"]], -1, (255, 0, 0), 2)
            cv2.circle(display, info["centroid"], 10, (255, 0, 0), -1)
            cv2.circle(display, (target_x, target_y), 10, (0, 0, 255), -1)

            cv2.putText(display, f"dy: {dy:+d}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            cv2.putText(display, f"steer: {steering:+.3f}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(display, f"speed: {speed_cmd:.3f}",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    return steering, speed_cmd, display, thresh, status, found

# ==========================================================
# FSM helpers
# ==========================================================
def enter_next_motion_state(mode, odom, qcar):
    odom.reset_segment(qcar)

    if mode == "Y":
        return State.YELLOW_FOLLOW

    if mode in ("S", "R"):
        return State.WHITE_FOLLOW

    if mode == "L":
        return State.TURN_LEFT

    raise ValueError(f"Unknown mode '{mode}'")

def advance_after_segment(step, route_nodes, dwell_nodes):
    step += 1
    if step >= len(route_nodes) - 1:
        return step, State.IDLE, None
    if route_nodes[step] in dwell_nodes:
        return step, State.STOP_WAIT, time.time() + DWELL_SECONDS
    return step, State.ROUTE, None

def armed_for_visual_exit(seg_dist_m, cur_dist_target):
    if cur_dist_target <= 0.0:
        return True
    return seg_dist_m >= (EXIT_ARM_FRACTION * cur_dist_target)

def past_hard_max(seg_dist_m, cur_dist_target):
    if cur_dist_target <= 0.0:
        return False
    return seg_dist_m >= (EXIT_HARD_MAX_FRACTION * cur_dist_target)

# ==========================================================
# Main
# ==========================================================
def main():
    frame_count = 0
    fps = 0
    last_time = time.time()

    myCar = QCar()

    grab = FrameGrabber()
    grab.start()

    rightCam = Camera2D(
        camera_id=RIGHT_CAM_ID,
        frame_width=RIGHT_FRAME_W,
        frame_height=RIGHT_FRAME_H,
        frame_rate=RIGHT_FRAME_FPS
    )

    odom = SegmentOdom()
    odom.reset_all(myCar)

    graph = build_graph(RAW_EDGES)
    dropoff = ask_dropoff(graph)
    route_nodes, route_modes, route_dists, route_exits, route_left_steers = plan_full_route(graph, dropoff)
    dwell_nodes = {dropoff}

    state = State.ROUTE
    step = 0
    wait_until = None
    lost_since = None
    turn_started_t = None
    manual_advance_requested = False

    # Exit-confirmation counters
    find_yellow_count = 0
    yellow_end_count = 0
    yellow_runout_count = 0

    if not HEADLESS:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.namedWindow(WIN_MASK, cv2.WINDOW_NORMAL)
        cv2.namedWindow(WIN_RIGHT, cv2.WINDOW_NORMAL)

        cv2.resizeWindow(WINDOW, 1280, 720)
        cv2.resizeWindow(WIN_MASK, 800, 240)
        cv2.resizeWindow(WIN_RIGHT, 1280, 960)

        cv2.moveWindow(WINDOW, 20, 20)
        cv2.moveWindow(WIN_MASK, 20, 780)
        cv2.moveWindow(WIN_RIGHT, 1320, 20)

    print("\nKeys: ESC/Q quit | R re-plan | S stop(IDLE) | SPACE advance")
    print("Mission:", " -> ".join(route_nodes))
    print("Modes:", route_modes)
    print("Dists:", route_dists)
    print("Exits:", route_exits)
    print("Left steers:", route_left_steers)

    try:
        escalation = 0

        while True:
            cycle_start = time.time()
            now = time.time()

            _d_step, seg_dist_m, total_dist_m = odom.update(myCar)

            img = grab.get_frame()

            right_img = None
            right_overlay = np.zeros((RIGHT_FRAME_H, RIGHT_FRAME_W, 3), dtype=np.uint8)
            right_thresh = np.zeros((RIGHT_FRAME_H // 2, int(RIGHT_FRAME_W * 0.8)), dtype=np.uint8)

            try:
                rightCam.read()
                if rightCam.image_data is not None and rightCam.image_data.size != 0:
                    right_img = rightCam.image_data.copy()
                    right_overlay = right_img.copy()
            except Exception:
                right_img = None

            if img is None:
                neutral_brake(myCar)
                if (time.time() - grab.last_good_t) > 2.0:
                    escalation += 1

                if escalation >= MAX_ESCALATION:
                    newCar = reopen_qcar(myCar)
                    if newCar is None:
                        os.execv(sys.executable, [sys.executable] + sys.argv)
                    else:
                        myCar = newCar
                        odom.reset_all(myCar)
                    escalation = 0

                time.sleep(0.02)
                continue
            else:
                escalation = 0

            overlay = img.copy()
            mask_show = np.zeros((100, 300), dtype=np.uint8)
            forward_cmd = 0.0
            steering_cmd = 0.0
            braking = False

            frame_count += 1
            if now - last_time >= 1.0:
                fps = frame_count
                frame_count = 0
                last_time = now

            cur_src = route_nodes[step] if step < len(route_nodes) else "?"
            cur_dst = route_nodes[step + 1] if step + 1 < len(route_nodes) else "?"
            cur_mode = route_modes[step] if step < len(route_modes) else "-"
            cur_dist_target = route_dists[step] if step < len(route_dists) else 0.0
            cur_exit_mode = route_exits[step] if step < len(route_exits) else "-"
            cur_left_steer = route_left_steers[step] if step < len(route_left_steers) else None

            manual_advance = manual_advance_requested
            manual_advance_requested = False

            # reset exit counters when switching to dispatch state
            if state == State.ROUTE:
                find_yellow_count = 0
                yellow_end_count = 0
                yellow_runout_count = 0

            if state == State.IDLE:
                braking = True
                cv2.putText(overlay, "STATE: IDLE (Press R to re-plan)", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            elif state == State.ROUTE:
                if step >= len(route_nodes) - 1:
                    state = State.IDLE
                else:
                    cur_src = route_nodes[step]
                    cur_dst = route_nodes[step + 1]
                    cur_mode = route_modes[step]
                    cur_dist_target = route_dists[step]
                    cur_exit_mode = route_exits[step]
                    cur_left_steer = route_left_steers[step]

                    state = enter_next_motion_state(cur_mode, odom, myCar)
                    lost_since = None
                    turn_started_t = None

                    cv2.putText(overlay, f"STATE: ROUTE | Dispatch {cur_mode} {cur_src}->{cur_dst} | exit={cur_exit_mode}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            elif state == State.YELLOW_FOLLOW:
                t0 = time.time()
                steering_cmd, overlay, mask_show, status, found, yellow_info = yellow_follow_step(img)
                calc_ms = (time.time() - t0) * 1000.0

                if found:
                    lost_since = None
                    forward_cmd = SPEED_YELLOW
                else:
                    if lost_since is None:
                        lost_since = now
                    if (now - lost_since) >= LOST_TIMEOUT_S:
                        forward_cmd = LOST_SPEED_CMD
                        steering_cmd = 0.0
                    else:
                        forward_cmd = SPEED_YELLOW * 0.5
                        steering_cmd = 0.0

                arrived = False
                reason = ""

                if cur_exit_mode == EXIT_ENCODER:
                    if cur_dist_target > 0.0:
                        arrived = seg_dist_m >= cur_dist_target
                        reason = "encoder"
                elif cur_exit_mode == EXIT_YELLOW_END:
                    if armed_for_visual_exit(seg_dist_m, cur_dist_target):
                        if not found:
                            yellow_end_count += 1
                        else:
                            yellow_end_count = 0

                        if yellow_end_count >= YELLOW_END_CONFIRM_FRAMES:
                            arrived = True
                            reason = "yellow_end"

                    if past_hard_max(seg_dist_m, cur_dist_target):
                        arrived = True
                        reason = "hard_max"

                if manual_advance:
                    arrived = True
                    reason = "manual"

                cv2.putText(overlay, f"STATE: YELLOW_FOLLOW | {cur_src}->{cur_dst} | {status}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(overlay, f"exit={cur_exit_mode} seg: {seg_dist_m:.3f}/{cur_dist_target:.3f} m | total: {total_dist_m:.3f} m",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(overlay, f"FPS:{fps} Calc:{calc_ms:.1f}ms steer:{steering_cmd:+.3f}",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(overlay, f"yellow_end_count={yellow_end_count}/{YELLOW_END_CONFIRM_FRAMES} reason={reason}",
                            (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

                if arrived:
                    step, state, wait_until = advance_after_segment(step, route_nodes, dwell_nodes)

            elif state == State.WHITE_FOLLOW:
                t0 = time.time()

                if right_img is None:
                    steering_cmd = 0.0
                    forward_cmd = 0.0
                    white_found = False
                    status = "RIGHT CAMERA MISSING"
                    calc_ms = (time.time() - t0) * 1000.0
                    braking = True
                else:
                    steering_cmd, forward_cmd, right_overlay, right_thresh, status, white_found = white_follow_step(right_img)
                    calc_ms = (time.time() - t0) * 1000.0

                    if not white_found:
                        forward_cmd = WHITE_SPEED_MIN
                        steering_cmd = 0.0

                arrived = False
                reason = ""

                front_yellow_area = get_front_yellow_area(img)
                front_yellow_found = front_yellow_area >= FIND_YELLOW_MIN_AREA

                if cur_exit_mode == EXIT_FIND_YELLOW:
                    if armed_for_visual_exit(seg_dist_m, cur_dist_target):
                        if front_yellow_found:
                            find_yellow_count += 1
                        else:
                            find_yellow_count = 0

                        if find_yellow_count >= FIND_YELLOW_CONFIRM_FRAMES:
                            arrived = True
                            reason = "find_yellow"

                    if past_hard_max(seg_dist_m, cur_dist_target):
                        arrived = True
                        reason = "hard_max"

                elif cur_exit_mode == EXIT_ENCODER:
                    if cur_dist_target > 0.0:
                        arrived = seg_dist_m >= cur_dist_target
                        reason = "encoder"

                if manual_advance:
                    arrived = True
                    reason = "manual"

                cv2.putText(overlay, f"STATE: WHITE_FOLLOW | {cur_src}->{cur_dst} | {status}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(overlay, f"exit={cur_exit_mode} seg: {seg_dist_m:.3f}/{cur_dist_target:.3f} m | total: {total_dist_m:.3f} m",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(overlay, f"FPS:{fps} Calc:{calc_ms:.1f}ms steer:{steering_cmd:+.3f} speed:{forward_cmd:.3f}",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(overlay, f"front_yellow_area={front_yellow_area:.0f} count={find_yellow_count}/{FIND_YELLOW_CONFIRM_FRAMES} reason={reason}",
                            (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

                if arrived:
                    step, state, wait_until = advance_after_segment(step, route_nodes, dwell_nodes)

            elif state == State.TURN_LEFT:
                if turn_started_t is None:
                    turn_started_t = now

                dt = now - turn_started_t
                forward_cmd = SPEED_TURN
                steering_cmd = cur_left_steer if cur_left_steer is not None else TURN_LEFT_STEER

                arrived = False
                reason = ""

                front_yellow_area = get_front_yellow_area(img)
                front_yellow_found = front_yellow_area >= FIND_YELLOW_MIN_AREA

                bottom_yellow_area, _bottom_mask = get_bottom_strip_yellow_area(img)
                bottom_yellow_present = bottom_yellow_area >= RUNOUT_MIN_AREA

                if cur_exit_mode == EXIT_FIND_YELLOW:
                    if armed_for_visual_exit(seg_dist_m, cur_dist_target):
                        if front_yellow_found:
                            find_yellow_count += 1
                        else:
                            find_yellow_count = 0

                        if find_yellow_count >= FIND_YELLOW_CONFIRM_FRAMES:
                            arrived = True
                            reason = "find_yellow"

                    if past_hard_max(seg_dist_m, cur_dist_target):
                        arrived = True
                        reason = "hard_max"

                elif cur_exit_mode == EXIT_YELLOW_RUNOUT_BOTTOM:
                    if armed_for_visual_exit(seg_dist_m, cur_dist_target):
                        if not bottom_yellow_present:
                            yellow_runout_count += 1
                        else:
                            yellow_runout_count = 0

                        if yellow_runout_count >= RUNOUT_CONFIRM_FRAMES:
                            arrived = True
                            reason = "yellow_runout_bottom"

                    if past_hard_max(seg_dist_m, cur_dist_target):
                        arrived = True
                        reason = "hard_max"

                elif cur_exit_mode == EXIT_ENCODER:
                    if cur_dist_target > 0.0:
                        arrived = seg_dist_m >= cur_dist_target
                        reason = "encoder"

                else:
                    if cur_dist_target > 0.0:
                        arrived = seg_dist_m >= cur_dist_target
                        reason = "encoder_default"
                    else:
                        arrived = dt >= TURN_LEFT_TIME_S
                        reason = "timer_fallback"

                if manual_advance:
                    arrived = True
                    reason = "manual"

                cv2.putText(overlay, f"STATE: TURN_LEFT | {cur_src}->{cur_dst}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(overlay, f"steer={steering_cmd:+.3f} exit={cur_exit_mode} seg:{seg_dist_m:.3f}/{cur_dist_target:.3f}m dt:{dt:.2f}s",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(overlay, f"front_yellow={front_yellow_area:.0f} find_count={find_yellow_count}/{FIND_YELLOW_CONFIRM_FRAMES}",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(overlay, f"bottom_yellow={bottom_yellow_area:.0f} runout_count={yellow_runout_count}/{RUNOUT_CONFIRM_FRAMES} reason={reason}",
                            (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

                if arrived:
                    step, state, wait_until = advance_after_segment(step, route_nodes, dwell_nodes)
                    turn_started_t = None

            elif state == State.TURN_RIGHT:
                braking = True
                cv2.putText(overlay, "STATE: TURN_RIGHT (unused)", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            elif state == State.STOP_WAIT:
                braking = True
                remaining = max(0.0, (wait_until - now) if wait_until is not None else 0.0)

                cv2.putText(overlay, f"STATE: STOP_WAIT at {route_nodes[step]} | {remaining:.1f}s left",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                if wait_until is not None and now >= wait_until:
                    wait_until = None
                    state = State.ROUTE

            mission_str = " -> ".join(route_nodes) if route_nodes else "(no route)"
            cv2.putText(overlay, f"Mission: {mission_str}",
                        (10, overlay.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

            cycle_dt = time.time() - cycle_start
            try:
                if cycle_dt > MAX_LOOP_TIME_S:
                    neutral_brake(myCar)
                else:
                    drive(myCar,
                          speed_cmd=forward_cmd if not braking else 0.0,
                          steering_cmd=np.clip(steering_cmd, -STEER_CLIP, STEER_CLIP),
                          brake=braking)
            except Exception:
                neutral_brake(myCar)
                newCar = reopen_qcar(myCar)
                if newCar is None:
                    escalation += 1
                    if escalation >= MAX_ESCALATION:
                        os.execv(sys.executable, [sys.executable] + sys.argv)
                else:
                    myCar = newCar
                    odom.reset_all(myCar)
                    escalation = 0

            if not HEADLESS:
                cv2.imshow(WINDOW, overlay)
                cv2.imshow(WIN_MASK, mask_show)
                cv2.imshow(WIN_RIGHT, right_overlay)

                k = cv2.waitKey(1) & 0xFF
                if k in (27, ord('q'), ord('Q')):
                    print("Quit requested.")
                    break
                elif k in (ord('s'), ord('S')):
                    state = State.IDLE
                    route_nodes = []
                    route_modes = []
                    route_dists = []
                    route_exits = []
                    route_left_steers = []
                    step = 0
                    wait_until = None
                    lost_since = None
                    turn_started_t = None
                    find_yellow_count = 0
                    yellow_end_count = 0
                    yellow_runout_count = 0
                    neutral_brake(myCar)
                    print("Stopped. (IDLE)")
                elif k in (ord('r'), ord('R')):
                    dropoff = ask_dropoff(graph)
                    route_nodes, route_modes, route_dists, route_exits, route_left_steers = plan_full_route(graph, dropoff)
                    dwell_nodes = {dropoff}
                    step = 0
                    wait_until = None
                    lost_since = None
                    turn_started_t = None
                    find_yellow_count = 0
                    yellow_end_count = 0
                    yellow_runout_count = 0
                    state = State.ROUTE
                    odom.reset_all(myCar)
                    print("Mission:", " -> ".join(route_nodes))
                    print("Modes:", route_modes)
                    print("Dists:", route_dists)
                    print("Exits:", route_exits)
                    print("Left steers:", route_left_steers)
                elif k == 32:
                    manual_advance_requested = True
                    print("[Advance] manual segment advance requested")
            else:
                time.sleep(0.01)

    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        try:
            grab.stop()
        except Exception:
            pass
        try:
            safe_terminate_camera2d(rightCam)
        except Exception:
            pass
        try:
            neutral_brake(myCar)
        except Exception:
            pass
        try:
            myCar.terminate()
        except Exception:
            pass

if __name__ == "__main__":
    main()