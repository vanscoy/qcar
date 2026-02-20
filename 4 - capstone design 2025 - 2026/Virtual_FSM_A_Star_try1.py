# qlabs_fsm_right_virtual.py
# QLabs QCar2 FSM (VIRTUAL):
# - DRIVE: front yellow-line follow using VIRTUAL HSV thresholds (like your working overlay script)
# - STOPPING: timed stop after stop-sign detect (RGB-only in QLabs)
# - TURN_RIGHT: minimal right-turn logic using RIGHT CSI cam (your logic preserved)
#
# Keys:
#   ESC / Q : quit
#   T       : toggle SOURCE_IS_RGB (RGB<->BGR fix)
#   V       : toggle RightCam preview when not turning
#   R       : toggle ROI for front (debug)
#   ]/[     : min contour area for front line (debug)
#   P       : manual exit TURN_RIGHT -> DRIVE

import time
import math
import enum
import numpy as np
import cv2

from qvl.qlabs import QuanserInteractiveLabs
from qvl.qcar2 import QLabsQCar2
from heapq import heappush, heappop

# ==========================================================
# A* (local) + map edges (scaled meters)
# ==========================================================
EDGE_SCALE = 14.1667

EDGES_UNITS = [
    ("HUB", "A", 7.0),
    ("A", "B", 4.0),
    ("A", "K", 3.0),
    ("A", "J", 7.5),
    ("B", "F", 2.0),
    ("B", "I", 2.8),
    ("B", "J", 2.0),
    ("B", "A", 4.0),
    ("C", "D", 2.0),
    ("C", "G", 7.5),
    ("D", "E", 2.0),
    ("D", "F", 4.2),
    ("E", "C", 2.0),
    ("E", "K", 3.0),
    ("F", "D", 4.2),
    ("F", "B", 2.0),
    ("F", "J", 2.8),
    ("F", "I", 2.0),
    ("G", "L", 1.0),
    ("G", "H", 3.5),
    ("H", "HUB", 5.5),
    ("H", "I", 3.0),
    ("H", "G", 3.5),
    ("I", "F", 2.0),
    ("I", "B", 2.8),
    ("I", "J", 2.0),
    ("I", "H", 3.0),
    ("J", "A", 7.5),
    ("J", "B", 2.0),
    ("J", "F", 2.8),
    ("J", "I", 2.0),
    ("K", "E", 3.0),
    ("K", "A", 3.0),
    ("K", "B", 4.0),
    ("L", "G", 1.0),
]

EDGES = [(u, v, w * EDGE_SCALE) for u, v, w in EDGES_UNITS]

TURN_RESTRICTIONS = {
    "A": {
        "HUB": ["B", "K"],
        "J": ["B", "K"],
        "K": ["J"],
        "B": ["J"],
    },
    "B": {
        "A": ["F", "I", "J"],
        "K": ["F", "I", "J"],
        "J": ["K", "A"],
        "F": ["A", "K"],
        "I": ["A", "K"],
    },
    "C": {
        "E": ["D", "G"],
        "D": ["G"],
    },
    "D": {
        "C": ["E", "F"],
        "F": ["E"],
    },
    "E": {
        "D": ["C", "K"],
        "K": ["C"],
    },
    "F": {
        "D": ["B", "J", "I"],
        "B": ["D"],
        "J": ["D"],
        "I": ["D"],
    },
    "G": {
        "L": ["H", "C"],
        "H": ["L", "C"],
        "C": ["L", "H"],
    },
    "H": {
        "G": ["HUB", "I"],
        "I": ["HUB", "G"],
    },
    "I": {
        "F": ["H"],
        "B": ["H"],
        "J": ["H"],
        "H": ["F", "B", "J"],
    },
    "J": {
        "A": ["B", "F", "I"],
        "B": ["A"],
        "F": ["A"],
        "I": ["A"],
    },
    "K": {
        "E": ["A", "B"],
        "A": ["E"],
        "B": ["E"],
    },
    "L": {
        "G": ["G"],
    },
    "HUB": {
        "H": ["A"],
    },
}


def build_graph(edges):
    graph = {}
    for u, v, w in edges:
        graph.setdefault(u, []).append((v, w))
        graph.setdefault(v, [])
    return graph


def build_edge_length_lookup(edges):
    return {(u, v): float(w) for u, v, w in edges}


def get_allowed_neighbors(current, prev_node, graph):
    all_neighbors = graph.get(current, [])
    if current not in TURN_RESTRICTIONS:
        return all_neighbors
    if prev_node is None:
        return all_neighbors
    restrictions = TURN_RESTRICTIONS[current]
    if prev_node not in restrictions:
        return all_neighbors
    allowed = set(restrictions[prev_node])
    return [(n, w) for n, w in all_neighbors if n in allowed]


def a_star(start, goal, graph, start_prev=None):
    # Dijkstra (heuristic = 0) with turn restrictions (state = (node, prev_node))
    open_heap = []
    heappush(open_heap, (0.0, start, start_prev))
    g_cost = {(start, start_prev): 0.0}
    parent = {}

    while open_heap:
        g, current, prev_node = heappop(open_heap)
        state = (current, prev_node)
        if current == goal:
            path = []
            node_state = state
            while node_state in parent:
                path.append(node_state[0])
                node_state = parent[node_state]
            path.append(start)
            path.reverse()
            return path, g_cost[state]

        if g_cost.get(state, float("inf")) < g:
            continue

        for neighbor, w in get_allowed_neighbors(current, prev_node, graph):
            neighbor_state = (neighbor, current)
            tentative = g_cost[state] + w
            if tentative < g_cost.get(neighbor_state, float("inf")):
                g_cost[neighbor_state] = tentative
                parent[neighbor_state] = state
                heappush(open_heap, (tentative, neighbor, current))

    return None, float("inf")


def compute_route_from_input(graph):
    print("Nodes available:", " ".join(sorted(graph.keys())))
    waypoints = []
    start = input("Enter start node (e.g., HUB): ").strip().upper()
    if start not in graph:
        print("Unknown start node.")
        return [], 0.0
    waypoints.append(start)

    while True:
        goal = input("Enter next goal node (or 'done' to finish): ").strip().upper()
        if goal == "DONE":
            break
        if goal not in graph:
            print(f"Unknown node '{goal}'. Try again.")
            continue
        waypoints.append(goal)

    if len(waypoints) < 2:
        print("Need at least a start and one goal node.")
        return [], 0.0

    full_path = []
    total_cost = 0.0
    prev_node_at_start = None
    for i in range(len(waypoints) - 1):
        if i > 0 and len(full_path) >= 2:
            prev_node_at_start = full_path[-2]
        else:
            prev_node_at_start = None
        path, cost = a_star(waypoints[i], waypoints[i + 1], graph, start_prev=prev_node_at_start)
        if path is None:
            print(f"No path found from {waypoints[i]} to {waypoints[i + 1]}.")
            return [], 0.0
        if i == 0:
            full_path.extend(path)
        else:
            full_path.extend(path[1:])
        total_cost += cost

    return full_path, total_cost

# ===================== CAMERA SETTING =====================
# QLabs frames are typically RGB. OpenCV expects BGR.
# Press 'T' while running to flip if your colors look wrong.
SOURCE_IS_RGB = True
# ==========================================================

ACTOR_NUMBER = 0

# ---------------- Speed rule (requested) ----------------
# If NO yellow line -> speed = 1.5
# If yellow line OR turning -> speed = 3.0
SPEED_NO_YELLOW = 1.5
SPEED_YELLOW_OR_TURN = 3.0

# ---------------- Time-based route progress (optional) ----------------
# Dead-reckoning estimate:
#   progressed_distance += commanded_speed * dt * SPEED_CMD_TO_MPS
# Advance edge when progressed_distance reaches edge distance.
ENABLE_TIME_BASED_ROUTE_PROGRESS = True
SPEED_CMD_TO_MPS = 1.0   # calibration: speed-command units -> m/s
EDGE_REACH_SCALE = 1.00  # >1.0 waits longer; <1.0 advances earlier
EDGE_MIN_SECONDS = 0.80  # minimum time before allowing edge advance

# ---------------- Front-line follower (VIRTUAL) ----------------
BOTTOM_FRAC = 0.40          # bottom portion of image
BAND_FRAC   = 0.20          # lower band inside ROI for centroid
MIN_BAND_PTS = 30
STEER_GAIN  = 0.0020
STEER_CLIP  = 0.5

# In your real FSM you had TARGET_OFFSET_RIGHT=1000 (worked for 1280 wide + real setup).
# In QLabs widths vary; keep the "bias to right" concept but clamp safely.
TARGET_OFFSET_RIGHT = 650   # bias target toward right side; SAFE in typical QLabs widths

# VIRTUAL yellow HSV range (from your working virtual script)
Y_LOWER = np.array([10,  35,  50], dtype=np.uint8)
Y_UPPER = np.array([70, 255, 255], dtype=np.uint8)

KERNEL5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# Front ROI toggle (debug)
FRONT_ROI_ON = True
FRONT_ROI_Y_START_FRAC = 0.45  # match your virtual script style

# Min contour area for line detect (debug)
FRONT_MIN_AREA = 50

# ---------------- STOP sign (RGB-only in QLabs) ----------------
# (No depth gating here; we trigger by red shape area/aspect)
RED1_LOWER = np.array([0,   150, 120], dtype=np.uint8)
RED1_UPPER = np.array([6,   255, 255], dtype=np.uint8)
RED2_LOWER = np.array([170, 150, 120], dtype=np.uint8)
RED2_UPPER = np.array([179, 255, 255], dtype=np.uint8)

STOP_DURATION = 3.0
STOP_DOWNSCALE = 0.6667
MIN_STOP_AREA = 450       # tune for QLabs if needed
MAX_STOP_AREA = 90000
MIN_ASPECT_RATIO = 0.65

KERNEL3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# ---------------- Minimal Right-Turn Tunables (your logic preserved) ----------------
CROP_LEFT_FRAC = 0.20
RIGHT_Y_TARGET_OFFSET = +15
RIGHT_STEER_GAIN_Y = 0.009
RIGHT_Y_DEADPX = 10
STEER_CLIP_TURN = 0.50

TURN_SPEED_MAX = 0.072
TURN_SPEED_MIN = 0.068
TURN_SPEED_KP  = 2.96296e-05

NUDGE_SPEED = 0.035
NUDGE_STEER = +0.35

# ---------------- UI ----------------
WINDOW = "QLabs QCar2 FSM (VIRTUAL)"
RIGHT_WINDOW = "RightCam"
SHOW_RIGHT_ALWAYS_DEFAULT = True

HUD_X, HUD_Y, HUD_DY = 10, 30, 30


# ==========================================================
# Utility
# ==========================================================
def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

def ensure_bgr(frame):
    if frame is None:
        return None
    if SOURCE_IS_RGB:
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame

def apply_roi(img_bgr, roi_on=True, y_start_frac=0.45):
    if not roi_on:
        return img_bgr, (0, 0)
    h, w = img_bgr.shape[:2]
    y0 = int(h * y_start_frac)
    return img_bgr[y0:h, :], (0, y0)


# ==========================================================
# Front: yellow mask + line info (VIRTUAL thresholds)
# ==========================================================
def make_yellow_mask_virtual(roi_bgr):
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, Y_LOWER, Y_UPPER)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  KERNEL5, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL5, iterations=1)
    return mask

def get_line_info_bottom_virtual(image_bgr, roi_on=True):
    h, w = image_bgr.shape[:2]

    # Optional extra ROI (like your virtual tool) BEFORE bottom crop
    roi_img, (x_off, y_off) = apply_roi(image_bgr, roi_on=roi_on, y_start_frac=FRONT_ROI_Y_START_FRAC)
    h2, w2 = roi_img.shape[:2]

    # Bottom crop inside that ROI
    y0_local = int(h2 * (1.0 - BOTTOM_FRAC))
    roi = roi_img[y0_local:h2, 0:w2]

    mask = make_yellow_mask_virtual(roi)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    largest = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(largest) < float(FRONT_MIN_AREA):
        return None

    pts = largest.reshape(-1, 2)
    roi_h = roi.shape[0]
    band_y_start = int(roi_h * (1.0 - BAND_FRAC))
    band_pts = pts[pts[:, 1] >= band_y_start]

    if band_pts.shape[0] >= MIN_BAND_PTS:
        cx = int(float(band_pts[:, 0].mean()))
        cy = int(float(band_pts[:, 1].mean()))
    else:
        N = min(50, pts.shape[0])
        sel = pts[np.argsort(pts[:, 1])[-N:]]
        cx = int(sel[:, 0].mean())
        cy = int(sel[:, 1].mean())

    # Map contour & centroid back to FULL image coordinates
    # roi_img starts at (0, y_off)
    # roi (bottom part) starts at (0, y_off + y0_local)
    contour_full = largest + np.array([0, y_off + y0_local], dtype=np.int32)
    centroid_full = (cx, y_off + y0_local + cy)

    return {
        "contour": contour_full,
        "centroid": centroid_full,
        "cx_full": cx,
        "roi_y_off": y_off,
        "roi_bottom_y_off": (y_off + y0_local),
        "roi_bottom_h": roi.shape[0],
        "roi_bottom_w": roi.shape[1],
        "band_y_start_full": (y_off + y0_local + band_y_start),
    }


# ==========================================================
# STOP sign detect (RGB-only, QLabs)
# ==========================================================
def detect_stop_sign_rgb_only(img_bgr):
    h, w = img_bgr.shape[:2]
    if STOP_DOWNSCALE != 1.0:
        small = cv2.resize(img_bgr, (int(w * STOP_DOWNSCALE), int(h * STOP_DOWNSCALE)),
                           interpolation=cv2.INTER_LINEAR)
        scale = 1.0 / STOP_DOWNSCALE
    else:
        small = img_bgr
        scale = 1.0

    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, RED1_LOWER, RED1_UPPER)
    m2 = cv2.inRange(hsv, RED2_LOWER, RED2_UPPER)
    mask = cv2.bitwise_or(m1, m2)

    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL3, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  KERNEL3, iterations=1)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        area = cv2.contourArea(c)
        if area < MIN_STOP_AREA * (STOP_DOWNSCALE ** 2) or area > MAX_STOP_AREA * (STOP_DOWNSCALE ** 2):
            continue
        x, y, ww, hh = cv2.boundingRect(c)
        ar = ww / float(hh) if hh else 0.0
        if ar < MIN_ASPECT_RATIO or ar > (1.0 / MIN_ASPECT_RATIO):
            continue

        # Convert bbox back to full res
        xF, yF = int(x * scale), int(y * scale)
        wF, hF = int(ww * scale), int(hh * scale)
        return True, (xF, yF, wF, hF)
    return False, None


# ==========================================================
# Right turn helpers (your logic)
# ==========================================================
def find_right_edge_overlay(img_bgr):
    """Return overlay info from lower-half right crop; None if nothing."""
    h, w = img_bgr.shape[:2]
    x0 = int(w * CROP_LEFT_FRAC)
    y0 = h // 2
    roi = img_bgr[y0:h, x0:w]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    c = max(cnts, key=cv2.contourArea)
    M = cv2.moments(c)
    if M["m00"] <= 0:
        return None

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    contour_full  = c + np.array([x0, y0], dtype=np.int32)
    centroid_full = (x0 + cx, y0 + cy)

    return {
        "contour":  contour_full,
        "centroid": centroid_full,
        "crop_box": (x0, y0, w - x0, h - y0)
    }

def compute_turn(dy):
    """Map vertical error -> steering & speed (never zero)."""
    if abs(dy) <= RIGHT_Y_DEADPX:
        steer = 0.0
    else:
        steer = float(np.clip(dy * RIGHT_STEER_GAIN_Y, STEER_CLIP_TURN, -STEER_CLIP_TURN))

    prop = abs(dy) + 1
    speed = float(np.clip(TURN_SPEED_MAX - TURN_SPEED_KP * prop,
                          TURN_SPEED_MIN, TURN_SPEED_MAX))
    return speed, steer


# ==========================================================
# FSM
# ==========================================================
class State(enum.Enum):
    DRIVE = 0
    STOPPING = 1
    TURN_RIGHT = 2


def main():
    global SOURCE_IS_RGB, FRONT_ROI_ON, FRONT_MIN_AREA

    graph = build_graph(EDGES)
    route, total_meters = compute_route_from_input(graph)
    if not route:
        print("Route not created. Exiting.")
        return
    print(f"Planned route: {' -> '.join(route)}")
    print(f"Total theoretical distance: {total_meters:.2f} meters")

    # --- Connect to QLabs ---
    qlabs = QuanserInteractiveLabs()
    print("Connecting to QLabs (localhost)...")
    qlabs.open("localhost")
    print("Connected.")

    car = QLabsQCar2(qlabs)
    car.actorNumber = ACTOR_NUMBER
    car.possess(car.CAMERA_TRAILING)

    FRONT_CAM = car.CAMERA_CSI_FRONT
    RIGHT_CAM = car.CAMERA_CSI_RIGHT

    show_right_always = SHOW_RIGHT_ALWAYS_DEFAULT

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.namedWindow(RIGHT_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 1100, 620)
    cv2.resizeWindow(RIGHT_WINDOW, 700, 450)

    state = State.DRIVE
    stop_t0 = 0.0

    route_index = 0
    current_edge = (route[0], route[1]) if len(route) >= 2 else None
    edge_lengths = build_edge_length_lookup(EDGES)
    edge_progress_m = 0.0
    edge_elapsed_s = 0.0
    progress_dt_prev = time.time()

    # Stop at start
    car.set_velocity_and_request_state(
        forward=0.0, turn=0.0,
        headlights=False, leftTurnSignal=False, rightTurnSignal=False,
        brakeSignal=False, reverseSignal=False
    )

    fps_count = 0
    fps = 0
    last_fps_t = time.time()

    try:
        while True:
            loop_t0 = time.time()
            dt_progress = max(0.0, loop_t0 - progress_dt_prev)
            progress_dt_prev = loop_t0

            okF, rawF = car.get_image(camera=FRONT_CAM)
            okR, rawR = car.get_image(camera=RIGHT_CAM)

            if not okF or rawF is None:
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q'), ord('Q')):
                    break
                continue

            imgF_bgr = ensure_bgr(rawF)
            disp = imgF_bgr.copy()
            hF, wF = disp.shape[:2]

            # Right preview when not turning
            if show_right_always and state != State.TURN_RIGHT and okR and rawR is not None:
                imgR_bgr = ensure_bgr(rawR)
                rc_vis = imgR_bgr.copy()
                infoR = find_right_edge_overlay(rc_vis)
                if infoR is not None:
                    x0, y0, ww, hh = infoR["crop_box"]
                    cv2.rectangle(rc_vis, (x0, y0), (x0 + ww - 1, y0 + hh - 1), (0, 255, 255), 2)
                    cv2.drawContours(rc_vis, [infoR["contour"]], -1, (255, 0, 0), 2)
                    cv2.circle(rc_vis, infoR["centroid"], 8, (255, 0, 0), -1)
                cv2.putText(rc_vis, "INACTIVE: FRONT DRIVE", (10, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.imshow(RIGHT_WINDOW, rc_vis)

            # ==================== FSM ====================
            if state == State.DRIVE:
                # STOP sign detect (virtual RGB-only)
                stop_seen, stop_box = detect_stop_sign_rgb_only(imgF_bgr)
                if stop_seen:
                    state = State.STOPPING
                    stop_t0 = time.time()
                    x, y, ww, hh = stop_box
                    cv2.rectangle(disp, (x, y), (x + ww, y + hh), (0, 255, 0), 3)
                    cv2.putText(disp, "STOP", (x, max(20, y - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    drive_speed = 0.0
                    drive_steer = 0.0
                else:
                    # ==================================================
                    # EDGE-SPECIFIC LINE FOLLOWING HOOKS
                    # TODO: Replace the default front-yellow follower per edge
                    # Example:
                    # if current_edge == ("A", "B"):
                    #     # TODO: call your custom line follower for edge A->B
                    # elif current_edge == ("B", "F"):
                    #     # TODO: call your custom line follower for edge B->F
                    # ==================================================
                    info = get_line_info_bottom_virtual(imgF_bgr, roi_on=FRONT_ROI_ON)
                    steering = 0.0

                    if info is not None:
                        # Desired x biased right, but clamped safely for any width
                        desired_x = int(clamp(wF - TARGET_OFFSET_RIGHT, 0, wF - 1))
                        error = int(info["cx_full"]) - desired_x
                        steering = float(np.clip(error * STEER_GAIN, -STEER_CLIP, STEER_CLIP))

                        # Draw debug
                        cv2.drawContours(disp, [info["contour"]], -1, (255, 0, 0), 2)
                        cv2.circle(disp, info["centroid"], 7, (255, 0, 0), -1)
                        cv2.circle(disp, (desired_x, info["roi_bottom_y_off"] + info["roi_bottom_h"] // 2),
                                   7, (0, 0, 255), -1)
                        cv2.rectangle(disp,
                                      (0, info["band_y_start_full"]),
                                      (wF - 1, info["roi_bottom_y_off"] + info["roi_bottom_h"] - 1),
                                      (0, 200, 200), 2)
                    else:
                        cv2.putText(disp, "NO YELLOW", (10, 140),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                    # --- SPEED RULE (requested) ---
                    drive_speed = SPEED_YELLOW_OR_TURN if (info is not None) else SPEED_NO_YELLOW
                    drive_steer = steering

            elif state == State.STOPPING:
                drive_speed = 0.0
                drive_steer = 0.0
                elapsed = time.time() - stop_t0
                cv2.putText(disp, f"STOPPING {elapsed:.1f}/{STOP_DURATION:.1f}s",
                            (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
                if elapsed >= STOP_DURATION:
                    state = State.TURN_RIGHT

            elif state == State.TURN_RIGHT:
                # Turning => speed must be 3.0 (requested)
                drive_speed = SPEED_YELLOW_OR_TURN

                if okR and rawR is not None:
                    imgR_bgr = ensure_bgr(rawR)
                    rc_vis = imgR_bgr.copy()

                    target_y_right = (rc_vis.shape[0] // 2) + (rc_vis.shape[0] // 4) + RIGHT_Y_TARGET_OFFSET

                    infoR = find_right_edge_overlay(rc_vis)
                    if infoR is not None:
                        x0, y0, ww, hh = infoR["crop_box"]
                        cv2.rectangle(rc_vis, (x0, y0), (x0 + ww - 1, y0 + hh - 1), (0, 255, 255), 2)
                        cv2.drawContours(rc_vis, [infoR["contour"]], -1, (255, 0, 0), 2)
                        cv2.circle(rc_vis, infoR["centroid"], 8, (255, 0, 0), -1)

                        dy = int(infoR["centroid"][1]) - int(target_y_right)
                        _spd_tmp, steer_cmd = compute_turn(dy)
                        ctrl_mode = "Y"
                    else:
                        dy = 0
                        steer_cmd = NUDGE_STEER
                        ctrl_mode = "BLIND"

                    cv2.putText(rc_vis, f"TURN_RIGHT ctrl:{ctrl_mode} dy:{dy:+d}", (10, 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    cv2.imshow(RIGHT_WINDOW, rc_vis)

                    drive_steer = float(np.clip(steer_cmd, -STEER_CLIP_TURN, STEER_CLIP_TURN))
                else:
                    drive_steer = float(np.clip(NUDGE_STEER, -STEER_CLIP_TURN, STEER_CLIP_TURN))

                # Relock: if front yellow is seen again, exit turn
                info_f = get_line_info_bottom_virtual(imgF_bgr, roi_on=FRONT_ROI_ON)
                if info_f is not None:
                    cv2.putText(disp, "RELOCK -> DRIVE", (10, 170),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    state = State.DRIVE

            # ==================== End FSM ====================

            # Time-based route progression (no localization required)
            if ENABLE_TIME_BASED_ROUTE_PROGRESS and current_edge is not None:
                u, v = current_edge
                edge_len_m = edge_lengths.get((u, v), None)
                if edge_len_m is not None:
                    est_speed_mps = max(0.0, float(drive_speed) * SPEED_CMD_TO_MPS)
                    edge_progress_m += est_speed_mps * dt_progress
                    edge_elapsed_s += dt_progress

                    required_m = max(0.0, edge_len_m * EDGE_REACH_SCALE)
                    if edge_elapsed_s >= EDGE_MIN_SECONDS and edge_progress_m >= required_m:
                        route_index += 1
                        reached_node = route[route_index]
                        print(f"[Route] Reached node {reached_node} (time-based estimate)")

                        edge_progress_m = 0.0
                        edge_elapsed_s = 0.0

                        if route_index < (len(route) - 1):
                            current_edge = (route[route_index], route[route_index + 1])
                            print(f"[Route] Next edge: {current_edge[0]} -> {current_edge[1]}")
                        else:
                            current_edge = None
                            print("[Route] Final destination reached (time-based estimate).")

            # HUD
            fps_count += 1
            now = time.time()
            if now - last_fps_t >= 1.0:
                fps = fps_count
                fps_count = 0
                last_fps_t = now

            calc_ms = (time.time() - loop_t0) * 1000.0
            cv2.putText(disp, f"FPS:{fps}  Calc:{calc_ms:.1f}ms  State:{state.name}  SOURCE_IS_RGB:{SOURCE_IS_RGB}",
                        (HUD_X, HUD_Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(disp, f"Speed:{drive_speed:.2f}  Steer:{drive_steer:+.3f}  ROI:{int(FRONT_ROI_ON)}  MinA:{FRONT_MIN_AREA}",
                        (HUD_X, HUD_Y + HUD_DY), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(disp, f"Edge:{current_edge}",
                        (HUD_X, HUD_Y + 2*HUD_DY), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(disp, f"RouteIdx:{route_index}/{max(0, len(route)-1)}  EdgeProg:{edge_progress_m:.2f}m",
                        (HUD_X, HUD_Y + 3*HUD_DY), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(disp, "Keys: ESC/Q quit | T RGB/BGR | V right-preview | R ROI | [ ] minA | P exit turn",
                        (HUD_X, HUD_Y + 4*HUD_DY), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imshow(WINDOW, disp)

            # Keys
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                print("Exit.")
                break
            elif key in (ord('t'), ord('T')):
                SOURCE_IS_RGB = not SOURCE_IS_RGB
                print("SOURCE_IS_RGB =", SOURCE_IS_RGB)
            elif key in (ord('v'), ord('V')):
                show_right_always = not show_right_always
                print("Right preview:", show_right_always)
            elif key in (ord('r'), ord('R')):
                FRONT_ROI_ON = not FRONT_ROI_ON
                print("Front ROI:", FRONT_ROI_ON)
            elif key == ord('['):
                FRONT_MIN_AREA = max(10, FRONT_MIN_AREA - 10)
            elif key == ord(']'):
                FRONT_MIN_AREA = min(5000, FRONT_MIN_AREA + 10)
            elif key in (ord('p'), ord('P')):
                print("Manual override: exit TURN_RIGHT -> DRIVE")
                state = State.DRIVE

            # Apply command to vehicle (QLabs)
            car.set_velocity_and_request_state(
                forward=float(drive_speed), turn=float(drive_steer),
                headlights=False, leftTurnSignal=False, rightTurnSignal=False,
                brakeSignal=False, reverseSignal=False
            )

            time.sleep(0.02)

    finally:
        try:
            car.set_velocity_and_request_state(
                forward=0.0, turn=0.0,
                headlights=False, leftTurnSignal=False, rightTurnSignal=False,
                brakeSignal=True, reverseSignal=False
            )
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        try:
            qlabs.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
