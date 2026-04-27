# qlabs_rgb_depth_red_light_stop.py
# Copy-paste whole file.
#
# QLabs version of your "red traffic light with depth check" demo.
# Uses CAMERA_RGB (NOT CSI front cam).
#
# Keys:
#   Q / ESC : quit
#   T       : toggle SOURCE_IS_RGB (if colors look wrong)
#   W/S     : increase/decrease cruising speed
#   A/D     : steer left/right (manual)
#   SPACE   : stop (speed=0, turn=0)

import time
import math
import numpy as np
import cv2

from qvl.qlabs import QuanserInteractiveLabs
from qvl.qcar2 import QLabsQCar2

# ==========================================================
# CAMERA COLOR HANDLING
# QLabs often gives RGB, OpenCV expects BGR.
# Press 'T' to flip if red mask looks wrong.
# ==========================================================
SOURCE_IS_RGB = False

# ------------------------------------------------------------
# Parameters
# ------------------------------------------------------------
WINDOW_MAIN = "QLabs RGB+Depth Red Light Stop"
WINDOWS = ["Upper ROI", "Red Mask", WINDOW_MAIN]

# Vehicle motion
base_speed = 1.5        # QLabs forward speed (tune as needed)
turn_cmd   = 0.0        # you can steer with A/D
stop_duration = 3.0

# ROI: top 40% of frame, middle 50% horizontally (ignore left/right 25%)
UPPER_ROI_RATIO = 0.4
LEFT_CROP_RATIO  = 0.25
RIGHT_CROP_RATIO = 0.25

# HSV thresholds for RED traffic light
RED1_LOWER = np.array([0, 200, 150], dtype=np.uint8)
RED1_UPPER = np.array([10, 255, 255], dtype=np.uint8)
RED2_LOWER = np.array([170, 200, 150], dtype=np.uint8)
RED2_UPPER = np.array([179, 255, 255], dtype=np.uint8)

# Traffic light detection parameters
MIN_LIGHT_AREA = 20
MAX_LIGHT_AREA = 15000
MIN_ASPECT_RATIO = 0.2
MAX_ASPECT_RATIO = 5.0

MAX_LIGHT_DISTANCE_M = 20.0   # ignore red lights beyond this range
CLEAR_DURATION = 1.0         # must not see the light for this long before resuming

# State
stopping = False
stop_timer = 0.0
last_light_seen_time = 0.0


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
def ensure_bgr(frame):
    """Convert RGB->BGR if SOURCE_IS_RGB. Otherwise assume already BGR."""
    global SOURCE_IS_RGB
    if frame is None:
        return None
    if SOURCE_IS_RGB:
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame


def normalize_depth(depth):
    """
    Try to normalize depth output into a single-channel float array (meters).
    QLabs depth may come as HxW or HxWxC. We take channel 0 if needed.
    """
    if depth is None or not hasattr(depth, "shape"):
        return None

    d = depth
    if len(d.shape) == 3:
        # If it’s HxWx3 or HxWx1, take first channel
        d = d[:, :, 0]

    # Ensure float for math checks
    if d.dtype != np.float32 and d.dtype != np.float64:
        d = d.astype(np.float32, copy=False)

    return d


def align_depth_to_rgb(depth_m, rgb_bgr):
    """
    If depth resolution differs from RGB, resize depth to match RGB shape.
    """
    if depth_m is None or rgb_bgr is None:
        return None

    h, w = rgb_bgr.shape[:2]
    dh, dw = depth_m.shape[:2]
    if (dh, dw) != (h, w):
        # nearest to avoid smoothing distances too much
        depth_m = cv2.resize(depth_m, (w, h), interpolation=cv2.INTER_NEAREST)
    return depth_m


def read_distance_scalar(depth_m, cx, cy):
    """Safely read a single distance value from depth (meters)."""
    try:
        if depth_m is None:
            return math.nan
        h, w = depth_m.shape[:2]
        if cx < 0 or cy < 0 or cx >= w or cy >= h:
            return math.nan

        val = depth_m[cy, cx]
        dist = float(val)

        if not math.isfinite(dist) or dist <= 0.0:
            return math.nan
        return dist
    except Exception:
        return math.nan


# ------------------------------------------------------------
# Traffic Light Detection (Red Only) with Depth Gate
# ------------------------------------------------------------
def detect_traffic_light(frame_bgr, depth_m):
    """
    Detect red traffic lights in upper-middle portion of the frame.
    ROI: top 40% vertically, middle 50% horizontally.
    Returns: (detected, 'red') or (False, None)
    """
    full_height, full_width = frame_bgr.shape[:2]

    upper_height = int(full_height * UPPER_ROI_RATIO)
    left_x       = int(full_width * LEFT_CROP_RATIO)
    right_x      = int(full_width * (1.0 - RIGHT_CROP_RATIO))

    upper_roi   = frame_bgr[0:upper_height, left_x:right_x]
    upper_depth = depth_m[0:upper_height, left_x:right_x] if depth_m is not None else None

    cv2.imshow("Upper ROI", upper_roi)

    hsv = cv2.cvtColor(upper_roi, cv2.COLOR_BGR2HSV)

    red_mask1 = cv2.inRange(hsv, RED1_LOWER, RED1_UPPER)
    red_mask2 = cv2.inRange(hsv, RED2_LOWER, RED2_UPPER)
    red_mask  = cv2.bitwise_or(red_mask1, red_mask2)

    cv2.imshow("Red Mask", red_mask)

    # Filter noise
    filtered_mask = cv2.GaussianBlur(red_mask, (5, 5), 0)
    kernel = np.ones((3, 3), np.uint8)
    filtered_mask = cv2.morphologyEx(filtered_mask, cv2.MORPH_CLOSE, kernel)
    filtered_mask = cv2.morphologyEx(filtered_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(filtered_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_lights = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_LIGHT_AREA or area > MAX_LIGHT_AREA:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        ar = float(w) / float(h) if h != 0 else 0.0
        if ar < MIN_ASPECT_RATIO or ar > MAX_ASPECT_RATIO:
            continue

        # Depth check inside ROI coordinates
        cx_roi, cy_roi = x + w // 2, y + h // 2
        dist = read_distance_scalar(upper_depth, cx_roi, cy_roi) if upper_depth is not None else math.nan

        # If valid and too far, skip
        if (not math.isnan(dist)) and dist > MAX_LIGHT_DISTANCE_M:
            continue

        # Confirm some red pixels
        roi_red = red_mask[y:y+h, x:x+w]
        if roi_red.size == 0 or cv2.countNonZero(roi_red) == 0:
            continue

        detected_lights.append((x + left_x, y, w, h, dist))

    # Draw ROI boundary
    cv2.rectangle(frame_bgr, (left_x, 0), (right_x, upper_height), (255, 255, 0), 2)

    # Draw detections
    for (x, y, w, h, dist) in detected_lights:
        cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 0, 255), 3)
        dist_str = f"{dist:.2f}m" if not math.isnan(dist) else "N/A"
        cv2.putText(frame_bgr, f"RED LIGHT {dist_str}", (x, max(20, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    if detected_lights:
        return True, "red"
    return False, None


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    global SOURCE_IS_RGB, stopping, stop_timer, last_light_seen_time
    global base_speed, turn_cmd

    qlabs = QuanserInteractiveLabs()
    print("Connecting to QLabs (localhost)...")
    qlabs.open("localhost")
    print("Connected.")

    car = QLabsQCar2(qlabs)
    car.actorNumber = 0
    car.possess(car.CAMERA_TRAILING)

    CAM_RGB   = car.CAMERA_RGB
    CAM_DEPTH = car.CAMERA_DEPTH

    for w in WINDOWS:
        cv2.namedWindow(w, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_MAIN, 1150, 650)
    cv2.resizeWindow("Upper ROI", 700, 350)
    cv2.resizeWindow("Red Mask", 700, 350)

    # Start stopped
    car.set_velocity_and_request_state(
        forward=0.0, turn=0.0,
        headlights=False, leftTurnSignal=False, rightTurnSignal=False,
        brakeSignal=False, reverseSignal=False
    )

    try:
        while True:
            ok_rgb, raw_rgb = car.get_image(camera=CAM_RGB)
            ok_d,   raw_d   = car.get_image(camera=CAM_DEPTH)

            if not ok_rgb or raw_rgb is None:
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q'), ord('Q')):
                    break
                continue

            img_bgr = ensure_bgr(raw_rgb)

            depth_m = None
            if ok_d and raw_d is not None:
                depth_m = normalize_depth(raw_d)
                depth_m = align_depth_to_rgb(depth_m, img_bgr)

            display = img_bgr.copy()

            # Detect light
            light_detected, color = detect_traffic_light(display, depth_m)

            if light_detected:
                last_light_seen_time = time.time()

            # Stop logic
            if light_detected and (not stopping):
                stopping = True
                stop_timer = time.time()
                print("Stopping for red light!")
            elif stopping:
                # Wait stop_duration, then require CLEAR_DURATION since last seen
                if (time.time() - stop_timer) >= stop_duration:
                    time_since_last = time.time() - last_light_seen_time
                    if time_since_last >= CLEAR_DURATION:
                        stopping = False
                        print("Resuming movement - light clear.")
            # else: keep moving

            # HUD
            status_text  = "STOPPED" if stopping else "MOVING"
            status_color = (0, 0, 255) if stopping else (0, 255, 0)
            cv2.putText(display, status_text, (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, status_color, 3)

            cv2.putText(display,
                        f"base_speed={base_speed:.2f}  turn={turn_cmd:.2f}  SOURCE_IS_RGB={int(SOURCE_IS_RGB)}",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

            cv2.putText(display,
                        "W/S speed  A/D steer  SPACE stop  T RGB/BGR  Q/ESC quit",
                        (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

            cv2.imshow(WINDOW_MAIN, display)

            # Key handling
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                print("Exit.")
                break
            elif key in (ord('t'), ord('T')):
                SOURCE_IS_RGB = not SOURCE_IS_RGB
                print("SOURCE_IS_RGB =", SOURCE_IS_RGB)
            elif key in (ord('w'), ord('W')):
                base_speed = min(5.0, base_speed + 0.25)
            elif key in (ord('s'), ord('S')):
                base_speed = max(-5.0, base_speed - 0.25)
            elif key in (ord('a'), ord('A')):
                turn_cmd = max(-0.55, turn_cmd - 0.10)
            elif key in (ord('d'), ord('D')):
                turn_cmd = min(0.55, turn_cmd + 0.10)
            elif key == 32:  # SPACE
                base_speed = 0.0
                turn_cmd = 0.0

            # Apply command
            forward_cmd = 0.0 if stopping else float(base_speed)
            car.set_velocity_and_request_state(
                forward=forward_cmd, turn=float(turn_cmd),
                headlights=False, leftTurnSignal=False, rightTurnSignal=False,
                brakeSignal=stopping, reverseSignal=False
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