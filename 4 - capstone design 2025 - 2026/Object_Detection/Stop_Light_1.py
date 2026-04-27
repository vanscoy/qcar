# qlabs_traffic_light_demo_virtual.py
# Copy-paste whole file.
#
# Converts your IRL (QCar + Camera3D RGB&DEPTH) traffic-light demo to QLabs (virtual):
# - Uses QuanserInteractiveLabs + QLabsQCar2
# - Uses FRONT CSI camera frames (typically RGB) with SOURCE_IS_RGB toggle (T)
# - NO depth in QLabs CSI frames -> distance gating removed (you can add size-based gating instead)
# - Same upper-ROI HSV traffic light detection + stop timer logic
#
# Keys:
#   ESC / Q : quit
#   T       : toggle SOURCE_IS_RGB (RGB<->BGR)
#   SPACE   : force stop (debug)
#   G       : force go (debug)

import time
import math
import numpy as np
import cv2

from qvl.qlabs import QuanserInteractiveLabs
from qvl.qcar2 import QLabsQCar2

# ===================== CAMERA SETTING =====================
# QLabs frames are typically RGB. OpenCV expects BGR.
SOURCE_IS_RGB = True
# ==========================================================

ACTOR_NUMBER = 0

# ------------------------------------------------------------
# Parameters
# ------------------------------------------------------------
WINDOW = "QLabs Traffic Light Demo (VIRTUAL)"
WINDOWS = ["Upper ROI", "Horizontal ROI", "Red Mask", "Yellow Mask", "Combined Mask", WINDOW]

base_speed = 2.0          # virtual speed (tune)
stop_duration = 3.0

# Crop to upper portion (top 40% of frame)
UPPER_ROI_RATIO = 0.4
HORIZONTAL_ROI_RATIO = 0.5  # optional extra horizontal crop (center 50%)

# HSV thresholds for RED traffic light
RED1_LOWER = np.array([0, 150, 200], dtype=np.uint8)
RED1_UPPER = np.array([10, 255, 255], dtype=np.uint8)
RED2_LOWER = np.array([170, 150, 150], dtype=np.uint8)
RED2_UPPER = np.array([179, 255, 255], dtype=np.uint8)
# Traffic light detection parameters
MIN_LIGHT_AREA = 64        # virtual tune
MAX_LIGHT_AREA = 30000
MIN_ASPECT_RATIO = 0.3
MAX_ASPECT_RATIO = 3.0

# Optional extra gating to reduce false positives in sim:
# Lights are usually small-ish blobs. If you get false triggers, LOWER this cap.
MAX_BBOX_W_FRAC = 0.30     # reject huge red/yellow regions (like walls)
MAX_BBOX_H_FRAC = 0.30

# State
stopping = False
stop_timer = 0.0
last_detected_color = None
last_light_seen_time = 0.0
CLEAR_DURATION = 1.0  # Must not see light for 1 second before resuming


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
def ensure_bgr(frame):
    """QLabs often gives RGB; OpenCV wants BGR."""
    if frame is None:
        return None
    if SOURCE_IS_RGB:
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame


# ------------------------------------------------------------
# Traffic Light Detection (RGB-only in QLabs)
# ------------------------------------------------------------
def detect_traffic_light(frame_bgr):
    """
    Detect red traffic lights in the top 40% and middle 50% of the frame.
    Returns: (detected, color, best_box)
      color in {'red'} or None
      best_box = (x,y,w,h) in full-frame coords or None
    """
    global last_detected_color

    if frame_bgr is None or not hasattr(frame_bgr, "shape"):
        print("Frame is None or invalid.")
        return False, None, None

    H, W = frame_bgr.shape[:2]

    # Crop to top 40% and middle 50% of the frame
    top_h = int(H * UPPER_ROI_RATIO)  # Top 40% of the frame
    middle_w = int(W * HORIZONTAL_ROI_RATIO)  # Middle 50% of the frame
    start_x = (W - middle_w) // 2  # Start x-coordinate for middle 50%
    top_middle_roi = frame_bgr[:top_h, start_x:start_x + middle_w]

    # Debugging: Show the dimensions of the cropped region
    print(f"Top 40% height: {top_h}, Middle 50% width: {middle_w}, Start X: {start_x}")

    cv2.imshow("Top Middle ROI", top_middle_roi)

    # Convert to HSV color space
    hsv = cv2.cvtColor(top_middle_roi, cv2.COLOR_BGR2HSV)

    # Create masks for red color
    red_mask1 = cv2.inRange(hsv, RED1_LOWER, RED1_UPPER)
    red_mask2 = cv2.inRange(hsv, RED2_LOWER, RED2_UPPER)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    # Debugging: Show the red mask
    cv2.imshow("Red Mask", red_mask)

    # Noise filtering
    red_mask = cv2.GaussianBlur(red_mask, (5, 5), 0)
    kernel = np.ones((3, 3), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Find contours in the red mask
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Debugging: Print the number of contours found
    print(f"Number of contours found: {len(contours)}")

    best = None
    best_score = -1e18

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_LIGHT_AREA or area > MAX_LIGHT_AREA:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        ar = float(w) / float(h) if h else 0.0
        if ar < MIN_ASPECT_RATIO or ar > MAX_ASPECT_RATIO:
            continue

        # Adjust bounding box coordinates to full-frame
        x += start_x

        # Reject giant blobs (sim walls / sun patches)
        if (w / float(W)) > MAX_BBOX_W_FRAC or (h / float(top_h)) > MAX_BBOX_H_FRAC:
            continue

        # Determine color by pixel count within bbox
        roi_red = red_mask[y:y+h, x-start_x:x-start_x+w]
        red_pixels = cv2.countNonZero(roi_red) if roi_red.size else 0

        if red_pixels <= 0:
            continue

        color = "red"
        box_color = (0, 0, 255)
        dom = red_pixels

        # Score: prefer more dominant pixels + smaller bbox (more "light-like") + higher in image
        compact = dom / float((w * h) + 1)
        top_bias = (top_h - (y + h/2)) / float(top_h)  # higher => bigger
        score = (dom * 1.0) + (compact * 5000.0) + (top_bias * 200.0) - (area * 0.01)

        # Debugging: Print the score for each contour
        print(f"Contour score: {score}, Area: {area}, Aspect Ratio: {ar}")

        if score > best_score:
            best_score = score
            best = (x, y, w, h, color, box_color)

    if best is None:
        print("No red light detected.")
        return False, None, None

    x, y, w, h, color, box_color = best

    # Draw on FULL frame
    cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), box_color, 3)
    cv2.putText(frame_bgr, f"{color.upper()} LIGHT",
                (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)

    last_detected_color = color
    print(f"Red light detected at: x={x}, y={y}, w={w}, h={h}")
    return True, color, (x, y, w, h)


# ------------------------------------------------------------
# Main Loop (QLabs)
# ------------------------------------------------------------
def main():
    global SOURCE_IS_RGB, stopping, stop_timer, last_light_seen_time

    # --- Connect to QLabs ---
    qlabs = QuanserInteractiveLabs()
    print("Connecting to QLabs (localhost)...")
    qlabs.open("localhost")
    print("Connected.")

    car = QLabsQCar2(qlabs)
    car.actorNumber = ACTOR_NUMBER
    car.possess(car.CAMERA_TRAILING)

    FRONT_CAM = car.CAMERA_CSI_FRONT

    # Windows
    for w in WINDOWS:
        cv2.namedWindow(w, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 1100, 620)
    cv2.resizeWindow("Upper ROI", 900, 260)
    cv2.resizeWindow("Red Mask", 450, 260)
    cv2.resizeWindow("Yellow Mask", 450, 260)
    cv2.resizeWindow("Combined Mask", 450, 260)

    # Stop at start
    car.set_velocity_and_request_state(
        forward=0.0, turn=0.0,
        headlights=False, leftTurnSignal=False, rightTurnSignal=False,
        brakeSignal=False, reverseSignal=False
    )

    try:
        while True:
            ok, raw = car.get_image(camera=FRONT_CAM)
            if not ok or raw is None:
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q'), ord('Q')):
                    break
                continue

            img_bgr = ensure_bgr(raw)
            if img_bgr is None or img_bgr.size == 0:
                continue

            display = img_bgr.copy()

            light_detected, color, _box = detect_traffic_light(display)

            # Update last light seen time
            if light_detected:
                last_light_seen_time = time.time()

            # Stop logic - stop for both red and yellow lights
            if light_detected and color == "red" and not stopping:
                stopping = True
                stop_timer = time.time()
                speed = 0.0
                print(f"Stopping for {color} light!")
            elif stopping:
                speed = 0.0
                if time.time() - stop_timer >= stop_duration:
                    time_since_last_light = time.time() - last_light_seen_time
                    if time_since_last_light >= CLEAR_DURATION:
                        stopping = False
                        speed = base_speed
                        print("Resuming movement - light clear for 1 second")
            else:
                speed = base_speed

            # Status HUD
            status_text = "STOPPED" if stopping else "MOVING"
            status_color = (0, 0, 255) if stopping else (0, 255, 0)
            cv2.putText(display, status_text, (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, status_color, 3)

            cv2.putText(display, f"SPEED:{speed:.2f}  SOURCE_IS_RGB:{SOURCE_IS_RGB}  (T toggles)",
                        (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.imshow(WINDOW, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                break
            elif key in (ord('t'), ord('T')):
                SOURCE_IS_RGB = not SOURCE_IS_RGB
                print("SOURCE_IS_RGB =", SOURCE_IS_RGB)
            elif key == 32:  # SPACE force stop
                stopping = True
                stop_timer = time.time()
                last_light_seen_time = time.time()
                print("FORCE STOP")
            elif key in (ord('g'), ord('G')):  # force go
                stopping = False
                print("FORCE GO")

            # Apply command (QLabs)
            car.set_velocity_and_request_state(
                forward=float(speed), turn=0.0,
                headlights=False, leftTurnSignal=False, rightTurnSignal=False,
                brakeSignal=bool(stopping), reverseSignal=False
            )

            time.sleep(0.05)

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