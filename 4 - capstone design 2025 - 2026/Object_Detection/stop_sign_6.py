from Quanser.product_QCar import QCar
from Quanser.q_essential import Camera3D
import Quanser.q_interpretation as qi
import cv2
import numpy as np
import time
import math

# ------------------------------------------------------------
# Camera + Car Init (RGB + Depth)
# ------------------------------------------------------------
myCar = QCar()
frontCam = Camera3D(mode='RGB&DEPTH',
                    frame_width_RGB=1280,
                    frame_height_RGB=720,
                    frame_rate_RGB=20.0,
                    device_id='0')

# ------------------------------------------------------------
# Parameters
# ------------------------------------------------------------
WINDOW = "QCar Stop Sign Demo"
WINDOWS = ["Original", "HSV", "Red Mask 1", "Red Mask 2",
           "Combined Mask", "Processed Mask", "ROI Threshold", WINDOW]

base_speed = 0.08
stop_duration = 3.0

# HSV thresholds for red
RED1_LOWER = np.array([0, 150, 120])
RED1_UPPER = np.array([6, 255, 255])
RED2_LOWER = np.array([170, 150, 120])
RED2_UPPER = np.array([179, 255, 255])

MIN_STOP_AREA = 500
MAX_STOP_AREA = 50000
MIN_ASPECT_RATIO = 0.7

MAX_STOP_DISTANCE_M = 0.75  # Require STOP sign to be within 1 meter

# State
stopping = False
stop_timer = 0.0


# ------------------------------------------------------------
# Helper: safe depth -> float
# ------------------------------------------------------------
def read_distance_scalar(depth_img, cx, cy):
    """
    Safely read a single distance value from depth_img at (cx, cy)
    and return a Python float or math.nan if invalid.
    """
    try:
        # Bounds check
        if depth_img is None:
            return math.nan
        h, w = depth_img.shape[:2]
        if cx < 0 or cy < 0 or cx >= w or cy >= h:
            return math.nan

        val = depth_img[cy, cx]

        # If val is an ndarray (unexpected), take first element
        if isinstance(val, np.ndarray):
            if val.size == 0:
                return math.nan
            val = val.flat[0]

        # Convert to float if possible
        dist = float(val)
        if not math.isfinite(dist) or dist <= 0.0:
            return math.nan
        return dist
    except Exception:
        return math.nan


# ------------------------------------------------------------
# STOP Sign Detection with Depth Check
# ------------------------------------------------------------
def detect_stop_sign(frame, depth_img):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Red masks
    mask1 = cv2.inRange(hsv, RED1_LOWER, RED1_UPPER)
    mask2 = cv2.inRange(hsv, RED2_LOWER, RED2_UPPER)
    mask = cv2.bitwise_or(mask1, mask2)

    # Noise filtering
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Debug windows
    cv2.imshow("Original", frame)
    cv2.imshow("HSV", hsv)                 # OK to show (visual debugging)
    cv2.imshow("Red Mask 1", mask1)
    cv2.imshow("Red Mask 2", mask2)
    cv2.imshow("Combined Mask", mask)
    cv2.imshow("Processed Mask", mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_STOP_AREA or area > MAX_STOP_AREA:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        ar = float(w) / float(h) if h != 0 else 0.0
        if ar < MIN_ASPECT_RATIO or ar > 1.0 / MIN_ASPECT_RATIO:
            continue

        roi_mask = mask[y:y + h, x:x + w]
        if roi_mask.size == 0:
            continue
        red_ratio = cv2.countNonZero(roi_mask) / float(w * h)
        if red_ratio < 0.6:
            continue

        # Depth check (sample center pixel) - robust conversion
        cx, cy = x + w // 2, y + h // 2
        dist = read_distance_scalar(depth_img, cx, cy)

        # Reject invalid or far distances
        if math.isnan(dist) or dist > MAX_STOP_DISTANCE_M:
            continue  # not close enough or invalid → ignore

        # Text check (simple connected components)
        roi = frame[max(0, y - 5):min(frame.shape[0], y + h + 5),
                    max(0, x - 5):min(frame.shape[1], x + w + 5)]
        if roi.size == 0:
            continue

        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, roi_thresh = cv2.threshold(roi_gray, 127, 255,
                                      cv2.THRESH_BINARY_INV)
        cv2.imshow("ROI Threshold", roi_thresh)

        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
            roi_thresh, 4, cv2.CV_32S)

        for i in range(1, num_labels):
            comp_area = stats[i, cv2.CC_STAT_AREA]
            comp_w = stats[i, cv2.CC_STAT_WIDTH]
            comp_h = stats[i, cv2.CC_STAT_HEIGHT]
            if comp_area > 50 and 0.5 <= (comp_h / comp_w) <= 2.5:
                # Safe string formatting now that dist is a float
                dist_str = "{:.2f}m".format(dist)
                cv2.rectangle(frame, (x, y), (x + w, y + h),
                              (0, 255, 0), 3)
                cv2.putText(frame, "STOP " + dist_str,
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2)
                return True

    return False


# ------------------------------------------------------------
# Main Loop
# ------------------------------------------------------------
try:
    while True:
        frontCam.read_RGB()
        frontCam.read_depth(dataMode='m')

        img = frontCam.image_buffer_RGB
        depth = frontCam.image_buffer_depth_m

        if img is None or img.size == 0:
            continue
        if depth is None or getattr(depth, "size", 0) == 0:
            # if depth is not ready, skip this frame (or you could fallback)
            continue

        display = img.copy()
        stop_seen = detect_stop_sign(display, depth)

        # Stop logic
        if stop_seen and not stopping:
            stopping = True
            stop_timer = time.time()
            speed = 0.0
        elif stopping:
            speed = 0.0
            if time.time() - stop_timer >= stop_duration:
                stopping = False
                speed = base_speed
        else:
            speed = base_speed

        # Drive
        mtr_cmd = np.array([speed, 0.0], dtype=np.float64)
        LEDs = np.array([0, 0, 0, 0, 0, 0, 1, 1], dtype=np.float64)
        myCar.read_write_std(mtr_cmd, LEDs)

        cv2.imshow(WINDOW, display)
        if (cv2.waitKey(1) & 0xFF) == 27:
            break

        time.sleep(0.05)

finally:
    for window in WINDOWS:
        try:
            cv2.destroyWindow(window)
        except Exception:
            pass
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass
    try:
        myCar.terminate()
    except Exception:
        pass
    if hasattr(frontCam, "terminate"):
        try:
            frontCam.terminate()
        except Exception:
            pass
