# -*- coding: utf-8 -*-
# QCar Yellow-Line Follower (Front 3D Camera + Motor Auto-Calibration)
# - Calibrates wheel deadzone & sign using encoder rates (safe, sub-clamp)
# - Very low base speed; tiny steering deltas; slew-limited
# - HSV mask for yellow; bottom-ROI centroid -> PD steering
# - ESC to stop (if SHOW_WINDOW=True)

from Quanser.product_QCar import QCar
from Quanser.q_essential import Camera3D
import numpy as np
import cv2
import time

# ===================== USER TUNABLES =====================
# Camera
CAMERA_ID = "0"
FRAME_W, FRAME_H, FPS = 640, 480, 30.0
SHOW_WINDOW = True   # Set False if running headless over SSH

# Line color (yellow) HSV range — tweak for your lighting
YELLOW_LOWER = np.array([18, 80, 120], dtype=np.uint8)
YELLOW_UPPER = np.array([48, 255, 255], dtype=np.uint8)
ROI_Y_FRACTION = 0.60   # use bottom 40% of image

# Motion limits (stay far below driver clamp, usually ±0.2)
CMD_LIMIT = 0.10        # hard clamp on per-wheel commands

# Base speed (very slow)
BASE_SPEED_TARGET = 0.020
MIN_BASE = 0.015        # don't fully stall when turning
SLEW_STEP_WHEEL = 0.010 # per-loop change limit (smooth)
SLEW_STEP_BASE  = 0.006

# Steering from pixel error (PD) -> small motor delta
Kp = 0.0015
Kd = 0.0008
MAX_DELTA = 0.015       # max steering delta per wheel (tiny)

# Calibration settings
DO_CALIBRATE = True
CAL_LEVELS   = [0.010, 0.015, 0.020, 0.025, 0.030, 0.035]  # safe, sub-clamp
RATE_HZ      = 40.0
DT           = 1.0 / RATE_HZ
HOLD_TIME_S  = 0.50
VEL_THRESH   = 4.0      # counts/sec threshold to consider "moving"
# =========================================================


def get_color_frame_3d(cam):
    """Return BGR color image from Camera3D, or None if not available this cycle."""
    cam.read()
    for attr in ("color_image", "color_data", "image_data"):
        if hasattr(cam, attr):
            img = getattr(cam, attr)
            if isinstance(img, np.ndarray) and img.size > 0:
                return img
    return None


def find_yellow_centroid(frame_bgr):
    """
    Detect yellow blob in bottom ROI. Returns (cx, cy, vis_mask) or (None, None, vis).
    cx,cy are in full-image coords.
    """
    h, w, _ = frame_bgr.shape
    y0 = int(h * ROI_Y_FRACTION)
    roi = frame_bgr[y0:h, 0:w]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    if not contours:
        return None, None, vis

    largest = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest)
    if M["m00"] <= 0:
        return None, None, vis

    cx_roi = int(M["m10"] / M["m00"])
    cy_roi = int(M["m01"] / M["m00"])
    cx = cx_roi
    cy = y0 + cy_roi

    if SHOW_WINDOW:
        cv2.circle(vis, (cx_roi, cy_roi), 6, (0, 255, 0), -1)

    return cx, cy, vis


def slew(u, prev, step):
    return float(np.clip(u, prev - step, prev + step))


def characterize_wheel(car, left_cmd, right_cmd, dur_s):
    """
    Apply constant commands for dur_s; return avg encoder rates [counts/s].
    """
    n = max(1, int(dur_s / DT))
    enc_prev = None
    vel_sum = np.array([0.0, 0.0])
    samples = 0
    for _ in range(n):
        mtr_cmd = np.array([left_cmd, right_cmd], dtype=np.float32)
        leds = np.array([0,0,0,0,0,0,1,1], dtype=np.uint8)
        current, battV, enc = car.read_write_std(mtr_cmd, leds)
        if enc_prev is not None:
            v = (np.array(enc, dtype=float) - enc_prev) / DT
            vel_sum += v
            samples += 1
        enc_prev = np.array(enc, dtype=float)
        time.sleep(DT)
    if samples == 0:
        return np.array([0.0, 0.0]), 0.0
    return vel_sum / samples, battV


def quick_calibration(car):
    """
    Estimate per-wheel deadzone (forward direction) and sign inversion.
    Returns (dz_L, dz_R, inv_L, inv_R).
    """
    print("Calibrating wheels (quick, safe)…")
    # settle
    characterize_wheel(car, 0.0, 0.0, 0.4)

    dz_L = CAL_LEVELS[-1]
    dz_R = CAL_LEVELS[-1]
    inv_L = False
    inv_R = False

    # Left wheel forward sweep (right=0)
    for u in CAL_LEVELS:
        v, _ = characterize_wheel(car, u, 0.0, HOLD_TIME_S)
        vL = v[0]
        if abs(vL) > VEL_THRESH:
            dz_L = u
            inv_L = (np.sign(vL) < 0)  # if positive cmd gives negative velocity -> inverted
            break

    # Right wheel forward sweep (left=0)
    for u in CAL_LEVELS:
        v, _ = characterize_wheel(car, 0.0, u, HOLD_TIME_S)
        vR = v[1]
        if abs(vR) > VEL_THRESH:
            dz_R = u
            inv_R = (np.sign(vR) < 0)
            break

    print(f"Calibrated deadzones: L≈{dz_L:.3f}, R≈{dz_R:.3f}; invert L={inv_L}, R={inv_R}")
    # stop
    characterize_wheel(car, 0.0, 0.0, 0.3)
    return dz_L, dz_R, inv_L, inv_R


def snap_deadzone_perwheel(u, dz):
    """Lift nonzero |u| up to the wheel's deadzone, without amplifying beyond what you asked for."""
    if abs(u) < 1e-9:
        return 0.0
    return float(np.copysign(max(dz, abs(u)), u))


def apply_inversion(l, r, inv_L, inv_R):
    if inv_L: l = -l
    if inv_R: r = -r
    return l, r


def main():
    car = QCar()
    cam = Camera3D(camera_id=CAMERA_ID, frame_width=FRAME_W, frame_height=FRAME_H, frame_rate=FPS)

    # LEDs
    LEDs = np.array([0,0,0,0,0,0,1,1], dtype=np.uint8)

    # --- Motor calibration ---
    if DO_CALIBRATE:
        dz_L, dz_R, inv_L, inv_R = quick_calibration(car)
        # keep a small safety margin
        dz_L += 0.002
        dz_R += 0.002
    else:
        dz_L, dz_R = 0.03, 0.03
        inv_L = inv_R = False

    prev_left = 0.0
    prev_right = 0.0
    base_cmd = 0.0  # soft-start
    prev_err = 0.0

    try:
        print("Yellow-line follower running. ESC to stop.")
        while True:
            frame = get_color_frame_3d(cam)
            if frame is None:
                # no frame: stop safely
                car.read_write_std(np.array([0.0, 0.0], dtype=np.float32), LEDs)
                time.sleep(DT)
                continue

            h, w, _ = frame.shape
            center_x = w // 2

            cx, cy, vis = find_yellow_centroid(frame)
            delta = 0.0

            if cx is not None:
                err = (cx - center_x)
                derr = err - prev_err
                prev_err = err

                # PD: pixels -> tiny motor delta
                delta = Kp * err + Kd * derr
                delta = float(np.clip(delta, -MAX_DELTA, MAX_DELTA))

                # Adaptive base: slow down when turning hard
                turn_ratio = min(1.0, abs(delta) / MAX_DELTA)  # 0..1
                target_base = max(MIN_BASE, BASE_SPEED_TARGET * (1.0 - 0.7 * turn_ratio))
            else:
                # Lost line: creep slowly, no steer
                prev_err *= 0.5
                target_base = MIN_BASE
                delta = 0.0

            # Slew-limit base speed
            base_cmd = slew(target_base, base_cmd, SLEW_STEP_BASE)

            # Differential mix (pre-inversion)
            left  = base_cmd - delta
            right = base_cmd + delta

            # Deadzone snap per wheel (only if nonzero)
            if abs(left)  > 0: left  = snap_deadzone_perwheel(left,  dz_L)
            if abs(right) > 0: right = snap_deadzone_perwheel(right, dz_R)

            # Clamp well under driver cap
            left  = float(np.clip(left,  -CMD_LIMIT, CMD_LIMIT))
            right = float(np.clip(right, -CMD_LIMIT, CMD_LIMIT))

            # Slew-limit wheel outputs
            left  = slew(left,  prev_left,  SLEW_STEP_WHEEL)
            right = slew(right, prev_right, SLEW_STEP_WHEEL)
            prev_left, prev_right = left, right

            # Apply sign inversion if needed
            left, right = apply_inversion(left, right, inv_L, inv_R)

            # Send
            car.read_write_std(np.array([left, right], dtype=np.float32), LEDs)

            # UI
            if SHOW_WINDOW:
                y0 = int(h * ROI_Y_FRACTION)
                disp = frame.copy()
                cv2.rectangle(disp, (0, y0), (w - 1, h - 1), (0, 180, 255), 2)
                cv2.line(disp, (center_x, 0), (center_x, h - 1), (255, 0, 0), 1)
                if cx is not None:
                    cv2.circle(disp, (cx, cy), 6, (0, 255, 0), -1)
                    cv2.putText(disp, f"err={err:.0f} d={delta:.3f} base={base_cmd:.3f}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60,220,60), 2)
                cv2.imshow("Front 3D Color", disp)
                cv2.imshow("Yellow ROI Mask", vis)
                if (cv2.waitKey(1) & 0xFF) == 27:
                    print("ESC pressed. Stopping.")
                    break

            time.sleep(DT)

    finally:
        try: cv2.destroyAllWindows()
        except: pass
        try: cam.terminate()
        except: pass
        try: car.terminate()
        except: pass


if __name__ == "__main__":
    main()
