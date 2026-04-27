#!/usr/bin/env python3
# QCar obstacle avoidance + encoder distance
#
# What this does:
# - Follows the yellow line normally
# - Detects an obstacle using the front depth image
# - Once obstacle is confirmed, it starts measuring encoder distance
# - It avoids by moving to the other lane for a fixed time
# - When avoidance ends, it stops measuring and shows total distance traveled
#
# Controls:
#   ESC : quit
#   R   : reset 3D camera
#
# Notes:
# - Distance measurement starts when obstacle detection is confirmed
# - Distance measurement ends when AVOID_DURATION is completed
# - Distance shown is only for that avoidance event

from Quanser.product_QCar import QCar
from Quanser.q_essential import Camera3D
import cv2, time, numpy as np, math, enum
from math import pi

# ============================================================
# ---------------- General tunables --------------------------
# ============================================================

WINDOW = "Obstacle Avoidance + Distance"

FRAME_W = 1280
FRAME_H = 720
FRAME_FPS = 20.0

MAX_STEER_ANGLE_DEG = 28.0
HUD_X, HUD_Y, HUD_DY = 10, 30, 30

# ============================================================
# ---------------- Yellow line follow tunables --------------
# ============================================================

BOTTOM_FRAC = 0.40
BAND_FRAC = 0.20
MIN_BAND_PTS = 30
MIN_CONTOUR_AREA = 50

TARGET_OFFSET_RIGHT = 1000
SPEED_BASE = 0.078
STEER_GAIN = 0.0012
STEER_CLIP = 0.5

KERNEL5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# ============================================================
# ---------------- Obstacle detection tunables --------------
# ============================================================

OBSTACLE_ROI_X_START = 0.30
OBSTACLE_ROI_X_END   = 0.70
OBSTACLE_ROI_Y_START = 0.15
OBSTACLE_ROI_Y_END   = 0.70

OBSTACLE_DEPTH_MIN_M = 0.05
OBSTACLE_DEPTH_MAX_M = 0.60
OBSTACLE_DEPTH_THRESHOLD_M = 0.60
OBSTACLE_PIXEL_FRAC = 0.24
OBSTACLE_MIN_VALID_PX = 80
OBSTACLE_CONSEC_N = 4

OBSTACLE_RGB_FALLBACK = False
OBSTACLE_RGB_MIN_BLOB_AREA = 3000
OBSTACLE_RGB_MAX_BRIGHT = 80

AVOID_DURATION = 2.0
AVOID_COOLDOWN = 2.0

TARGET_OFFSET_LEFT = 1000
AVOID_SPEED = SPEED_BASE
AVOID_STEER_GAIN = STEER_GAIN
AVOID_STEER_CLIP = STEER_CLIP

# ============================================================
# ---------------- Encoder / odometry ------------------------
# ============================================================

TICKS_PER_REV = 31844.0
WHEEL_DIAM_M = 0.066
WHEEL_CIRC_M = pi * WHEEL_DIAM_M
EMA_ALPHA = 0.35
DT_MIN, DT_MAX = 0.004, 0.35
MPS_HARD_MAX = 6.0

# ============================================================
# ---------------- Camera wrapper ----------------------------
# ============================================================

class SafeCamera3D:
    def __init__(self, mode='RGB&DEPTH', frame_width=1280, frame_height=720, frame_rate=20.0,
                 device_id='0', fail_reset_threshold=8, max_no_good_secs=2.5, verbose=True):
        self.mode = mode
        self.w, self.h, self.fps, self.dev = frame_width, frame_height, frame_rate, device_id
        self.fail_reset_threshold = fail_reset_threshold
        self.max_no_good_secs = max_no_good_secs
        self.verbose = verbose
        self.cam = None
        self._consec_fail = 0
        self._last_good_ts = 0.0
        self._init_cam()

    def _log(self, *a):
        if self.verbose:
            print("[SafeCamera3D]", *a)

    def _safe_terminate(self, cam):
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

    def _init_cam(self):
        if self.cam is not None:
            try:
                self._safe_terminate(self.cam)
            except Exception:
                pass

        self._log(f"Init Camera3D {self.mode} {self.w}x{self.h}@{self.fps} dev={self.dev}")
        self.cam = Camera3D(
            mode=self.mode,
            frame_width_RGB=self.w,
            frame_height_RGB=self.h,
            frame_rate_RGB=self.fps,
            device_id=self.dev
        )
        self._consec_fail = 0
        self._last_good_ts = 0.0

        try:
            if "RGB" in self.mode:
                self.cam.read_RGB()
            if "DEPTH" in self.mode:
                self.cam.read_depth(dataMode='m')
            rgb = getattr(self.cam, "image_buffer_RGB", None)
            dep = getattr(self.cam, "image_buffer_depth_m", None) if "DEPTH" in self.mode else None
            if self._is_valid(rgb) and (dep is None or self._is_valid_depth(dep)):
                self._last_good_ts = time.time()
                self._log("Warm-up OK.")
        except Exception:
            self._log("Warm-up read failed (will recover).")

    @staticmethod
    def _is_valid(img):
        return (img is not None and hasattr(img, "shape")
                and len(img.shape) == 3 and img.shape[0] > 0 and img.shape[1] > 0)

    @staticmethod
    def _is_valid_depth(img):
        return (img is not None and hasattr(img, "shape")
                and img.shape[0] > 0 and img.shape[1] > 0)

    def _needs_reset(self):
        if self._consec_fail >= self.fail_reset_threshold:
            return True
        if self._last_good_ts and (time.time() - self._last_good_ts) > self.max_no_good_secs:
            return True
        return False

    def read(self):
        try:
            if "RGB" in self.mode:
                self.cam.read_RGB()
            if "DEPTH" in self.mode:
                self.cam.read_depth(dataMode='m')

            rgb = getattr(self.cam, "image_buffer_RGB", None) if "RGB" in self.mode else None
            dep = getattr(self.cam, "image_buffer_depth_m", None) if "DEPTH" in self.mode else None

            ok_rgb = ("RGB" not in self.mode) or self._is_valid(rgb)
            ok_dep = ("DEPTH" not in self.mode) or self._is_valid_depth(dep)

            if ok_rgb and ok_dep:
                self._consec_fail = 0
                self._last_good_ts = time.time()
                return rgb, dep
            else:
                self._consec_fail += 1
                if self._needs_reset():
                    self._log("Invalid frames. Resetting stream...")
                    self._init_cam()
        except Exception as e:
            self._consec_fail += 1
            self._log(f"read() exception: {e}")
            if self._needs_reset():
                self._log("Exceptions persisted. Resetting stream...")
                self._init_cam()
        return None, None

    def force_reset(self):
        self._log("Force reset.")
        self._init_cam()

    def terminate(self):
        if self.cam is not None:
            try:
                self._safe_terminate(self.cam)
            except Exception:
                pass
            self.cam = None
            self._log("Camera terminated.")

# ============================================================
# ---------------- Odometry ----------------------------------
# ============================================================

def read_ticks(qcar) -> float:
    return float(qcar.read_encoder())

class SpeedOdom:
    def __init__(self, alpha=EMA_ALPHA):
        self.alpha = alpha
        self.v_filt = 0.0
        self.total_dist = 0.0
        self.prev_ticks = None
        self.prev_t = None

    def reset(self, qcar):
        self.prev_ticks = read_ticks(qcar)
        self.prev_t = time.time()
        self.v_filt = 0.0
        self.total_dist = 0.0

    def update(self, qcar):
        now = time.time()
        if self.prev_t is None:
            self.reset(qcar)
            return 0.0, 0.0, 0.0, 0.0

        dt = max(1e-3, now - self.prev_t)
        ticks_now = read_ticks(qcar)
        d_ticks = ticks_now - self.prev_ticks

        self.prev_ticks = ticks_now
        self.prev_t = now

        dist = (d_ticks / TICKS_PER_REV) * WHEEL_CIRC_M
        v = dist / dt

        if dt < DT_MIN or dt > DT_MAX or abs(v) > MPS_HARD_MAX:
            v = self.v_filt

        self.total_dist += abs(dist)
        self.v_filt = self.alpha * v + (1.0 - self.alpha) * self.v_filt
        return v, self.v_filt, self.total_dist, dt

# ============================================================
# ---------------- Yellow line helpers -----------------------
# ============================================================

def make_yellow_mask(roi_bgr):
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

    lower1 = (15, 90, 80)
    upper1 = (45, 255, 255)
    lower2 = (15, 40, 60)
    upper2 = (45, 255, 200)

    m1 = cv2.inRange(hsv, lower1, upper1)
    m2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(m1, m2)

    white_glare = cv2.inRange(hsv, (0, 0, 220), (180, 60, 255))
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(white_glare))

    lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)
    _, b_bin = cv2.threshold(lab[:, :, 2], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.bitwise_and(mask, b_bin)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL5, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL5, iterations=1)
    return mask

def get_line_info_bottom(image_bgr):
    h, w, _ = image_bgr.shape
    y0 = int(h * (1.0 - BOTTOM_FRAC))
    roi = image_bgr[y0:h, 0:w]
    mask = make_yellow_mask(roi)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    largest = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(largest) < MIN_CONTOUR_AREA:
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
        cx, cy = int(sel[:, 0].mean()), int(sel[:, 1].mean())

    contour_full = largest + np.array([0, y0])
    centroid_full = (cx, y0 + cy)

    return {
        "contour": contour_full,
        "centroid": centroid_full,
        "cx_full": cx,
        "roi_origin": (0, y0),
        "roi_size": (w, h - y0),
        "band_y_start_full": y0 + band_y_start
    }

# ============================================================
# ---------------- Obstacle helpers --------------------------
# ============================================================

def _get_obstacle_roi_coords(frame_h, frame_w):
    return (
        int(frame_w * OBSTACLE_ROI_X_START),
        int(frame_h * OBSTACLE_ROI_Y_START),
        int(frame_w * OBSTACLE_ROI_X_END),
        int(frame_h * OBSTACLE_ROI_Y_END)
    )

def _depth_metrics(depth_m, x0, y0, x1, y1):
    roi = depth_m[y0:y1, x0:x1].astype(np.float32)
    if roi.ndim == 3:
        roi = roi.squeeze(axis=-1)

    mask = np.isfinite(roi) & (roi > OBSTACLE_DEPTH_MIN_M) & (roi < OBSTACLE_DEPTH_MAX_M)
    valid = roi[mask]
    total = roi.size or 1
    p10 = float(np.percentile(valid, 10)) if valid.size >= OBSTACLE_MIN_VALID_PX else math.nan
    frac = valid.size / total
    return valid, p10, frac

def _rgb_contour_obstacle(rgb, x0, y0, x1, y1):
    roi = rgb[y0:y1, x0:x1]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, OBSTACLE_RGB_MAX_BRIGHT, 255, cv2.THRESH_BINARY_INV)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, KERNEL5, iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, KERNEL5, iterations=2)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return any(cv2.contourArea(c) >= OBSTACLE_RGB_MIN_BLOB_AREA for c in cnts)

def detect_obstacle(rgb, depth_m, frame_h, frame_w):
    if depth_m is None:
        if rgb is not None and OBSTACLE_RGB_FALLBACK:
            x0, y0, x1, y1 = _get_obstacle_roi_coords(frame_h, frame_w)
            return _rgb_contour_obstacle(rgb, x0, y0, x1, y1)
        return False

    x0, y0, x1, y1 = _get_obstacle_roi_coords(frame_h, frame_w)
    valid, p10, frac = _depth_metrics(depth_m, x0, y0, x1, y1)

    if valid.size < OBSTACLE_MIN_VALID_PX:
        if rgb is not None and OBSTACLE_RGB_FALLBACK:
            return _rgb_contour_obstacle(rgb, x0, y0, x1, y1)
        return False

    if math.isfinite(p10) and p10 < OBSTACLE_DEPTH_THRESHOLD_M:
        return True
    if frac >= OBSTACLE_PIXEL_FRAC:
        return True
    return False

def obstacle_p10(depth_m, frame_h, frame_w):
    if depth_m is None:
        return math.nan
    x0, y0, x1, y1 = _get_obstacle_roi_coords(frame_h, frame_w)
    _, p10, _ = _depth_metrics(depth_m, x0, y0, x1, y1)
    return p10

def draw_obstacle_roi(disp, frame_h, frame_w, detected, p10_dist=None):
    x0, y0, x1, y1 = _get_obstacle_roi_coords(frame_h, frame_w)
    colour = (0, 0, 255) if detected else (200, 200, 0)
    cv2.rectangle(disp, (x0, y0), (x1, y1), colour, 2)
    if detected:
        label = "OBSTACLE!"
        if p10_dist is not None:
            label += f" {p10_dist:.2f}m"
        cv2.putText(disp, label, (x0, y0 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

# ============================================================
# ---------------- State machine -----------------------------
# ============================================================

class State(enum.Enum):
    DRIVE = 0
    AVOID_OBSTACLE = 1

class QCarObstacleAvoidDistance:
    def __init__(self):
        self.car = QCar()
        self.cam3d = SafeCamera3D(
            mode='RGB&DEPTH',
            frame_width=FRAME_W,
            frame_height=FRAME_H,
            frame_rate=FRAME_FPS,
            device_id='0',
            fail_reset_threshold=8,
            max_no_good_secs=2.5,
            verbose=True
        )

        self.odo = SpeedOdom()
        self.odo.reset(self.car)

        self.state = State.DRIVE

        self.obstacle_consec = 0
        self.avoid_t0 = 0.0
        self.avoid_cooldown_t0 = 0.0

        self.measure_active = False
        self.last_avoid_distance_m = 0.0
        self.last_avoid_time_s = 0.0

        self.frame_count = 0
        self.fps = 0
        self.last_time = time.time()

        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, 1280, 720)

    def run(self):
        try:
            while True:
                loop_t0 = time.time()

                rgb, depth = self.cam3d.read()
                if rgb is None:
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:
                        break
                    continue

                v_raw, v_filt, dist_m, dt = self.odo.update(self.car)

                hF, wF, _ = rgb.shape
                disp = rgb.copy()

                if self.state == State.DRIVE:
                    drive_speed, drive_steer = self._state_drive(disp, rgb, depth, wF, hF)
                elif self.state == State.AVOID_OBSTACLE:
                    drive_speed, drive_steer = self._state_avoid_obstacle(disp, rgb, wF, hF)
                else:
                    drive_speed, drive_steer = 0.0, 0.0

                self._update_hud(disp, drive_speed, drive_steer, loop_t0, v_raw, v_filt, dist_m)

                cv2.imshow(WINDOW, disp)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    print("ESC -> exit")
                    break
                elif key in (ord('r'), ord('R')):
                    print("R -> camera reset")
                    self.cam3d.force_reset()

                self._send_drive_command(drive_speed, drive_steer)
                time.sleep(0.02)

        finally:
            self._cleanup()

    def _state_drive(self, disp, rgb, depth, wF, hF):
        in_cooldown = (time.time() - self.avoid_cooldown_t0) < AVOID_COOLDOWN
        p10 = obstacle_p10(depth, hF, wF)
        obs_now = (not in_cooldown) and detect_obstacle(rgb, depth, hF, wF)

        if obs_now:
            self.obstacle_consec += 1
        else:
            self.obstacle_consec = max(0, self.obstacle_consec - 1)

        draw_obstacle_roi(disp, hF, wF, detected=obs_now,
                          p10_dist=p10 if math.isfinite(p10) else None)

        if self.obstacle_consec >= OBSTACLE_CONSEC_N:
            print(f"[FSM] Obstacle confirmed ({self.obstacle_consec} frames) -> AVOID_OBSTACLE")
            self.state = State.AVOID_OBSTACLE
            self.avoid_t0 = time.time()
            self.obstacle_consec = 0

            # Start measuring here
            self.odo.reset(self.car)
            self.measure_active = True
            self.last_avoid_distance_m = 0.0
            self.last_avoid_time_s = 0.0

            return 0.0, 0.0

        info = get_line_info_bottom(rgb)
        steering = 0.0

        if info is not None:
            desired_x = wF - TARGET_OFFSET_RIGHT
            error = desired_x - info["cx_full"]
            steering = float(np.clip(error * STEER_GAIN, -STEER_CLIP, STEER_CLIP))

            (rx0, ry0) = info["roi_origin"]
            (rw, rh) = info["roi_size"]
            cv2.rectangle(disp, (rx0, ry0), (rx0 + rw - 1, ry0 + rh - 1), (0, 255, 0), 2)
            cv2.drawContours(disp, [info["contour"]], -1, (255, 0, 0), 2)
            cv2.circle(disp, info["centroid"], 7, (255, 0, 0), -1)
            cv2.circle(disp, (desired_x, ry0 + rh // 2), 7, (0, 0, 255), -1)
            cv2.rectangle(disp,
                          (rx0, info["band_y_start_full"]),
                          (rx0 + rw - 1, ry0 + rh - 1),
                          (0, 200, 200), 2)
        else:
            cv2.putText(disp, "NO YELLOW", (10, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        return SPEED_BASE, steering

    def _state_avoid_obstacle(self, disp, rgb, wF, hF):
        elapsed = time.time() - self.avoid_t0
        remaining = max(0.0, AVOID_DURATION - elapsed)

        bar_full = wF - 40
        bar_filled = int(bar_full * min(1.0, elapsed / AVOID_DURATION))
        cv2.rectangle(disp, (20, hF - 30), (20 + bar_full, hF - 10), (50, 50, 200), -1)
        cv2.rectangle(disp, (20, hF - 30), (20 + bar_filled, hF - 10), (0, 165, 255), -1)
        cv2.putText(disp,
                    f"AVOID OBSTACLE  {remaining:.1f}s remaining",
                    (10, hF - 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 165, 255), 2)

        if elapsed >= AVOID_DURATION:
            if self.measure_active:
                self.last_avoid_distance_m = self.odo.total_dist
                self.last_avoid_time_s = elapsed
                self.measure_active = False
                print(f"[Avoid] DONE | distance = {self.last_avoid_distance_m:.3f} m | time = {self.last_avoid_time_s:.2f} s")

            self.state = State.DRIVE
            self.obstacle_consec = 0
            self.avoid_cooldown_t0 = time.time()
            return 0.0, 0.0

        info = get_line_info_bottom(rgb)
        steering = 0.0

        if info is not None:
            desired_x = TARGET_OFFSET_LEFT
            error = desired_x - info["cx_full"]
            steering = float(np.clip(error * AVOID_STEER_GAIN,
                                     -AVOID_STEER_CLIP, AVOID_STEER_CLIP))

            (rx0, ry0) = info["roi_origin"]
            (rw, rh) = info["roi_size"]

            cv2.rectangle(disp, (rx0, ry0), (rx0 + rw - 1, ry0 + rh - 1), (0, 165, 255), 2)
            cv2.drawContours(disp, [info["contour"]], -1, (0, 165, 255), 2)
            cv2.circle(disp, info["centroid"], 7, (0, 165, 255), -1)
            cv2.circle(disp, (desired_x, ry0 + rh // 2), 8, (255, 255, 0), -1)
            cv2.putText(disp, f"LEFT err:{error:+d}px", (rx0, ry0 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        else:
            steering = -AVOID_STEER_CLIP * 0.4
            cv2.putText(disp, "AVOID: NO YELLOW - holding left",
                        (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

        return AVOID_SPEED, steering

    def _update_hud(self, disp, drive_speed, drive_steer, loop_t0, v_raw, v_filt, dist_m):
        self.frame_count += 1
        now = time.time()
        if now - self.last_time >= 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.last_time = now

        calc_ms = (time.time() - loop_t0) * 1000.0
        angle_deg = drive_steer * MAX_STEER_ANGLE_DEG

        cv2.putText(disp,
                    f"FPS:{self.fps}  Calc:{calc_ms:.1f} ms  State:{self.state.name}",
                    (HUD_X, HUD_Y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(disp,
                    f"Speed:{drive_speed:.3f}  Steer:{drive_steer:+.3f}  Angle:{angle_deg:+.1f} deg",
                    (HUD_X, HUD_Y + HUD_DY),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        if self.measure_active:
            cv2.putText(disp,
                        f"AVOID DIST ACTIVE: {dist_m:.3f} m",
                        (HUD_X, HUD_Y + 2 * HUD_DY),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        else:
            cv2.putText(disp,
                        f"LAST AVOID DIST: {self.last_avoid_distance_m:.3f} m   TIME: {self.last_avoid_time_s:.2f} s",
                        (HUD_X, HUD_Y + 2 * HUD_DY),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

        cv2.putText(disp,
                    f"v_raw:{v_raw:+.3f} m/s  v_filt:{v_filt:+.3f} m/s",
                    (HUD_X, HUD_Y + 3 * HUD_DY),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 2)

        if self.state == State.DRIVE and self.obstacle_consec > 0:
            cv2.putText(disp,
                        f"OBS consec:{self.obstacle_consec}/{OBSTACLE_CONSEC_N}",
                        (HUD_X, HUD_Y + 4 * HUD_DY),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2)

    def _send_drive_command(self, drive_speed, drive_steer):
        mtr_cmd = np.array([drive_speed, drive_steer], dtype=np.float64)
        LEDs = np.array([0, 0, 0, 0, 0, 0, 1, 1], dtype=np.float64)
        try:
            self.car.read_write_std(mtr_cmd, LEDs)
        except Exception:
            pass

    def _cleanup(self):
        try:
            self.car.read_write_std(
                np.array([0.0, 0.0], dtype=np.float64),
                np.array([1,0,0,0, 1,0,0,0], dtype=np.float64)
            )
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        try:
            self.car.terminate()
        except Exception:
            pass
        try:
            self.cam3d.terminate()
        except Exception:
            pass

def main():
    controller = QCarObstacleAvoidDistance()
    controller.run()

if __name__ == "__main__":
    main()