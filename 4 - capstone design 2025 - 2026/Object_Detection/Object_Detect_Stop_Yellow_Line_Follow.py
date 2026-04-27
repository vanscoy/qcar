# qcar_fsm_yellow_pink_obstacle_stop.py
# QCar FSM: Yellow first, Pink fallback, Obstacle Stop
#
# Behavior:
# - Follow YELLOW if visible
# - If no yellow, follow PINK
# - If yellow appears, it overrides pink immediately
# - If no yellow and no pink, stop
# - If obstacle in front, keep steering the same but force speed = 0
# - Stop sign still causes timed stop, then returns to DRIVE
#
# Keys:
#   [ESC] = quit
#   [R]   = reset 3D cam

from Quanser.product_QCar import QCar
from Quanser.q_essential import Camera3D
import cv2, time, numpy as np, math, enum

# ---------------- HUD / General ----------------
HUD_X, HUD_Y, HUD_DY  = 10, 30, 30
WINDOW = "QCar FSM"
MAX_STEER_ANGLE_DEG = 28

# ---------------- Drive Tunables ----------------
BOTTOM_FRAC = 0.40
BAND_FRAC   = 0.20
MIN_BAND_PTS = 30

TARGET_OFFSET_RIGHT = 1000
SPEED_BASE = 0.078
STEER_GAIN = 0.0012
STEER_CLIP = 0.5

# ---------------- Stop Sign ----------------
RED1_LOWER = np.array([0,   150, 120], dtype=np.uint8)
RED1_UPPER = np.array([6,   255, 255], dtype=np.uint8)
RED2_LOWER = np.array([170, 150, 120], dtype=np.uint8)
RED2_UPPER = np.array([179, 255, 255], dtype=np.uint8)
MIN_STOP_AREA = 500
MAX_STOP_AREA = 50000
MIN_ASPECT_RATIO = 0.7
MAX_STOP_DISTANCE_M = 0.75
STOP_DURATION = 3.0
STOP_DOWNSCALE = 0.6667

KERNEL3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
KERNEL5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))

# ---------------- Obstacle Detection ----------------
OBSTACLE_ROI_X_START  = 0.30
OBSTACLE_ROI_X_END    = 0.70
OBSTACLE_ROI_Y_START  = 0.15
OBSTACLE_ROI_Y_END    = 0.70

OBSTACLE_DEPTH_MIN_M  = 0.05
OBSTACLE_DEPTH_MAX_M  = 0.60
OBSTACLE_DEPTH_THRESHOLD_M = 0.60

OBSTACLE_PIXEL_FRAC   = 0.24
OBSTACLE_MIN_VALID_PX = 80

OBSTACLE_RGB_FALLBACK      = False
OBSTACLE_RGB_MIN_BLOB_AREA = 3000
OBSTACLE_RGB_MAX_BRIGHT    = 80

OBSTACLE_CONSEC_N = 4

# ---------------- Pink Mask Tunables ----------------
# You can tune these if needed.
PINK_HSV_LOWER1 = np.array([140, 70, 70], dtype=np.uint8)
PINK_HSV_UPPER1 = np.array([169, 255, 255], dtype=np.uint8)

# Optional extra pink range if your track sometimes trends red-magenta
PINK_HSV_LOWER2 = np.array([170, 70, 70], dtype=np.uint8)
PINK_HSV_UPPER2 = np.array([179, 255, 255], dtype=np.uint8)

MIN_LINE_AREA = 50

# =====================================================================
# ---------------- Safe 3D Camera wrapper ------------------------------
# =====================================================================

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
        self.cam = Camera3D(mode=self.mode,
                            frame_width_RGB=self.w,
                            frame_height_RGB=self.h,
                            frame_rate_RGB=self.fps,
                            device_id=self.dev)
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

# =====================================================================
# ---------------- Helpers: Line masks --------------------------------
# =====================================================================

def make_yellow_mask(roi_bgr):
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    lower1 = (15,  90,  80); upper1 = (45, 255, 255)
    lower2 = (15,  40,  60); upper2 = (45, 255, 200)
    m1 = cv2.inRange(hsv, lower1, upper1)
    m2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(m1, m2)

    white_glare = cv2.inRange(hsv, (0,0,220), (180,60,255))
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(white_glare))

    lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)
    _, b_bin = cv2.threshold(lab[:,:,2], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.bitwise_and(mask, b_bin)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL5, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL5, iterations=1)
    return mask

def make_pink_mask(roi_bgr):
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

    m1 = cv2.inRange(hsv, PINK_HSV_LOWER1, PINK_HSV_UPPER1)
    m2 = cv2.inRange(hsv, PINK_HSV_LOWER2, PINK_HSV_UPPER2)
    mask = cv2.bitwise_or(m1, m2)

    # remove white glare
    white_glare = cv2.inRange(hsv, (0,0,220), (180,60,255))
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(white_glare))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL5, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL5, iterations=1)
    return mask

def get_line_info_from_mask(image_bgr, mask, source_name):
    h, w, _ = image_bgr.shape
    y0 = int(h * (1.0 - BOTTOM_FRAC))

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    largest = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(largest) < MIN_LINE_AREA:
        return None

    pts = largest.reshape(-1, 2)
    roi_h = mask.shape[0]
    band_y_start = int(roi_h * (1.0 - BAND_FRAC))
    band_pts = pts[pts[:, 1] >= band_y_start]

    if band_pts.shape[0] >= MIN_BAND_PTS:
        cx = int(float(band_pts[:, 0].mean()))
        cy = int(float(band_pts[:, 1].mean()))
    else:
        N = min(50, pts.shape[0])
        sel = pts[np.argsort(pts[:, 1])[-N:]]
        cx, cy = int(sel[:,0].mean()), int(sel[:,1].mean())

    contour_full = largest + np.array([0, y0])
    centroid_full = (cx, y0 + cy)

    return {
        "source": source_name,
        "contour": contour_full,
        "centroid": centroid_full,
        "cx_full": cx,
        "roi_origin": (0, y0),
        "roi_size": (w, h - y0),
        "band_y_start_full": y0 + band_y_start
    }

def get_priority_line_info(image_bgr):
    h, w, _ = image_bgr.shape
    y0 = int(h * (1.0 - BOTTOM_FRAC))
    roi = image_bgr[y0:h, 0:w]

    yellow_mask = make_yellow_mask(roi)
    pink_mask   = make_pink_mask(roi)

    yellow_info = get_line_info_from_mask(image_bgr, yellow_mask, "YELLOW")
    if yellow_info is not None:
        return yellow_info, yellow_mask, pink_mask

    pink_info = get_line_info_from_mask(image_bgr, pink_mask, "PINK")
    if pink_info is not None:
        return pink_info, yellow_mask, pink_mask

    return None, yellow_mask, pink_mask

# =====================================================================
# ---------------- Helpers: Stop sign ---------------------------------
# =====================================================================

def read_depth_center(depth_img, cx, cy):
    if depth_img is None:
        return math.nan
    h, w = depth_img.shape[:2]
    if cx < 1 or cy < 1 or cx >= w-1 or cy >= h-1:
        return math.nan
    patch = depth_img[cy-1:cy+2, cx-1:cx+2].astype(np.float32)
    patch = patch[np.isfinite(patch) & (patch > 0)]
    if patch.size == 0:
        return math.nan
    return float(np.median(patch))

def detect_stop_sign(rgb, depth_m):
    h, w = rgb.shape[:2]
    small = cv2.resize(rgb, (int(w*STOP_DOWNSCALE), int(h*STOP_DOWNSCALE)), interpolation=cv2.INTER_LINEAR) \
            if STOP_DOWNSCALE != 1.0 else rgb

    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, RED1_LOWER, RED1_UPPER)
    mask2 = cv2.inRange(hsv, RED2_LOWER, RED2_UPPER)
    mask = cv2.bitwise_or(mask1, mask2)

    mask = cv2.GaussianBlur(mask, (5,5), 0)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL3, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  KERNEL3, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    scale = 1.0 / STOP_DOWNSCALE if STOP_DOWNSCALE != 1.0 else 1.0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_STOP_AREA * (STOP_DOWNSCALE**2) or area > MAX_STOP_AREA * (STOP_DOWNSCALE**2):
            continue

        x, y, ww, hh = cv2.boundingRect(cnt)
        ar = ww / float(hh) if hh else 0.0
        if ar < MIN_ASPECT_RATIO or ar > 1.0 / MIN_ASPECT_RATIO:
            continue

        cx_small, cy_small = x + ww//2, y + hh//2
        cx_full, cy_full = int(cx_small * scale), int(cy_small * scale)
        dist = read_depth_center(depth_m, cx_full, cy_full)
        if not math.isfinite(dist) or dist > MAX_STOP_DISTANCE_M:
            continue

        x_full, y_full = int(x * scale), int(y * scale)
        w_full, h_full = int(ww * scale), int(hh * scale)
        return True, (x_full, y_full, w_full, h_full), dist

    return False, None, None

# =====================================================================
# ---------------- Helpers: Obstacle detection -------------------------
# =====================================================================

def _get_obstacle_roi_coords(frame_h, frame_w):
    return (int(frame_w * OBSTACLE_ROI_X_START),
            int(frame_h * OBSTACLE_ROI_Y_START),
            int(frame_w * OBSTACLE_ROI_X_END),
            int(frame_h * OBSTACLE_ROI_Y_END))

def _depth_metrics(depth_m, x0, y0, x1, y1):
    roi = depth_m[y0:y1, x0:x1].astype(np.float32)
    if roi.ndim == 3:
        roi = roi.squeeze(axis=-1)
    mask  = np.isfinite(roi) & (roi > OBSTACLE_DEPTH_MIN_M) & (roi < OBSTACLE_DEPTH_MAX_M)
    valid = roi[mask]
    total = roi.size or 1
    p10   = float(np.percentile(valid, 10)) if valid.size >= OBSTACLE_MIN_VALID_PX else math.nan
    frac  = valid.size / total
    return valid, p10, frac

def _rgb_contour_obstacle(rgb, x0, y0, x1, y1):
    roi  = rgb[y0:y1, x0:x1]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, OBSTACLE_RGB_MAX_BRIGHT, 255, cv2.THRESH_BINARY_INV)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN,  KERNEL5, iterations=1)
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

def draw_obstacle_roi(disp, frame_h, frame_w, detected: bool, p10_dist=None):
    x0, y0, x1, y1 = _get_obstacle_roi_coords(frame_h, frame_w)
    colour = (0, 0, 255) if detected else (200, 200, 0)
    cv2.rectangle(disp, (x0, y0), (x1, y1), colour, 2)
    if detected:
        label = "OBSTACLE!"
        if p10_dist is not None:
            label += f" {p10_dist:.2f}m"
        cv2.putText(disp, label, (x0, y0 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

def obstacle_p10(depth_m, frame_h, frame_w):
    if depth_m is None:
        return math.nan
    x0, y0, x1, y1 = _get_obstacle_roi_coords(frame_h, frame_w)
    _, p10, _ = _depth_metrics(depth_m, x0, y0, x1, y1)
    return p10

# =====================================================================
# ---------------- FSM Controller -------------------------------------
# =====================================================================

class State(enum.Enum):
    DRIVE    = 0
    STOPPING = 1

class QCarFSMController:
    def __init__(self):
        self.car = QCar()
        self.cam3d = SafeCamera3D(mode='RGB&DEPTH',
                                  frame_width=1280, frame_height=720,
                                  frame_rate=20.0, device_id='0',
                                  fail_reset_threshold=8,
                                  max_no_good_secs=2.5,
                                  verbose=True)

        self.state = State.DRIVE
        self.stop_t0 = 0.0
        self.obstacle_consec = 0

        self.frame_count = 0
        self.fps = 0
        self.last_time = time.time()

        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, 960, 540)
        cv2.moveWindow(WINDOW, 50, 50)

    def run(self):
        try:
            while True:
                loop_t0 = time.time()
                rgb, depth = self.cam3d.read()

                if rgb is None:
                    key = (cv2.waitKey(1) & 0xFF)
                    if key == 27:
                        break
                    continue

                hF, wF, _ = rgb.shape
                disp = rgb.copy()

                if self.state == State.DRIVE:
                    drive_speed, drive_steer = self._state_drive(disp, rgb, depth, wF, hF)
                elif self.state == State.STOPPING:
                    drive_speed, drive_steer = self._state_stopping(disp)
                else:
                    drive_speed, drive_steer = 0.0, 0.0

                self._update_hud(disp, drive_speed, drive_steer, loop_t0)

                cv2.imshow(WINDOW, disp)

                key = (cv2.waitKey(1) & 0xFF)
                if self._handle_key(key):
                    break

                self._send_drive_command(drive_speed, drive_steer)
                time.sleep(0.02)

        finally:
            self._cleanup()

    def _state_drive(self, disp, rgb, depth, wF, hF):
        # ---- Stop-sign detection ----
        stop_seen, stop_box, stop_dist = detect_stop_sign(rgb, depth)
        if stop_seen:
            self.state = State.STOPPING
            self.stop_t0 = time.time()
            x, y, ww, hh = stop_box
            cv2.rectangle(disp, (x, y), (x + ww, y + hh), (0, 255, 0), 3)
            cv2.putText(disp, f"STOP {stop_dist:.2f}m", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            return 0.0, 0.0

        # ---- Yellow priority, Pink fallback ----
        info, yellow_mask, pink_mask = get_priority_line_info(rgb)
        steering = 0.0
        line_source = "NONE"

        if info is not None:
            line_source = info["source"]
            desired_x = wF - TARGET_OFFSET_RIGHT
            error = desired_x - info["cx_full"]
            steering = float(np.clip(error * STEER_GAIN, -STEER_CLIP, STEER_CLIP))

            (rx0, ry0) = info["roi_origin"]
            (rw, rh) = info["roi_size"]

            if line_source == "YELLOW":
                color_main = (0, 255, 255)
                color_text = (0, 255, 255)
            else:
                color_main = (255, 0, 255)
                color_text = (255, 0, 255)

            cv2.rectangle(disp, (rx0, ry0), (rx0 + rw - 1, ry0 + rh - 1), color_main, 2)
            cv2.drawContours(disp, [info["contour"]], -1, color_main, 2)
            cv2.circle(disp, info["centroid"], 7, color_main, -1)
            cv2.circle(disp, (desired_x, ry0 + rh // 2), 7, (0, 0, 255), -1)
            cv2.rectangle(disp,
                          (rx0, info["band_y_start_full"]),
                          (rx0 + rw - 1, ry0 + rh - 1),
                          (0, 200, 200), 2)

            cv2.putText(disp, f"LINE: {line_source}", (10, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_text, 2)
        else:
            cv2.putText(disp, "NO YELLOW / NO PINK", (10, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # ---- Obstacle detection ----
        p10 = obstacle_p10(depth, hF, wF)
        obs_now = detect_obstacle(rgb, depth, hF, wF)

        if obs_now:
            self.obstacle_consec = min(OBSTACLE_CONSEC_N, self.obstacle_consec + 1)
        else:
            self.obstacle_consec = 0

        draw_obstacle_roi(disp, hF, wF, detected=obs_now,
                        p10_dist=p10 if math.isfinite(p10) else None)

        if info is None:
            return 0.0, 0.0

        if self.obstacle_consec >= OBSTACLE_CONSEC_N:
            cv2.putText(disp, "OBSTACLE STOP", (10, 170),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            return 0.0, steering

        return SPEED_BASE, steering

    def _state_stopping(self, disp):
        drive_speed = 0.0
        drive_steer = 0.0
        elapsed = time.time() - self.stop_t0

        cv2.putText(disp, f"STOPPING {elapsed:.1f}/{STOP_DURATION:.1f}s",
                    (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)

        if elapsed >= STOP_DURATION:
            self.state = State.DRIVE

        return drive_speed, drive_steer

    def _update_hud(self, disp, drive_speed, drive_steer, loop_t0):
        self.frame_count += 1
        now = time.time()
        if now - self.last_time >= 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.last_time = now

        calc_ms = (time.time() - loop_t0) * 1000.0
        angle_deg = drive_steer * MAX_STEER_ANGLE_DEG

        cv2.putText(disp,
                    f'FPS:{self.fps}  Calc:{calc_ms:.1f} ms  State:{self.state.name}',
                    (HUD_X, HUD_Y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(disp,
                    f'Speed:{drive_speed:.3f}  Steer:{drive_steer:+.3f}  Angle:{angle_deg:+.1f} deg',
                    (HUD_X, HUD_Y + HUD_DY),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        if self.state == State.DRIVE and self.obstacle_consec > 0:
            cv2.putText(disp,
                        f'OBS consec:{self.obstacle_consec}/{OBSTACLE_CONSEC_N}',
                        (HUD_X, HUD_Y + 2 * HUD_DY),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2)

    def _handle_key(self, key):
        if key == 27:
            print("ESC -> exit")
            return True
        elif key in (ord('r'), ord('R')):
            print("R -> camera reset")
            self.cam3d.force_reset()
        return False

    def _send_drive_command(self, drive_speed, drive_steer):
        mtr_cmd = np.array([drive_speed, drive_steer], dtype=np.float64)
        LEDs = np.array([0,0,0,0, 0,0,1,1], dtype=np.float64)
        try:
            self.car.read_write_std(mtr_cmd, LEDs)
        except Exception:
            pass

    def _cleanup(self):
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

# =====================================================================
# ---------------- main ------------------------------------------------
# =====================================================================

def main():
    controller = QCarFSMController()
    controller.run()

if __name__ == "__main__":
    main()