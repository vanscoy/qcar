# qcar_fsm_left_hardturn.py
# QCar FSM: Line-Follow + Stop-Sign + Hardcoded Left Turn
# - DRIVE: front yellow-line follower
# - STOPPING: timed stop after stop-sign detect
# - TURN_LEFT: open-loop left turn for fixed time, then back to DRIVE
#
# Keys: [ESC]=quit, [R]=reset 3D cam, [V]=toggle RightCam preview, [T]=force back to DRIVE

from Quanser.product_QCar import QCar
from Quanser.q_essential import Camera3D, Camera2D
import cv2, time, numpy as np, math, enum

# ---------------- Minimal Right-Cam (still for preview if you want) ----------------
RIGHT_CAM_ID     = "0"
FRAME_W, FRAME_H = 640, 480
FPS_RIGHT        = 30.0

CROP_LEFT_FRAC   = 0.20
RIGHT_Y_TARGET_OFFSET = +15
RIGHT_STEER_GAIN_Y    = 0.009
RIGHT_Y_DEADPX        = 10
STEER_CLIP_TURN       = 0.50
MAX_STEER_ANGLE_DEG   = 28

TURN_SPEED_MAX        = 0.072
TURN_SPEED_MIN        = 0.068
TURN_SPEED_KP         = 2.96296e-05

NUDGE_SPEED           = 0.035
NUDGE_STEER           = +0.35

# HUD placement (front window)
HUD_X, HUD_Y, HUD_DY  = 10, 30, 30

# ---------------- HARD-CODED LEFT TURN (from your stand-alone script) ---------------
INIT_TURN_TIME      = 11.0      # seconds of open-loop turn
INIT_TURN_SPEED     = 0.075     # forward speed during turn
INIT_TURN_STEERING  = 0.21       # steering command during turn (this gave you a good left arc)

# ---------------- DRIVE/STOP Front-Cam Tunables (same as your FSM) ----------------
BOTTOM_FRAC = 0.40
BAND_FRAC   = 0.20
MIN_BAND_PTS = 30
TARGET_OFFSET_RIGHT = 1000
SPEED_BASE = 0.078
STEER_GAIN = 0.0012
STEER_CLIP = 0.5

# STOP sign (same as your FSM)
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

# Front relock gating (still here if you ever bring back closed-loop turning)
MIN_RELOCK_M = 0.20
MAX_RELOCK_M = 1.00
RELOCK_CONSEC_N = 3

WINDOW = "QCar FSM"
RIGHT_WINDOW = "RightCam"
SHOW_RIGHT_ALWAYS_DEFAULT = True

# ---------------- Helpers: RightCam overlay (for preview only now) ----------------
def find_right_edge_overlay(img_bgr):
    """Return overlay info from lower-half right crop; None if nothing."""
    h, w, _ = img_bgr.shape
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
    contour_full  = c + np.array([x0, y0])
    centroid_full = (x0 + cx, y0 + cy)

    return {
        "contour":  contour_full,
        "centroid": centroid_full,
        "crop_box": (x0, y0, w - x0, h - y0)
    }

# ---------------- Front (Drive) helpers (same as your FSM core) ----------------
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

def get_line_info_bottom(image_bgr):
    h, w, _ = image_bgr.shape
    y0 = int(h * (1.0 - BOTTOM_FRAC))
    roi = image_bgr[y0:h, 0:w]
    mask = make_yellow_mask(roi)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    largest = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(largest) < 50: return None
    pts = largest.reshape(-1, 2)
    roi_h = roi.shape[0]
    band_y_start = int(roi_h * (1.0 - BAND_FRAC))
    band_pts = pts[pts[:, 1] >= band_y_start]
    if band_pts.shape[0] >= MIN_BAND_PTS:
        cx = int(float(band_pts[:, 0].mean())); cy = int(float(band_pts[:, 1].mean()))
    else:
        N = min(50, pts.shape[0])
        sel = pts[np.argsort(pts[:, 1])[-N:]]
        cx, cy = int(sel[:,0].mean()), int(sel[:,1].mean())
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

def read_depth_center(depth_img, cx, cy):
    if depth_img is None: return math.nan
    h, w = depth_img.shape[:2]
    if cx < 1 or cy < 1 or cx >= w-1: return math.nan
    if cy >= h-1: return math.nan
    patch = depth_img[cy-1:cy+2, cx-1:cx+2].astype(np.float32)
    patch = patch[np.isfinite(patch) & (patch > 0)]
    if patch.size == 0: return math.nan
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

# ---------------- Cameras ----------------
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
        if self.verbose: print("[SafeCamera3D]", *a)

    def _safe_terminate(self, cam):
        for name in ("terminate_RGB", "stop_RGB", "stop_rgb", "stop"):
            if hasattr(cam, name):
                try: getattr(cam, name)()
                except Exception: pass
        try: cam.terminate()
        except AttributeError as e:
            if "video3d" not in str(e): raise
        except Exception: pass

    def _init_cam(self):
        if self.cam is not None:
            try: self._safe_terminate(self.cam)
            except Exception: pass
        self._log(f"Init Camera3D {self.mode} {self.w}x{self.h}@{self.fps} dev={self.dev}")
        self.cam = Camera3D(mode=self.mode,
                            frame_width_RGB=self.w,
                            frame_height_RGB=self.h,
                            frame_rate_RGB=self.fps,
                            device_id=self.dev)
        self._consec_fail = 0
        self._last_good_ts = 0.0
        try:
            if "RGB" in self.mode: self.cam.read_RGB()
            if "DEPTH" in self.mode: self.cam.read_depth(dataMode='m')
            rgb = getattr(self.cam, "image_buffer_RGB", None)
            dep = getattr(self.cam, "image_buffer_depth_m", None) if "DEPTH" in self.mode else None
            if self._is_valid(rgb) and (dep is None or self._is_valid_depth(dep)):
                self._last_good_ts = time.time()
                self._log("Warm-up OK.")
        except Exception:
            self._log("Warm-up read failed (will recover).")

    @staticmethod
    def _is_valid(img):
        return (img is not None and hasattr(img, "shape") and len(img.shape) == 3
                and img.shape[0] > 0 and img.shape[1] > 0)

    @staticmethod
    def _is_valid_depth(img):
        return (img is not None and hasattr(img, "shape") and img.shape[0] > 0 and img.shape[1] > 0)

    def _needs_reset(self):
        if self._consec_fail >= self.fail_reset_threshold: return True
        if self._last_good_ts and (time.time() - self._last_good_ts) > self.max_no_good_secs: return True
        return False

    def read(self):
        try:
            if "RGB" in self.mode: self.cam.read_RGB()
            if "DEPTH" in self.mode: self.cam.read_depth(dataMode='m')
            rgb = getattr(self.cam, "image_buffer_RGB", None) if "RGB" in self.mode else None
            dep = getattr(self.cam, "image_buffer_depth_m", None) if "DEPTH" in self.mode else None
            ok_rgb = (not "RGB" in self.mode) or self._is_valid(rgb)
            ok_dep = (not "DEPTH" in self.mode) or self._is_valid_depth(dep)
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
        self._log("Force reset."); self._init_cam()

    def terminate(self):
        if self.cam is not None:
            try: self._safe_terminate(self.cam)
            except Exception: pass
            self.cam = None
            self._log("Camera terminated.")

# ---------------- FSM ----------------
class State(enum.Enum):
    DRIVE = 0
    STOPPING = 1
    TURN_LEFT = 2   # hardcoded open-loop

def main():
    car = QCar()
    cam3d = SafeCamera3D(mode='RGB&DEPTH', frame_width=1280, frame_height=720, frame_rate=20.0,
                         device_id='0', fail_reset_threshold=8, max_no_good_secs=2.5, verbose=True)
    right_cam = Camera2D(camera_id=RIGHT_CAM_ID, frame_width=FRAME_W, frame_height=FRAME_H, frame_rate=FPS_RIGHT)

    show_right_always = SHOW_RIGHT_ALWAYS_DEFAULT

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.namedWindow(RIGHT_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 960, 540)
    cv2.resizeWindow(RIGHT_WINDOW, 640, 480)
    cv2.moveWindow(WINDOW, 50, 50)
    cv2.moveWindow(RIGHT_WINDOW, 1050, 50)

    # Boot a frame on RightCam if available (just for preview)
    try:
        right_cam.read()
        if right_cam.image_data is not None and right_cam.image_data.size > 0:
            cv2.imshow(RIGHT_WINDOW, right_cam.image_data); cv2.waitKey(1)
    except Exception:
        print(f"[RightCam] id:{RIGHT_CAM_ID} NO SIGNAL at startup")

    state = State.DRIVE
    stop_t0 = 0.0
    turn_t0 = 0.0   # start time of hardcoded turn

    frame_count, fps, last_time = 0, 0, time.time()

    try:
        while True:
            loop_t0 = time.time()
            rgb, depth = cam3d.read()
            if rgb is None:
                key = (cv2.waitKey(1) & 0xFF)
                if key == 27: break
                continue

            hF, wF, _ = rgb.shape
            disp = rgb.copy()

            # Optional RightCam preview when not turning
            if show_right_always and state == State.DRIVE:
                try:
                    right_cam.read()
                    rc = right_cam.image_data
                    if rc is not None and rc.size > 0:
                        rc_vis = rc.copy()
                        infoR = find_right_edge_overlay(rc_vis)
                        if infoR is not None:
                            x0, y0, ww, hh = infoR["crop_box"]
                            cv2.rectangle(rc_vis, (x0, y0), (x0 + ww - 1, y0 + hh - 1), (0, 255, 255), 2)
                            cv2.drawContours(rc_vis, [infoR["contour"]], -1, (255, 0, 0), 2)
                            cv2.circle(rc_vis, infoR["centroid"], 8, (255, 0, 0), -1)
                        cv2.putText(rc_vis, 'INACTIVE: FRONT DRIVE', (10, 22),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                        cv2.imshow(RIGHT_WINDOW, rc_vis)
                except Exception:
                    pass

            # ==================== FSM ====================
            if state == State.DRIVE:
                # Stop-sign detection
                stop_seen, stop_box, stop_dist = detect_stop_sign(rgb, depth)
                if stop_seen:
                    state = State.STOPPING
                    stop_t0 = time.time()
                    x,y,ww,hh = stop_box
                    cv2.rectangle(disp, (x,y), (x+ww,y+hh), (0,255,0), 3)
                    cv2.putText(disp, f"STOP {stop_dist:.2f}m", (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                    drive_speed = 0.0
                    drive_steer = 0.0
                else:
                    # Yellow-line follower (front)
                    info = get_line_info_bottom(rgb)
                    steering = 0.0
                    if info is not None:
                        desired_x = wF - TARGET_OFFSET_RIGHT
                        error = desired_x - info["cx_full"]
                        steering = float(np.clip(error * STEER_GAIN, -STEER_CLIP, STEER_CLIP))
                        (rx0, ry0) = info["roi_origin"]; (rw, rh) = info["roi_size"]
                        cv2.rectangle(disp, (rx0, ry0), (rx0+rw-1, ry0+rh-1), (0,255,0), 2)
                        cv2.drawContours(disp, [info["contour"]], -1, (255,0,0), 2)
                        cv2.circle(disp, info["centroid"], 7, (255,0,0), -1)
                        cv2.circle(disp, (desired_x, ry0 + rh//2), 7, (0,0,255), -1)
                        cv2.rectangle(disp, (rx0, info["band_y_start_full"]),
                                      (rx0 + rw - 1, ry0 + rh - 1), (0, 200, 200), 2)
                    else:
                        cv2.putText(disp, "NO YELLOW", (10, 140),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

                    drive_speed = SPEED_BASE
                    drive_steer = steering

            elif state == State.STOPPING:
                drive_speed = 0.0
                drive_steer = 0.0
                elapsed = time.time() - stop_t0
                cv2.putText(disp, f"STOPPING {elapsed:.1f}/{STOP_DURATION:.1f}s",
                            (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,200,255), 2)
                if elapsed >= STOP_DURATION:
                    # Go into hardcoded left turn
                    print("[FSM] STOP complete -> HARD TURN LEFT")
                    state = State.TURN_LEFT
                    turn_t0 = time.time()

            elif state == State.TURN_LEFT:
                # ---------- Hardcoded open-loop left turn ----------
                elapsed_turn = time.time() - turn_t0

                if elapsed_turn < INIT_TURN_TIME:
                    drive_speed = float(INIT_TURN_SPEED)
                    # Clamp steering to [-0.5, 0.5] just in case
                    drive_steer = float(np.clip(INIT_TURN_STEERING, -0.5, 0.5))

                    cv2.putText(disp, f"TURN LEFT t={elapsed_turn:.1f}/{INIT_TURN_TIME:.1f}s",
                                (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,200,255), 2)
                else:
                    # Done turning: go back to DRIVE, let front line follower take over
                    print("[FSM] Hard turn finished -> DRIVE")
                    state = State.DRIVE
                    drive_speed = SPEED_BASE
                    drive_steer = 0.0

            # ---------------- HUD + Drive ----------------
            frame_count += 1
            now = time.time()
            if now - last_time >= 1.0:
                fps = frame_count; frame_count = 0; last_time = now
            calc_ms = (time.time() - loop_t0) * 1000.0

            angle_deg = drive_steer * MAX_STEER_ANGLE_DEG
            cv2.putText(disp, f'FPS:{fps}  Calc:{calc_ms:.1f} ms  State:{state.name}',
                        (HUD_X, HUD_Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.putText(disp, f'Speed:{drive_speed:.3f}  Steer:{drive_steer:+.3f}  Angle:{angle_deg:+.1f} deg',
                        (HUD_X, HUD_Y + HUD_DY), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

            cv2.imshow(WINDOW, disp)

            key = (cv2.waitKey(1) & 0xFF)
            if key == 27:  # ESC
                print("ESC -> exit"); break
            elif key in (ord('r'), ord('R')):
                print("R -> camera reset"); cam3d.force_reset()
            elif key in (ord('v'), ord('V')):
                show_right_always = not show_right_always
                print("Right-cam always-on:", show_right_always)
            elif key in (ord('t'), ord('T')):
                # manual safety: immediately go back to DRIVE
                print("Manual override: exit TURN_* to DRIVE")
                state = State.DRIVE

            # Drive car
            mtr_cmd = np.array([drive_speed, drive_steer], dtype=np.float64)
            LEDs = np.array([0,0,0,0, 0,0,1,1], dtype=np.float64)
            try:
                car.read_write_std(mtr_cmd, LEDs)
            except Exception:
                pass

            time.sleep(0.02)

    finally:
        try: cv2.destroyAllWindows()
        except Exception: pass
        try: car.terminate()
        except Exception: pass
        try: cam3d.terminate()
        except Exception: pass
        try: right_cam.terminate()
        except Exception: pass

if __name__ == "__main__":
    main()
