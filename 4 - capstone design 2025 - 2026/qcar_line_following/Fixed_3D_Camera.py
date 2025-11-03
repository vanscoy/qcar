# QCar + Camera3D yellow-line follower (bottom 40% ROI, bottom 20% band)
# with non-blocking FrameGrabber, watchdogs, auto-recovery, and safe actuation.

from Quanser.product_QCar import QCar
from Quanser.q_essential import Camera3D
import cv2, time, numpy as np
import threading, os, sys

# ========================= Tunables =========================
# Vision ROI
bottom_frac = 0.40             # use bottom 40% of the image (full width)
band_frac   = 0.20             # centroid from bottom 20% of that ROI
MIN_BAND_PTS = 30              # fallback threshold for too-few points
MIN_CONTOUR_AREA = 50          # ignore tiny blobs

# Control target
target_offset_right = 1000     # your code uses desired_x = w - target_offset_right
speed = 0.085
steering_gain = 0.0012
max_steering_angle = 28.0

# Camera settings
FRAME_W, FRAME_H, FRAME_FPS = 1280, 720, 20.0

# Morphology
KERNEL = np.ones((5,5), np.uint8)
OPEN_ITERS = 1

# HSV (yellow) — from your original function
LOWER_YELLOW = np.array([15, 80, 80], dtype=np.uint8)
UPPER_YELLOW = np.array([40, 255, 255], dtype=np.uint8)

# Watchdogs / robustness
HEADLESS = False          # True = no OpenCV windows (safer when VNC/HDMI is flaky)
MAX_FRAME_AGE_S   = 0.25  # if no fresh frame for this long -> neutral + camera soft-reset
STALE_MAX_FRAMES  = 8     # same image repeated this many frames -> treat as stuck
MAX_LOOP_TIME_S   = 0.20  # if one control cycle exceeds this -> neutral (skip actuation)
RESET_COOLDOWN_S  = 1.0   # minimum seconds between camera reset attempts
MAX_ESCALATION    = 3     # after N failed recoveries -> restart process
# ============================================================

frame_count, fps, last_time = 0, 0, time.time()
WINDOW = 'Front 3D Camera View'

# ---------------------- Safe termination --------------------
def safe_terminate_camera3d(cam):
    """Avoid AttributeError when in RGB-only mode."""
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

# ---------------------- Non-blocking capture ----------------
class FrameGrabber:
    """Camera3D wrapper that captures frames on a background thread and
       auto-resets the camera if frames go stale/old."""
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
        self.cam = Camera3D(mode='RGB',
                            frame_width_RGB=FRAME_W,
                            frame_height_RGB=FRAME_H,
                            frame_rate_RGB=FRAME_FPS,
                            device_id='0')

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
        # lightweight frozen-frame detection via tiny thumbnail hash
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
                        # copy to decouple from the camera buffer
                        self.frame = img.copy()
                # else: treat as stale; let watchdog in get_frame handle it
            except Exception:
                # camera exception → allow soft reset in get_frame
                pass
            time.sleep(0.001)

    def get_frame(self):
        """Return latest frame or None. Attempt camera soft reset if stale/old."""
        too_old = (time.time() - self.last_good_t) > MAX_FRAME_AGE_S
        too_stale = self.stale_count >= STALE_MAX_FRAMES
        if too_old or too_stale:
            self._soft_reset()
        with self.lock:
            return None if self.frame is None else self.frame.copy()

# ---------------------- QCar helpers ------------------------
def neutral_brake(car):
    try:
        car.read_write_std(np.array([0.0, 0.0], dtype=np.float64),
                           np.array([1,0,0,0, 1,0,0,0], dtype=np.float64))
    except Exception:
        pass

def reopen_qcar(car):
    """Terminate and recreate QCar handle."""
    try:
        car.terminate()
    except Exception:
        pass
    try:
        return QCar()
    except Exception:
        return None

# ---------------------- Your yellow mask --------------------
def make_yellow_mask(roi):
    # 1) HSV thresholds: normal yellow AND a paler variant (shade)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower1 = (15,  90,  80)   # normal yellow (good saturation)
    upper1 = (45, 255, 255)
    lower2 = (15,  40,  60)   # pale yellow in shadow
    upper2 = (45, 255, 200)

    m1 = cv2.inRange(hsv, lower1, upper1)
    m2 = cv2.inRange(hsv, lower2, upper2)
    mask_hsv = cv2.bitwise_or(m1, m2)

    # 2) Kill near-white/glare (very bright + desaturated)
    white_glare = cv2.inRange(hsv, (0, 0, 220), (180, 60, 255))
    mask = cv2.bitwise_and(mask_hsv, cv2.bitwise_not(white_glare))

    # 3) Reinforce yellow using LAB 'b' channel (yellow has high b)
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    _, b_bin = cv2.threshold(lab[:, :, 2], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.bitwise_and(mask, b_bin)

    # 4) Clean up small specks
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL, iterations=1)
    return mask

# ---------------------- Your ROI logic ----------------------
def get_line_info_bottom(image):
    """Bottom-40% ROI; centroid computed from bottom-20% of that ROI.
       Returns dict or None."""
    h, w, _ = image.shape
    y0 = int(h * (1.0 - bottom_frac))       # ROI = bottom 40% of full frame
    roi = image[y0:h, 0:w]

    mask = make_yellow_mask(roi)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    largest = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(largest) < MIN_CONTOUR_AREA:
        return None

    # ---- Compute centroid using only the bottom band of the contour ----
    pts = largest.reshape(-1, 2)
    roi_h = roi.shape[0]
    band_y_start = int(roi_h * (1.0 - band_frac))  # bottom 20% start (ROI coords)
    band_pts = pts[pts[:, 1] >= band_y_start]

    if band_pts.shape[0] >= MIN_BAND_PTS:
        cx, cy = int(band_pts[:, 0].mean()), int(band_pts[:, 1].mean())
    else:
        # Fallback: use bottom-most N points if band is too sparse
        N = min(50, pts.shape[0])
        idx = np.argsort(pts[:, 1])[-N:]   # take lowest (largest y)
        sel = pts[idx]
        cx, cy = int(sel[:,0].mean()), int(sel[:,1].mean())

    # Map for drawing on full frame
    contour_full = largest + np.array([0, y0])    # whole contour (for overlay)
    centroid_full = (cx, y0 + cy)                 # centroid of bottom band

    return {
        "contour": contour_full,
        "centroid": centroid_full,
        "cx_full": cx,                   # NOTE: ROI spans full width, so this is full-frame X
        "roi_origin": (0, y0),
        "roi_size": (w, h - y0),
        "band_y_start_full": y0 + band_y_start
    }

# ========================= Main =============================
def main():
    global frame_count, fps, last_time

    # Create QCar and start grabber
    myCar = QCar()
    grab = FrameGrabber()
    grab.start()

    try:
        escalation = 0
        while True:
            cycle_start = time.time()

            # ---- Non-blocking get of the most recent frame ----
            img = grab.get_frame()
            if img is None:
                # No frame currently available: safe + escalate if prolonged
                neutral_brake(myCar)
                if (time.time() - grab.last_good_t) > 2.0:
                    escalation += 1
                if escalation >= MAX_ESCALATION:
                    # Try reopening QCar (if I/O wedged)
                    newCar = reopen_qcar(myCar)
                    if newCar is None:
                        # Hard reset of process (clears stubborn driver deadlocks)
                        os.execv(sys.executable, [sys.executable] + sys.argv)
                    else:
                        myCar = newCar
                    escalation = 0
                time.sleep(0.02)
                continue
            else:
                escalation = 0

            # ---- Perception + steering (your logic) ----
            t0 = time.time()
            info = get_line_info_bottom(img)
            h, w, _ = img.shape
            display = img.copy()

            frame_count += 1
            now = time.time()
            if now - last_time >= 1.0:
                fps = frame_count
                frame_count = 0
                last_time = now

            steering = 0.0
            if info is not None:
                desired_x = w - target_offset_right
                error = desired_x - info["cx_full"]
                steering = float(np.clip(error * steering_gain, -0.5, 0.5))

                (rx0, ry0) = info["roi_origin"]; (rw, rh) = info["roi_size"]
                # HUD
                if not HEADLESS:
                    cv2.rectangle(display, (rx0, ry0), (rx0+rw-1, ry0+rh-1), (0,255,0), 2)
                    cv2.drawContours(display, [info["contour"]], -1, (255,0,0), 2)
                    cv2.circle(display, info["centroid"], 8, (255,0,0), -1)
                    cv2.circle(display, (desired_x, ry0 + rh//2), 8, (0,0,255), -1)
                    cv2.rectangle(display, (rx0, info["band_y_start_full"]),
                                  (rx0 + rw - 1, ry0 + rh - 1), (0, 200, 200), 2)
            else:
                if not HEADLESS:
                    cv2.putText(display, "NO YELLOW DETECTED", (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                # Vision lost: neutral this cycle
                neutral_brake(myCar)

            calc_ms = (time.time() - t0) * 1000.0
            if not HEADLESS:
                cv2.putText(display, f'FPS:{fps}  Calc:{calc_ms:.1f} ms', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                cv2.putText(display, f'Steering:{steering:+.3f}  Gain:{steering_gain}', (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
                cv2.putText(display, f'Angle:{steering*max_steering_angle:+.1f} deg', (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,128,255), 2)

                # Display (optional; disable with HEADLESS=True for max robustness)
                cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
                cv2.imshow(WINDOW, display)
                if (cv2.waitKey(1) & 0xFF) == 27:
                    print("Kill switch activated: ESC pressed.")
                    break

            # ---- Rate watchdog (skip actuation if cycle overran) ----
            cycle_dt = time.time() - cycle_start
            try:
                if cycle_dt > MAX_LOOP_TIME_S:
                    neutral_brake(myCar)
                else:
                    mtr_cmd = np.array([speed, steering], dtype=np.float64)
                    LEDs = np.array([0,0,0,0, 0,0,1,1], dtype=np.float64)
                    myCar.read_write_std(mtr_cmd, LEDs)
            except Exception:
                # QCar I/O error — try to reopen
                neutral_brake(myCar)
                newCar = reopen_qcar(myCar)
                if newCar is None:
                    escalation += 1
                    if escalation >= MAX_ESCALATION:
                        os.execv(sys.executable, [sys.executable] + sys.argv)
                else:
                    myCar = newCar
                    escalation = 0

            time.sleep(0.01)

    finally:
        try: cv2.destroyAllWindows()
        except Exception: pass
        try: grab.stop()
        except Exception: pass
        try: myCar.terminate()
        except Exception: pass

# ------------------------- Run ------------------------------
if __name__ == "__main__":
    main()
