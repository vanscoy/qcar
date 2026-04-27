#!/usr/bin/env python3
# QCar yellow-line follower with encoder distance
#
# Goal:
# - Follow the yellow line using your real Camera3D setup
# - Continuously show encoder distance traveled
# - Press S to start following
# - Press X to stop and save the run
# - Repeat as many times as you want
#
# Controls:
#   S : start yellow follow
#   X : stop yellow follow and save result
#   R : reset odometry
#   A : move target 20 px more left
#   D : move target 20 px more right
#   Z : decrease speed by 0.002
#   C : increase speed by 0.002
#   Q : neutral stop
#   ESC : quit

from Quanser.product_QCar import QCar
from Quanser.q_essential import Camera3D
import cv2, time, numpy as np
import threading, os, sys
from math import pi

# ========================= Tunables =========================
HEADLESS = False

# Vision ROI
bottom_frac = 0.40
band_frac   = 0.20
MIN_BAND_PTS = 30
MIN_CONTOUR_AREA = 50

# Control
target_offset_right = 1000
speed = 0.078
steering_gain = 0.0012
max_steering_angle = 28.0
STEER_CMD_CLIP = 0.5

# Camera
FRAME_W, FRAME_H, FRAME_FPS = 1280, 720, 20.0

# Morphology
KERNEL = np.ones((5,5), np.uint8)

# Watchdogs
MAX_FRAME_AGE_S   = 0.25
STALE_MAX_FRAMES  = 8
MAX_LOOP_TIME_S   = 0.20
RESET_COOLDOWN_S  = 1.0
MAX_ESCALATION    = 3

# Encoder / odometry
TICKS_PER_REV = 31844.0
WHEEL_DIAM_M  = 0.066
WHEEL_CIRC_M  = pi * WHEEL_DIAM_M
EMA_ALPHA     = 0.35
DT_MIN, DT_MAX = 0.004, 0.35
MPS_HARD_MAX  = 6.0

# Logging
WINDOW = "Yellow Follow + Distance"
MASK_WINDOW = "Yellow Mask"
LOG_FILE = "yellow_follow_distance_log.txt"

frame_count, fps, last_time = 0, 0, time.time()

# ========================= Safe camera stop =========================
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

# ========================= Frame grabber =========================
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

# ========================= QCar helpers =========================
def neutral_brake(car):
    try:
        car.read_write_std(
            np.array([0.0, 0.0], dtype=np.float64),
            np.array([1,0,0,0, 1,0,0,0], dtype=np.float64)
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

# ========================= Odometry =========================
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

# ========================= Yellow detection =========================
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
    }, mask

# ========================= Logging =========================
def ensure_log():
    try:
        with open(LOG_FILE, "x") as f:
            f.write("# ts\tspeed\ttarget_offset_right\tsteering_gain\trun_time_s\tdist_m\tmean_v\tfinal_v\n")
    except FileExistsError:
        pass

# ========================= Main =========================
def main():
    global frame_count, fps, last_time, speed, target_offset_right

    ensure_log()

    myCar = QCar()
    odo = SpeedOdom()
    odo.reset(myCar)

    grab = FrameGrabber()
    grab.start()

    running = False
    run_t0 = None
    speed_samples = []

    if not HEADLESS:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.namedWindow(MASK_WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, 1280, 720)
        cv2.resizeWindow(MASK_WINDOW, 800, 240)

    print("""
Controls:
  S : start yellow follow
  X : stop yellow follow and save result
  R : reset odometry
  A : target 20 px more left
  D : target 20 px more right
  Z : speed -0.002
  C : speed +0.002
  Q : neutral stop
  ESC : quit
""")

    try:
        escalation = 0
        while True:
            cycle_start = time.time()
            now = time.time()

            v_raw, v_filt, dist_m, dt = odo.update(myCar)

            img = grab.get_frame()
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
                        odo.reset(myCar)
                    escalation = 0
                time.sleep(0.02)
                continue
            else:
                escalation = 0

            display = img.copy()
            info, mask = get_line_info_bottom(img)

            frame_count += 1
            if now - last_time >= 1.0:
                fps = frame_count
                frame_count = 0
                last_time = now

            steering = 0.0
            found = False
            status = "NO YELLOW DETECTED"

            if info is not None:
                found = True
                h, w, _ = img.shape
                desired_x = w - target_offset_right
                error = desired_x - info["cx_full"]
                steering = float(np.clip(error * steering_gain, -STEER_CMD_CLIP, STEER_CMD_CLIP))
                status = f"FOUND | err={error:+.1f} steer={steering:+.3f}"

                if not HEADLESS:
                    (rx0, ry0) = info["roi_origin"]
                    (rw, rh) = info["roi_size"]
                    cv2.rectangle(display, (rx0, ry0), (rx0+rw-1, ry0+rh-1), (0,255,0), 2)
                    cv2.drawContours(display, [info["contour"]], -1, (255,0,0), 2)
                    cv2.circle(display, info["centroid"], 8, (255,0,0), -1)
                    cv2.circle(display, (desired_x, ry0 + rh//2), 8, (0,0,255), -1)
                    cv2.rectangle(display, (rx0, info["band_y_start_full"]),
                                  (rx0 + rw - 1, ry0 + rh - 1), (0, 200, 200), 2)

            calc_ms = (time.time() - cycle_start) * 1000.0

            if running:
                try:
                    if found:
                        mtr_cmd = np.array([speed, steering], dtype=np.float64)
                        LEDs = np.array([0,0,0,0, 0,0,1,1], dtype=np.float64)
                        if (time.time() - cycle_start) > MAX_LOOP_TIME_S:
                            neutral_brake(myCar)
                        else:
                            myCar.read_write_std(mtr_cmd, LEDs)
                    else:
                        neutral_brake(myCar)
                    speed_samples.append(v_filt)
                except Exception:
                    neutral_brake(myCar)
                    newCar = reopen_qcar(myCar)
                    if newCar is None:
                        escalation += 1
                        if escalation >= MAX_ESCALATION:
                            os.execv(sys.executable, [sys.executable] + sys.argv)
                    else:
                        myCar = newCar
                        odo.reset(myCar)
                        escalation = 0
            else:
                neutral_brake(myCar)

            if not HEADLESS:
                cv2.putText(display, f"Mode: {'RUNNING' if running else 'IDLE'}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
                cv2.putText(display, f"FPS:{fps}  Calc:{calc_ms:.1f} ms", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                cv2.putText(display, f"Status: {status}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
                cv2.putText(display, f"Speed cmd: {speed:.3f}", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
                cv2.putText(display, f"Steering: {steering:+.3f}  Angle:{steering*max_steering_angle:+.1f} deg", (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
                cv2.putText(display, f"Target offset right: {target_offset_right}", (10, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
                cv2.putText(display, f"Distance: {dist_m:.3f} m", (10, 210),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
                cv2.putText(display, f"v_raw: {v_raw:+.3f} m/s   v_filt: {v_filt:+.3f} m/s", (10, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,255), 2)

                cv2.imshow(WINDOW, display)
                cv2.imshow(MASK_WINDOW, mask)

                k = cv2.waitKey(1) & 0xFF
            else:
                time.sleep(0.01)
                k = -1

            if k == 27:
                break
            elif k in (ord('q'), ord('Q')):
                running = False
                neutral_brake(myCar)
                print("[Neutral] stopped")
            elif k in (ord('a'), ord('A')):
                target_offset_right = max(0, target_offset_right + 20)
                print(f"[Target offset right] {target_offset_right}")
            elif k in (ord('d'), ord('D')):
                target_offset_right = max(0, target_offset_right - 20)
                print(f"[Target offset right] {target_offset_right}")
            elif k in (ord('z'), ord('Z')):
                speed = max(0.0, round(speed - 0.002, 3))
                print(f"[Speed] {speed:.3f}")
            elif k in (ord('c'), ord('C')):
                speed = round(speed + 0.002, 3)
                print(f"[Speed] {speed:.3f}")
            elif k in (ord('r'), ord('R')):
                odo.reset(myCar)
                speed_samples = []
                run_t0 = None
                print("[Reset] odometry reset")
            elif k in (ord('s'), ord('S')):
                if not running:
                    odo.reset(myCar)
                    speed_samples = []
                    run_t0 = time.time()
                    running = True
                    print(f"[Run] START yellow follow | speed={speed:.3f} | target_offset_right={target_offset_right}")
            elif k in (ord('x'), ord('X')):
                if running:
                    running = False
                    neutral_brake(myCar)

                    run_time = time.time() - run_t0 if run_t0 is not None else 0.0
                    mean_v = float(np.mean(speed_samples)) if speed_samples else 0.0
                    final_v = float(v_filt)

                    with open(LOG_FILE, "a") as f:
                        f.write(
                            f"{time.time():.6f}\t"
                            f"{speed:.3f}\t"
                            f"{target_offset_right}\t"
                            f"{steering_gain:.6f}\t"
                            f"{run_time:.3f}\t"
                            f"{dist_m:.3f}\t"
                            f"{mean_v:.6f}\t"
                            f"{final_v:.6f}\n"
                        )

                    print(f"[Run] STOP yellow follow | speed={speed:.3f} | "
                          f"time={run_time:.2f}s | dist={dist_m:.3f}m")

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
            neutral_brake(myCar)
        except Exception:
            pass
        try:
            myCar.terminate()
        except Exception:
            pass
        print(f"\nSaved log: {LOG_FILE}")

if __name__ == "__main__":
    main()