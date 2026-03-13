#!/usr/bin/env python3
# QCar encoder logger with:
#   - line follow mode
#   - hardcoded left turn mode
#   - hardcoded right turn mode
#
# Controls:
#   F     : select LINE FOLLOW mode
#   W     : select LINE FOLLOW + WHITE STOP mode
#   L     : select LEFT TURN mode
#   R     : select RIGHT TURN mode
#   S     : start run
#   X     : stop run and save summary
#   Q     : neutral / stop motors
#   T     : reset odometry baseline
#   C     : reset camera
#   ESC   : quit

from Quanser.product_QCar import QCar
from Quanser.q_essential import Camera3D
import numpy as np
import cv2
import time
from math import pi
import enum

# ===================== Line-follow tunables =====================
BOTTOM_FRAC = 0.40
BAND_FRAC = 0.20
MIN_BAND_PTS = 30

TARGET_OFFSET_RIGHT = 1000
SPEED_BASE = 0.078
STEER_GAIN = 0.0012
STEER_CLIP = 0.5
MAX_STEER_ANGLE_DEG = 28.0

KERNEL5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# ===================== Horizontal white stop tunables =====================
WHITE_STOP_DIST_M = 11.69 * 0.0254  # 11.69 inches
WHITE_ROI_TOP_FRAC = 0.45
WHITE_MIN_AREA_PX = 650
WHITE_MIN_ASPECT = 2.0
WHITE_MIN_WIDTH_FRAC = 0.25
WHITE_BOTTOM_MARGIN_PX = 3

# ===================== Hardcoded turns =====================
LEFT_TURN_SPEED = 0.075
LEFT_TURN_STEER = 0.16

RIGHT_TURN_SPEED = 0.075
RIGHT_TURN_STEER = -0.21

# ===================== Encoder / odometry params =====================
TICKS_PER_REV = 31844.0
WHEEL_DIAM_M = 0.066
WHEEL_CIRC_M = pi * WHEEL_DIAM_M

EMA_ALPHA = 0.35
DT_MIN = 0.004
DT_MAX = 0.35
MPS_HARD_MAX = 6.0

# ===================== Logging =====================
SAMPLES_LOG = "segment_encoder_samples.txt"
SUMMARY_LOG = "segment_encoder_summary.txt"
TEST_LABEL = "segment_encoder_v2"

SAMPLE_DT = 0.05
SETTLE_S = 1.2

# ===================== Display =====================
WINDOW = "QCar Segment Encoder Logger"
HUD_X, HUD_Y, HUD_DY = 10, 30, 30


class RunMode(enum.Enum):
    LINE_FOLLOW = 0
    LINE_FOLLOW_WHITE_STOP = 1
    LEFT_TURN = 2
    RIGHT_TURN = 3


# ---------------------------------------------------------------------
# Camera wrapper
# ---------------------------------------------------------------------
class SafeCamera3D:
    def __init__(
        self,
        mode='RGB',
        frame_width=1280,
        frame_height=720,
        frame_rate=20.0,
        device_id='0',
        fail_reset_threshold=8,
        max_no_good_secs=2.5,
        verbose=True
    ):
        self.mode = mode
        self.w = frame_width
        self.h = frame_height
        self.fps = frame_rate
        self.dev = device_id
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
            rgb = getattr(self.cam, "image_buffer_RGB", None)
            if self._is_valid(rgb):
                self._last_good_ts = time.time()
                self._log("Warm-up OK.")
        except Exception:
            self._log("Warm-up read failed (will recover).")

    @staticmethod
    def _is_valid(img):
        return (
            img is not None
            and hasattr(img, "shape")
            and len(img.shape) == 3
            and img.shape[0] > 0
            and img.shape[1] > 0
        )

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

            rgb = getattr(self.cam, "image_buffer_RGB", None)
            if self._is_valid(rgb):
                self._consec_fail = 0
                self._last_good_ts = time.time()
                return rgb
            else:
                self._consec_fail += 1
                if self._needs_reset():
                    self._log("Invalid RGB frames. Resetting stream...")
                    self._init_cam()
        except Exception as e:
            self._consec_fail += 1
            self._log(f"read() exception: {e}")
            if self._needs_reset():
                self._log("Exceptions persisted. Resetting stream...")
                self._init_cam()

        return None

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


# ---------------------------------------------------------------------
# Yellow line follower helpers
# ---------------------------------------------------------------------
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
    if cv2.contourArea(largest) < 50:
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


def detect_horizontal_white_line_bottom(image_bgr):
    """Detect horizontal white contour near the bottom of the frame."""
    h, w, _ = image_bgr.shape
    y0 = int(h * WHITE_ROI_TOP_FRAC)
    roi = image_bgr[y0:h, 0:w]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 185), (179, 55, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL5, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL5, iterations=1)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_bbox = None
    best_area = 0.0

    for c in cnts:
        area = cv2.contourArea(c)
        if area < WHITE_MIN_AREA_PX:
            continue

        x, y, cw, ch = cv2.boundingRect(c)
        if ch <= 0:
            continue

        aspect = float(cw) / float(ch)
        if aspect < WHITE_MIN_ASPECT:
            continue

        if area > best_area:
            best_area = area
            best_bbox = (x, y + y0, cw, ch)

    if best_bbox is None:
        return {
            "detected": False,
            "is_decently_long": False,
            "touches_bottom": False,
            "trigger": False,
            "bbox": None,
        }

    x, y, cw, ch = best_bbox
    is_decently_long = cw >= int(WHITE_MIN_WIDTH_FRAC * w)
    touches_bottom = (y + ch) >= (h - WHITE_BOTTOM_MARGIN_PX)

    return {
        "detected": True,
        "is_decently_long": is_decently_long,
        "touches_bottom": touches_bottom,
        "trigger": bool(is_decently_long and touches_bottom),
        "bbox": best_bbox,
    }


# ---------------------------------------------------------------------
# Encoder / odometry
# ---------------------------------------------------------------------
def read_ticks(qcar) -> float:
    return float(qcar.read_encoder())


def neutral(qcar):
    try:
        qcar.read_write_std(
            np.array([0.0, 0.0], dtype=np.float64),
            np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
        )
    except Exception:
        pass


class EncoderOdom:
    def __init__(self, alpha=EMA_ALPHA):
        self.alpha = alpha
        self.v_filt = 0.0
        self.prev_ticks = None
        self.prev_t = None
        self.start_ticks = None
        self.total_ticks = 0.0
        self.total_dist = 0.0

    def reset(self, qcar):
        ticks_now = read_ticks(qcar)
        now = time.time()

        self.prev_ticks = ticks_now
        self.prev_t = now
        self.start_ticks = ticks_now
        self.v_filt = 0.0
        self.total_ticks = 0.0
        self.total_dist = 0.0

    def update(self, qcar):
        now = time.time()
        ticks_now = read_ticks(qcar)

        if self.prev_t is None or self.prev_ticks is None or self.start_ticks is None:
            self.prev_ticks = ticks_now
            self.prev_t = now
            self.start_ticks = ticks_now
            return {
                "ticks_now": ticks_now,
                "d_ticks": 0.0,
                "ticks_from_start": 0.0,
                "d_dist": 0.0,
                "total_dist": 0.0,
                "v_raw": 0.0,
                "v_filt": 0.0,
                "dt": 0.0
            }

        dt = max(1e-3, now - self.prev_t)
        d_ticks = ticks_now - self.prev_ticks

        self.prev_ticks = ticks_now
        self.prev_t = now

        d_dist = (d_ticks / TICKS_PER_REV) * WHEEL_CIRC_M
        v_raw = d_dist / dt

        if dt < DT_MIN or dt > DT_MAX or abs(v_raw) > MPS_HARD_MAX:
            v_raw = self.v_filt

        self.v_filt = self.alpha * v_raw + (1.0 - self.alpha) * self.v_filt
        self.total_ticks = ticks_now - self.start_ticks
        self.total_dist += d_dist

        return {
            "ticks_now": ticks_now,
            "d_ticks": d_ticks,
            "ticks_from_start": self.total_ticks,
            "d_dist": d_dist,
            "total_dist": self.total_dist,
            "v_raw": v_raw,
            "v_filt": self.v_filt,
            "dt": dt
        }


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
def ensure_logs():
    try:
        with open(SAMPLES_LOG, "x") as f:
            f.write("# samples: ts  t_rel  mode  cmd  steer  ticks_now  d_ticks  ticks_from_start  d_dist  total_dist  v_raw  v_filt\n")
            f.write(f"# test={TEST_LABEL}\n\n")
    except FileExistsError:
        with open(SAMPLES_LOG, "a") as f:
            f.write("\n# --- APPEND RUN ---\n")

    try:
        with open(SUMMARY_LOG, "x") as f:
            f.write("# summary: ts  mode  total_ticks  total_dist  v_mean  v_std  N\n")
            f.write(f"# test={TEST_LABEL}\n\n")
    except FileExistsError:
        with open(SUMMARY_LOG, "a") as f:
            f.write("\n# --- APPEND RUN ---\n")


def mode_name(mode: RunMode) -> str:
    if mode == RunMode.LINE_FOLLOW:
        return "LINE_FOLLOW"
    if mode == RunMode.LINE_FOLLOW_WHITE_STOP:
        return "LINE_FOLLOW_WHITE_STOP"
    if mode == RunMode.LEFT_TURN:
        return "LEFT_TURN"
    if mode == RunMode.RIGHT_TURN:
        return "RIGHT_TURN"
    return "UNKNOWN"


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    ensure_logs()

    car = QCar()
    cam3d = SafeCamera3D(
        mode='RGB',
        frame_width=1280,
        frame_height=720,
        frame_rate=20.0,
        device_id='0',
        fail_reset_threshold=8,
        max_no_good_secs=2.5,
        verbose=True
    )

    odo = EncoderOdom()
    odo.reset(car)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 960, 540)

    run_mode = RunMode.LINE_FOLLOW
    running = False
    run_t0 = None
    program_t0 = time.time()
    last_sample_ts = 0.0
    summary_speed_samples = []

    frame_count = 0
    fps = 0
    last_fps_t = time.time()

    white_stop_triggered = False
    white_stop_start_dist_m = 0.0
    white_stop_progress_m = 0.0

    print("""
Controls:
  F     : select LINE FOLLOW mode
    W     : select LINE FOLLOW + WHITE STOP mode
  L     : select LEFT TURN mode
  R     : select RIGHT TURN mode
  S     : start run
  X     : stop run and save summary
  Q     : neutral / stop motors
  T     : reset odometry baseline
  C     : reset camera
  ESC   : quit
""")

    try:
        while True:
            loop_t0 = time.time()

            rgb = cam3d.read()
            enc = odo.update(car)

            if rgb is None:
                neutral(car)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break
                continue

            disp = rgb.copy()
            hF, wF, _ = rgb.shape

            drive_speed = 0.0
            drive_steer = 0.0
            info = get_line_info_bottom(rgb)
            white_det = {"trigger": False, "bbox": None}

            if running:
                if run_mode in (RunMode.LINE_FOLLOW, RunMode.LINE_FOLLOW_WHITE_STOP):
                    if info is not None:
                        desired_x = wF - TARGET_OFFSET_RIGHT
                        error = desired_x - info["cx_full"]
                        steering = float(np.clip(error * STEER_GAIN, -STEER_CLIP, STEER_CLIP))

                        drive_speed = SPEED_BASE
                        drive_steer = steering

                        (rx0, ry0) = info["roi_origin"]
                        (rw, rh) = info["roi_size"]

                        cv2.rectangle(disp, (rx0, ry0), (rx0 + rw - 1, ry0 + rh - 1), (0, 255, 0), 2)
                        cv2.drawContours(disp, [info["contour"]], -1, (255, 0, 0), 2)
                        cv2.circle(disp, info["centroid"], 7, (255, 0, 0), -1)
                        cv2.circle(disp, (desired_x, ry0 + rh // 2), 7, (0, 0, 255), -1)
                        cv2.rectangle(
                            disp,
                            (rx0, info["band_y_start_full"]),
                            (rx0 + rw - 1, ry0 + rh - 1),
                            (0, 200, 200),
                            2
                        )
                    else:
                        drive_speed = 0.0
                        drive_steer = 0.0
                        cv2.putText(
                            disp, "NO YELLOW -> STOP",
                            (10, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2
                        )

                    if run_mode == RunMode.LINE_FOLLOW_WHITE_STOP:
                        white_det = detect_horizontal_white_line_bottom(rgb)

                        if white_det["bbox"] is not None:
                            x, y, ww, hh = white_det["bbox"]
                            cv2.rectangle(disp, (x, y), (x + ww, y + hh), (255, 255, 255), 2)

                        if (not white_stop_triggered) and white_det["trigger"]:
                            white_stop_triggered = True
                            white_stop_start_dist_m = enc["total_dist"]
                            white_stop_progress_m = 0.0
                            print(
                                f"[WhiteStop] Triggered at total_dist={white_stop_start_dist_m:.3f} m. "
                                f"Advancing {WHITE_STOP_DIST_M:.3f} m then stopping."
                            )

                        if white_stop_triggered:
                            # Keep moving forward after trigger, even if yellow is lost.
                            white_stop_progress_m += abs(enc["d_dist"])
                            if white_stop_progress_m >= WHITE_STOP_DIST_M:
                                drive_speed = 0.0
                                drive_steer = 0.0
                                running = False
                                neutral(car)
                                print(
                                    f"[WhiteStop] Complete: advanced {white_stop_progress_m:.3f} m "
                                    f"after trigger. Vehicle stopped."
                                )
                            else:
                                drive_speed = SPEED_BASE
                                drive_steer = 0.0

                        status_text = "ARMED" if not white_stop_triggered else "ADVANCING"
                        cv2.putText(
                            disp,
                            f"WHITE STOP: {status_text}  trigger={white_det['trigger']}  "
                            f"d={white_stop_progress_m:.3f}/{WHITE_STOP_DIST_M:.3f} m",
                            (10, 170),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2
                        )

                elif run_mode == RunMode.LEFT_TURN:
                    drive_speed = LEFT_TURN_SPEED
                    drive_steer = LEFT_TURN_STEER
                    cv2.putText(
                        disp, "MODE: LEFT TURN",
                        (10, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2
                    )

                elif run_mode == RunMode.RIGHT_TURN:
                    drive_speed = RIGHT_TURN_SPEED
                    drive_steer = RIGHT_TURN_STEER
                    cv2.putText(
                        disp, "MODE: RIGHT TURN",
                        (10, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2
                    )

            else:
                if info is not None:
                    (rx0, ry0) = info["roi_origin"]
                    (rw, rh) = info["roi_size"]
                    cv2.rectangle(disp, (rx0, ry0), (rx0 + rw - 1, ry0 + rh - 1), (120, 120, 120), 1)
                    cv2.drawContours(disp, [info["contour"]], -1, (120, 120, 120), 1)
                    cv2.circle(disp, info["centroid"], 5, (120, 120, 120), -1)

            mtr_cmd = np.array([drive_speed, drive_steer], dtype=np.float64)
            LEDs = np.array([0, 0, 0, 0, 0, 0, 1 if running else 0, 1 if running else 0], dtype=np.float64)

            try:
                car.read_write_std(mtr_cmd, LEDs)
            except Exception:
                neutral(car)

            now = time.time()
            t_rel = now - program_t0

            if running and (now - last_sample_ts) >= SAMPLE_DT:
                last_sample_ts = now

                with open(SAMPLES_LOG, "a") as f:
                    f.write(
                        f"{now:.6f}\t{t_rel:.3f}\t{mode_name(run_mode)}\t{drive_speed:.3f}\t{drive_steer:+.6f}\t"
                        f"{enc['ticks_now']:.3f}\t{enc['d_ticks']:.3f}\t{enc['ticks_from_start']:.3f}\t"
                        f"{enc['d_dist']:.6f}\t{enc['total_dist']:.6f}\t"
                        f"{enc['v_raw']:.6f}\t{enc['v_filt']:.6f}\n"
                    )

                if run_t0 is not None and (now - run_t0) >= SETTLE_S:
                    summary_speed_samples.append(enc["v_filt"])

            frame_count += 1
            if now - last_fps_t >= 1.0:
                fps = frame_count
                frame_count = 0
                last_fps_t = now

            calc_ms = (time.time() - loop_t0) * 1000.0
            angle_deg = drive_steer * MAX_STEER_ANGLE_DEG

            cv2.putText(
                disp,
                f"FPS:{fps}  Calc:{calc_ms:.1f} ms  Running:{running}",
                (HUD_X, HUD_Y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
            cv2.putText(
                disp,
                f"Selected Mode:{mode_name(run_mode)}",
                (HUD_X, HUD_Y + HUD_DY),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2
            )
            cv2.putText(
                disp,
                f"Cmd:{drive_speed:.3f}  Steer:{drive_steer:+.3f}  Angle:{angle_deg:+.1f} deg",
                (HUD_X, HUD_Y + 2 * HUD_DY),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2
            )
            cv2.putText(
                disp,
                f"Ticks now:{enc['ticks_now']:.1f}  dTicks:{enc['d_ticks']:.1f}  RunTicks:{enc['ticks_from_start']:.1f}",
                (HUD_X, HUD_Y + 3 * HUD_DY),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2
            )
            cv2.putText(
                disp,
                f"dDist:{enc['d_dist']:.4f} m  TotalDist:{enc['total_dist']:.3f} m",
                (HUD_X, HUD_Y + 4 * HUD_DY),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2
            )
            cv2.putText(
                disp,
                f"v_raw:{enc['v_raw']:+.3f} m/s  v_filt:{enc['v_filt']:+.3f} m/s",
                (HUD_X, HUD_Y + 5 * HUD_DY),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 180, 0), 2
            )

            if running and run_t0 is not None:
                cv2.putText(
                    disp,
                    f"Run t:{now - run_t0:.2f}s  Settle:{SETTLE_S:.1f}s",
                    (HUD_X, HUD_Y + 6 * HUD_DY),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 2
                )

            cv2.putText(
                disp,
                "Keys: F=line  W=line+white-stop  L=left  R=right  S=start  X=stop  Q=neutral  T=reset odo  C=cam  ESC=quit",
                (HUD_X, HUD_Y + 7 * HUD_DY),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 2
            )

            cv2.imshow(WINDOW, disp)

            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                break

            elif key in (ord('q'), ord('Q')):
                running = False
                neutral(car)
                print("[Neutral] Stopped motors.")

            elif key in (ord('c'), ord('C')):
                print("[Camera] Force reset.")
                cam3d.force_reset()

            elif key in (ord('t'), ord('T')):
                odo.reset(car)
                print("[Odom] Reset baseline ticks and distance.")

            elif key in (ord('f'), ord('F')):
                if not running:
                    run_mode = RunMode.LINE_FOLLOW
                    white_stop_triggered = False
                    white_stop_start_dist_m = 0.0
                    white_stop_progress_m = 0.0
                    print("[Mode] LINE_FOLLOW")

            elif key in (ord('w'), ord('W')):
                if not running:
                    run_mode = RunMode.LINE_FOLLOW_WHITE_STOP
                    white_stop_triggered = False
                    white_stop_start_dist_m = 0.0
                    white_stop_progress_m = 0.0
                    print("[Mode] LINE_FOLLOW_WHITE_STOP")

            elif key in (ord('l'), ord('L')):
                if not running:
                    run_mode = RunMode.LEFT_TURN
                    white_stop_triggered = False
                    white_stop_start_dist_m = 0.0
                    white_stop_progress_m = 0.0
                    print("[Mode] LEFT_TURN")

            elif key in (ord('r'), ord('R')):
                if not running:
                    run_mode = RunMode.RIGHT_TURN
                    white_stop_triggered = False
                    white_stop_start_dist_m = 0.0
                    white_stop_progress_m = 0.0
                    print("[Mode] RIGHT_TURN")

            elif key in (ord('s'), ord('S')):
                if not running:
                    odo.reset(car)
                    summary_speed_samples = []
                    run_t0 = time.time()
                    white_stop_triggered = False
                    white_stop_start_dist_m = 0.0
                    white_stop_progress_m = 0.0
                    running = True
                    print(f"[Run] START mode={mode_name(run_mode)}")

            elif key in (ord('x'), ord('X')):
                if running:
                    running = False
                    neutral(car)

                    if summary_speed_samples:
                        arr = np.asarray(summary_speed_samples, dtype=float)
                        v_mean = float(arr.mean())
                        v_std = float(arr.std(ddof=0))
                        n = int(arr.size)
                    else:
                        v_mean = 0.0
                        v_std = 0.0
                        n = 0

                    with open(SUMMARY_LOG, "a") as f:
                        f.write(
                            f"{time.time():.6f}\t{mode_name(run_mode)}\t"
                            f"{enc['ticks_from_start']:.3f}\t{enc['total_dist']:.6f}\t"
                            f"{v_mean:.6f}\t{v_std:.6f}\t{n}\n"
                        )

                    print(
                        f"[Run] STOP mode={mode_name(run_mode)} -> "
                        f"total_ticks={enc['ticks_from_start']:.1f}, "
                        f"total_dist={enc['total_dist']:.3f} m, "
                        f"v_mean={v_mean:.3f}, v_std={v_std:.3f}, N={n}"
                    )

            time.sleep(0.005)

    except KeyboardInterrupt:
        pass

    finally:
        try:
            neutral(car)
        except Exception:
            pass
        try:
            car.terminate()
        except Exception:
            pass
        try:
            cam3d.terminate()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        print(f"\nLogs written:\n  {SAMPLES_LOG}\n  {SUMMARY_LOG}\nBye!")


if __name__ == "__main__":
    main()
