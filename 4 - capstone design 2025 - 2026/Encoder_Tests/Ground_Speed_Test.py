#!/usr/bin/env python3
# QCar manual straight-line speed test (no line following).
# You control start/stop per run. Steering locked straight (0.0).
# Logs per-sample and per-run averages (with settle-time discard).

from Quanser.product_QCar import QCar
import numpy as np
import cv2, time
from math import pi

# ====== Test range & timing ======
CMD_MIN      = 0.000
CMD_MAX      = 0.085       # your cap on the small oval
CMD_STEP     = 0.005
SETTLE_S     = 1.2         # discard this much time from start of each run
SAMPLE_DT    = 0.05        # ~20 Hz per-sample logging
HUD_FPS_CAP  = 60

# ====== Encoder/vehicle params ======
TICKS_PER_REV = 31844.0
WHEEL_DIAM_M  = 0.066
WHEEL_CIRC_M  = pi * WHEEL_DIAM_M
EMA_ALPHA     = 0.35
DT_MIN, DT_MAX = 0.004, 0.35
MPS_HARD_MAX  = 6.0

# ====== Logging ======
SAMPLES_LOG = "ground_straight_samples.txt"
SUMMARY_LOG = "ground_straight_summary.txt"
TEST_LABEL  = "ground_straight_v1"

# ====== HUD window ======
WINDOW = "Straight-Line Speed Test"

# ---------- Helpers ----------
def read_ticks(qcar) -> float:
    return float(qcar.read_encoder())

def neutral(qcar):
    try:
        qcar.read_write_std(np.array([0.0, 0.0], dtype=np.float64),
                            np.array([0,0,0,0, 0,0,0,0], dtype=np.float64))
    except Exception:
        pass

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

        # guard insane
        if dt < DT_MIN or dt > DT_MAX or abs(v) > MPS_HARD_MAX:
            v = self.v_filt

        self.total_dist += dist
        self.v_filt = self.alpha * v + (1.0 - self.alpha) * self.v_filt
        return v, self.v_filt, self.total_dist, dt

# ---------- File headers ----------
def ensure_logs():
    try:
        with open(SAMPLES_LOG, "x") as f:
            f.write("# samples: ts  t_rel  cmd  v_raw  v_filt  dist\n")
            f.write(f"# test={TEST_LABEL}\n\n")
    except FileExistsError:
        with open(SAMPLES_LOG, "a") as f:
            f.write("\n# --- APPEND RUN ---\n")
    try:
        with open(SUMMARY_LOG, "x") as f:
            f.write("# summary: ts  cmd  v_mean  v_std  N\n")
            f.write(f"# test={TEST_LABEL}\n\n")
    except FileExistsError:
        with open(SUMMARY_LOG, "a") as f:
            f.write("\n# --- APPEND RUN ---\n")

# ---------- Main ----------
def main():
    ensure_logs()
    q = QCar()
    odo = SpeedOdom(); odo.reset(q)

    # State
    current_cmd = CMD_MIN
    running = False
    run_t0 = None
    last_sample_ts = 0.0
    t_prog0 = time.time()
    samples_for_run = []

    # HUD canvas
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 640, 360)
    last_hud = 0.0

    print("""
Controls:
  A / D : Decrease / Increase command by 0.005
  S     : Start run at current command (steering locked = 0)
  X     : Stop run, save average (discards first SETTLE_S sec), neutral
  N     : Next command (+= 0.005), no immediate run
  Q     : Immediate neutral (safety)
  ESC   : Quit

Tips:
- Place car at start of straight, press S to start, let it roll.
- Press X near end of straight; script saves the run and brakes.
- Pick up car, return to start. Adjust cmd (A/D) or press N, then S again.
""")

    try:
        while True:
            loop_t0 = time.time()

            # Update odometry
            v_raw, v_filt, dist_m, dt = odo.update(q)

            # Actuation: steering locked straight (0.0)
            steer = 0.0
            cmd = float(np.clip(current_cmd, CMD_MIN, CMD_MAX))
            try:
                q.read_write_std(np.array([cmd if running else 0.0, steer], dtype=np.float64),
                                 np.array([0,0,0,0, 0,0, 1 if running else 0, 1 if running else 0], dtype=np.float64))
            except Exception:
                neutral(q)

            # Per-sample logging while running
            now = time.time()
            t_rel = now - t_prog0
            if running and (now - last_sample_ts) >= SAMPLE_DT:
                last_sample_ts = now
                with open(SAMPLES_LOG, "a") as f:
                    f.write(f"{now:.6f}\t{t_rel:.3f}\t{cmd:.3f}\t{v_raw:.6f}\t{v_filt:.6f}\t{dist_m:.3f}\n")
                # keep for summary if beyond settle
                if (now - run_t0) >= SETTLE_S:
                    samples_for_run.append(v_filt)

            # HUD (limit FPS)
            if now - last_hud >= 1.0 / HUD_FPS_CAP:
                last_hud = now
                canvas = np.zeros((360, 640, 3), dtype=np.uint8)
                def put(y, text, color=(0,255,255), scale=0.8):
                    cv2.putText(canvas, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)

                put(40,  f"Mode: {'RUNNING' if running else 'IDLE'}", (0,200,255))
                put(80,  f"Cmd: {cmd:.3f} (range {CMD_MIN:.3f}..{CMD_MAX:.3f}, step {CMD_STEP:.3f})", (0,255,0))
                put(120, f"v_raw:  {v_raw:+.3f} m/s", (255,255,0))
                put(150, f"v_filt: {v_filt:+.3f} m/s", (255,255,0))
                put(180, f"dist:   {dist_m:.2f} m", (255,255,0))
                if running:
                    put(220, f"Run t:  {now - run_t0:.2f}s (SETTLE {SETTLE_S:.1f}s)", (200,200,255))
                put(260, "Keys: A/D=cmd  S=start  X=stop+save  N=next  Q=neutral  ESC=quit", (200,200,200), 0.6)
                cv2.imshow(WINDOW, canvas)

            # Keyboard
            k = cv2.waitKey(1) & 0xFF
            if k == 27:  # ESC
                break
            elif k in (ord('q'), ord('Q')):
                running = False
                neutral(q)
                print("[Neutral] Stopped motors.")
            elif k in (ord('a'), ord('A')):
                current_cmd = max(CMD_MIN, round(current_cmd - CMD_STEP, 3))
                print(f"[Cmd] {current_cmd:.3f}")
            elif k in (ord('d'), ord('D')):
                current_cmd = min(CMD_MAX, round(current_cmd + CMD_STEP, 3))
                print(f"[Cmd] {current_cmd:.3f}")
            elif k in (ord('n'), ord('N')):
                current_cmd = min(CMD_MAX, round(current_cmd + CMD_STEP, 3))
                print(f"[Next Cmd] {current_cmd:.3f}")
            elif k in (ord('s'), ord('S')):  # start run
                if not running:
                    samples_for_run = []
                    run_t0 = time.time()
                    running = True
                    print(f"[Run] START @ cmd={current_cmd:.3f}")
            elif k in (ord('x'), ord('X')):  # stop & save
                if running:
                    running = False
                    neutral(q)
                    # summarize
                    if samples_for_run:
                        arr = np.asarray(samples_for_run, dtype=float)
                        v_mean = float(arr.mean())
                        v_std  = float(arr.std(ddof=0))
                        n      = int(arr.size)
                    else:
                        v_mean = 0.0; v_std = 0.0; n = 0
                    with open(SUMMARY_LOG, "a") as f:
                        f.write(f"{time.time():.6f}\t{current_cmd:.3f}\t{v_mean:.6f}\t{v_std:.6f}\t{n}\n")
                    print(f"[Run] STOP @ cmd={current_cmd:.3f} → mean={v_mean:.3f} std={v_std:.3f} N={n}")

            # pace loop a bit
            time.sleep(0.005)

    except KeyboardInterrupt:
        pass
    finally:
        try: neutral(q)
        except Exception: pass
        try: q.terminate()
        except Exception: pass
        cv2.destroyAllWindows()
        print(f"\nLogs written:\n  {SAMPLES_LOG}\n  {SUMMARY_LOG}\nBye!")

if __name__ == "__main__":
    main()
