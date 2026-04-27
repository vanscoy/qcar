#!/usr/bin/env python3
# QCar straight-drive distance tester
#
# Goal:
# - Drive straight ahead only
# - Continuously track encoder distance
# - Press S to start
# - Press X to stop and save
# - Repeat as many times as you want
#
# Controls:
#   S : start driving straight
#   X : stop and save result
#   A : speed -0.002
#   D : speed +0.002
#   R : reset odometry
#   Q : neutral stop
#   ESC : quit

from Quanser.product_QCar import QCar
import cv2
import time
import numpy as np
from math import pi

# ========================= Tunables =========================
WINDOW = "QCar Straight + Distance"

INIT_SPEED = 0.080
SPEED_STEP = 0.002

STEERING_CMD = 0.0   # straight

TICKS_PER_REV = 31844.0
WHEEL_DIAM_M  = 0.066
WHEEL_CIRC_M  = pi * WHEEL_DIAM_M

EMA_ALPHA = 0.35
DT_MIN, DT_MAX = 0.004, 0.35
MPS_HARD_MAX = 6.0

LOG_FILE = "straight_distance_log.txt"

# ========================= Helpers =========================
def neutral(qcar):
    try:
        qcar.read_write_std(
            np.array([0.0, 0.0], dtype=np.float64),
            np.array([0,0,0,0, 0,0,0,0], dtype=np.float64)
        )
    except Exception:
        pass

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

def ensure_log():
    try:
        with open(LOG_FILE, "x") as f:
            f.write("# ts\tspeed\trun_time_s\tdist_m\tmean_v\tfinal_v\n")
    except FileExistsError:
        pass

# ========================= Main =========================
def main():
    ensure_log()

    q = QCar()
    odo = SpeedOdom()
    odo.reset(q)

    speed = INIT_SPEED
    running = False
    run_t0 = None
    speed_samples = []

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 760, 420)

    print("""
Controls:
  S : start driving straight
  X : stop and save result
  A : speed -0.002
  D : speed +0.002
  R : reset odometry
  Q : neutral stop
  ESC : quit
""")

    try:
        while True:
            v_raw, v_filt, dist_m, dt = odo.update(q)

            if running:
                try:
                    q.read_write_std(
                        np.array([speed, STEERING_CMD], dtype=np.float64),
                        np.array([0,0,0,0, 0,0,1,1], dtype=np.float64)
                    )
                except Exception:
                    neutral(q)
                speed_samples.append(v_filt)
            else:
                neutral(q)

            run_time = (time.time() - run_t0) if (running and run_t0 is not None) else 0.0

            canvas = np.zeros((420, 760, 3), dtype=np.uint8)

            def put(y, text, color=(0,255,255), scale=0.8):
                cv2.putText(canvas, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                            scale, color, 2, cv2.LINE_AA)

            put(40,  f"Mode: {'RUNNING STRAIGHT' if running else 'IDLE'}", (0,200,255))
            put(90,  f"Speed cmd:   {speed:.3f}", (0,255,0))
            put(130, f"Steering:    {STEERING_CMD:+.3f}", (255,255,0))
            put(170, f"Run time:    {run_time:.2f} s", (255,255,0))
            put(210, f"Distance:    {dist_m:.3f} m", (255,255,0))
            put(250, f"v_raw:       {v_raw:+.3f} m/s", (200,200,255))
            put(290, f"v_filt:      {v_filt:+.3f} m/s", (200,200,255))
            put(350, "Keys: S=start  X=stop+save  A/D=speed  R=reset  Q=neutral  ESC=quit",
                (180,180,180), 0.55)

            cv2.imshow(WINDOW, canvas)

            k = cv2.waitKey(1) & 0xFF

            if k == 27:  # ESC
                break

            elif k in (ord('q'), ord('Q')):
                running = False
                neutral(q)
                print("[Neutral] stopped")

            elif k in (ord('a'), ord('A')):
                speed = max(0.0, round(speed - SPEED_STEP, 3))
                print(f"[Speed] {speed:.3f}")

            elif k in (ord('d'), ord('D')):
                speed = round(speed + SPEED_STEP, 3)
                print(f"[Speed] {speed:.3f}")

            elif k in (ord('r'), ord('R')):
                odo.reset(q)
                speed_samples = []
                run_t0 = None
                print("[Reset] odometry reset")

            elif k in (ord('s'), ord('S')):
                if not running:
                    odo.reset(q)
                    speed_samples = []
                    run_t0 = time.time()
                    running = True
                    print(f"[Run] START straight | speed={speed:.3f}")

            elif k in (ord('x'), ord('X')):
                if running:
                    running = False
                    neutral(q)

                    run_time = time.time() - run_t0 if run_t0 is not None else 0.0
                    mean_v = float(np.mean(speed_samples)) if speed_samples else 0.0
                    final_v = float(v_filt)

                    with open(LOG_FILE, "a") as f:
                        f.write(
                            f"{time.time():.6f}\t"
                            f"{speed:.3f}\t"
                            f"{run_time:.3f}\t"
                            f"{dist_m:.3f}\t"
                            f"{mean_v:.6f}\t"
                            f"{final_v:.6f}\n"
                        )

                    print(f"[Run] STOP straight | speed={speed:.3f} | time={run_time:.2f}s | dist={dist_m:.3f}m")

            time.sleep(0.005)

    except KeyboardInterrupt:
        pass
    finally:
        try:
            neutral(q)
        except Exception:
            pass
        try:
            q.terminate()
        except Exception:
            pass
        cv2.destroyAllWindows()
        print(f"\nSaved log: {LOG_FILE}")

if __name__ == "__main__":
    main()