#!/usr/bin/env python3
# QCar commanded-speed vs actual speed (stand test; robust summary logger)
# Dynamic guards so high-speed steps don't get over-filtered.

from Quanser.product_QCar import QCar
import time, numpy as np
from math import pi

# ===== Vehicle / encoder constants (from your setup) =====
TICKS_PER_REV = 31844.0     # ticks per wheel revolution (includes gearing)
WHEEL_DIAM_M  = 0.066       # meters
WHEEL_CIRC_M  = pi * WHEEL_DIAM_M

# ===== Sweep & logging settings =====
CMD_START   = 0.000
CMD_STOP    = 0.300          # inclusive
CMD_STEP    = 0.005
STEP_HOLD_S = 5.0            # total time per step
SETTLE_S    = 1.5            # discard this much at start of each step
DT_S        = 0.05           # ~20 Hz sampling
OUT_PATH    = "cmd_vs_speed_summary.txt"
TEST_LABEL  = "stand_v3"

# ===== Sample guardrails (dynamic) =====
MPS_MAX_HARD   = 6.0         # absolute hard cap on |v| (very generous)
DT_MIN, DT_MAX = 0.008, 0.30 # valid per-sample dt window (s)

# Dynamic tick jump limit:
# We assume a plausible top speed on a stand (m/s), convert to ticks per dt, add margin.
V_MAX_PLAUSIBLE = 4.0        # tune if needed
JUMP_MARGIN     = 1.6        # >1 leaves headroom for noise

def read_ticks(qcar) -> float:
    return float(qcar.read_encoder())

def neutral(qcar):
    try:
        qcar.read_write_std(np.array([0.0, 0.0], dtype=np.float64),
                            np.array([0,0,0,0, 0,0,0,0], dtype=np.float64))
    except Exception:
        pass

def trimmed_mean_std(samples, trim=0.05):
    if not samples:
        return 0.0, 0.0
    a = np.sort(np.asarray(samples, dtype=float))
    k = int(len(a) * trim)
    if len(a) > 2 * k:
        a = a[k: len(a) - k]
    return float(a.mean()), float(a.std(ddof=0))

def main():
    cmds = [round(x, 3) for x in np.arange(CMD_START, CMD_STOP + 1e-12, CMD_STEP)]

    # Header (append if exists)
    header = [
        "# QCar speed sweep summary (stand test, robust v3 - dynamic guards)",
        f"# test={TEST_LABEL}",
        f"# cmd_start={CMD_START:.3f}, cmd_stop={CMD_STOP:.3f}, cmd_step={CMD_STEP:.3f}",
        f"# step_hold_s={STEP_HOLD_S}, settle_s={SETTLE_S}, dt≈{DT_S}s",
        f"# wheel_diam_m={WHEEL_DIAM_M}, ticks_per_rev={TICKS_PER_REV}",
        f"# guards: MPS_MAX_HARD={MPS_MAX_HARD}, DT_MIN={DT_MIN}, DT_MAX={DT_MAX},",
        f"#         V_MAX_PLAUSIBLE={V_MAX_PLAUSIBLE}, JUMP_MARGIN={JUMP_MARGIN}",
        "#",
        "# Columns (TSV): timestamp  test  cmd  v_mean_mps  v_std_mps  samples",
        ""
    ]
    try:
        with open(OUT_PATH, "x") as f:
            f.write("\n".join(header) + "\n")
    except FileExistsError:
        with open(OUT_PATH, "a") as f:
            f.write("\n# --- APPEND RUN ---\n")

    qcar = QCar()
    try:
        with open(OUT_PATH, "a", buffering=1) as f:
            for idx, u in enumerate(cmds, 1):
                print(f"\n=== Step {idx}/{len(cmds)}: cmd={u:.3f} ===")
                cmd_vec = np.array([u, 0.0], dtype=np.float64)

                ticks_prev = read_ticks(qcar)
                step_t0 = time.time()
                samples = []
                kept = 0

                while True:
                    # Command (steering=0; LEDs optional)
                    qcar.read_write_std(cmd_vec, np.array([0,0,0,0, 0,0,1,1], dtype=np.float64))

                    t_before = time.time()
                    time.sleep(DT_S)
                    t_after = time.time()

                    dt_sample = max(1e-3, t_after - t_before)
                    ticks_now = read_ticks(qcar)
                    d_ticks = ticks_now - ticks_prev
                    ticks_prev = ticks_now

                    # distance & instantaneous speed
                    dist_m = (d_ticks / TICKS_PER_REV) * WHEEL_CIRC_M
                    v = dist_m / dt_sample
                    t_rel = t_after - step_t0

                    # --- dynamic tick jump limit for this dt ---
                    revs_per_sec_max = V_MAX_PLAUSIBLE / WHEEL_CIRC_M
                    ticks_jump_max = JUMP_MARGIN * (revs_per_sec_max * TICKS_PER_REV * dt_sample)

                    # ---- guards ----
                    bad = False
                    if not (DT_MIN <= dt_sample <= DT_MAX):
                        bad = True
                    elif abs(v) > MPS_MAX_HARD:
                        bad = True
                    elif abs(d_ticks) > ticks_jump_max:
                        bad = True

                    if (not bad) and (t_rel >= SETTLE_S):
                        samples.append(v); kept += 1

                    if int(t_rel / DT_S) % int(max(1, 0.5/DT_S)) == 0:
                        print(f"\r  t={t_rel:4.1f}s  v={v:+.3f} m/s  kept={kept}", end="")

                    if t_rel >= STEP_HOLD_S:
                        break

                v_mean, v_std = trimmed_mean_std(samples, trim=0.05)
                n = int(len(samples))
                print(f"\n -> mean={v_mean:.3f} m/s  std={v_std:.3f}  N={n}")

                ts = time.time()
                f.write(f"{ts:.6f}\t{TEST_LABEL}\t{u:.3f}\t{v_mean:.6f}\t{v_std:.6f}\t{n}\n")

                # neutral between steps
                qcar.read_write_std(np.array([0.0, 0.0], dtype=np.float64),
                                    np.array([0,0,0,0, 0,0,0,0], dtype=np.float64))
                time.sleep(0.3)

    except KeyboardInterrupt:
        print("\nInterrupted — writing what we have.")
    finally:
        try: neutral(qcar)
        except Exception: pass
        try: qcar.terminate()
        except Exception: pass
        print(f"\nDone. Summary in {OUT_PATH}")

if __name__ == "__main__":
    main()
