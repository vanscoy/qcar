#!/usr/bin/env python3
# qcar_drive_with_encoder.py
# Moves QCar forward and computes encoder-based speed (m/s)

from Quanser.product_QCar import QCar
import numpy as np
import time
import signal

# --- Constants ---
WHEEL_RADIUS_M = 0.033     # QCar wheel radius (meters)
TICKS_PER_REV = 720.0     # Encoder ticks per wheel revolution
GEAR_RATIO = 20.0           # Adjust if gearbox exists
LEDs = np.array([0, 0, 1, 1, 0, 0, 0, 0], dtype=np.float64)

# --- Control parameters ---
speed_cmd = 0.1    # forward command (try 0.02–0.06)
steer_cmd = 0.0    # 0 = straight

# --- Setup QCar ---
car = QCar()
running = True

def _sig_handler(sig, frame):
    global running
    running = False
signal.signal(signal.SIGINT, _sig_handler)
signal.signal(signal.SIGTERM, _sig_handler)

print("🚗 QCar Drive + Encoder Speed Test (Ctrl+C to stop)\n")

# --- Initialize state ---
last_ticks = 0
last_time = time.time()

try:
    while running:
        # --- Send motor + steering command ---
        car.read_write_std(np.array([speed_cmd, steer_cmd], dtype=np.float64), LEDs)

        # --- Read encoders ---
        data = car.read_std()
        if not isinstance(data, tuple) or len(data) < 3:
            print("⚠️ Unexpected data format:", data)
            continue

        ticks = data[2]  # encoder count (from your observation)
        now = time.time()
        dt = now - last_time

        if dt > 0:
            delta_ticks = ticks - last_ticks
            delta_rad = (delta_ticks / TICKS_PER_REV) * 2 * np.pi / GEAR_RATIO
            distance_m = WHEEL_RADIUS_M * delta_rad
            speed_mps = distance_m / dt

            print(f"Encoder: {ticks:.0f}  Δticks: {delta_ticks:.1f}  Speed: {speed_mps:.3f} m/s")

        last_ticks = ticks
        last_time = now
        time.sleep(0.05)  # ~20 Hz loop rate

finally:
    print("\n🛑 Stopping QCar...")
    try:
        car.read_write_std(np.array([0.0, 0.0], dtype=np.float64), np.zeros(8))
    except Exception:
        pass
    car.terminate()
