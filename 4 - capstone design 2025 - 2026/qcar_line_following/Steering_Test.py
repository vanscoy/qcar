# steering_speed_tester.py
# QCar steering + speed keyboard test (steer range [-0.5, +0.5]) with on-screen HUD

from Quanser.product_QCar import QCar
import numpy as np
import cv2, time, signal

# ---- QCar ----
car = QCar()

# ---- Params ----
STEER_MAX_DEG = 28.0      # physical limit at full lock (for display only)
STEER_FULL_SCALE = 0.5    # full-lock command magnitude is ±0.5
speed = 0.00              # suggested useful range ~0.00..0.08
steer = 0.00              # normalized steering command in [-0.5, +0.5]
DSPEED = 0.005
DSTEER = 0.05
LEDs = np.array([0,0,0,0, 0,0,1,1], dtype=np.float64)

# ---- HUD window ----
WIN = "QCar Steering/Speed Test"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)   # create once (don't resize each frame)
W, H = 800, 380
FONT = cv2.FONT_HERSHEY_SIMPLEX

# ---- Clean stop on Ctrl-C / kill ----
running = True
def _sig_handler(sig, frame):
    global running
    running = False
signal.signal(signal.SIGINT, _sig_handler)
signal.signal(signal.SIGTERM, _sig_handler)

def draw_hud(canvas, speed, steer):
    canvas[:] = (20, 20, 20)

    # Headline
    cv2.putText(canvas, "QCar Steering/Speed Tester", (18, 36), FONT, 0.9, (200,255,200), 2)

    # Angle estimate: ±28 deg at ±0.5 command
    angle_deg = (steer / STEER_FULL_SCALE) * STEER_MAX_DEG
    y0 = 85
    cv2.putText(canvas, "Speed:  {:.3f}".format(speed), (18, y0), FONT, 0.8, (0,255,255), 2)
    cv2.putText(canvas, "Steer:  {:+.3f}  (limit -0.5..+0.5)".format(steer), (18, y0+35), FONT, 0.8, (255,255,0), 2)
    cv2.putText(canvas, "Angle:  {:+.1f} deg (est.)".format(angle_deg), (18, y0+70), FONT, 0.8, (0,200,255), 2)

    # Speed bar (0..0.12 visual)
    sp_min, sp_max = 0.0, 0.12
    sp_norm = np.clip((speed - sp_min) / (sp_max - sp_min), 0.0, 1.0)
    cv2.putText(canvas, "Speed bar (0..0.12)", (18, y0+120), FONT, 0.6, (180,180,180), 1)
    cv2.rectangle(canvas, (18, y0+140), (18+500, y0+170), (80,80,80), 2)
    cv2.rectangle(canvas, (18, y0+140), (18+int(500*sp_norm), y0+170), (0,255,255), -1)
    if speed > 0.08:
        cv2.putText(canvas, "NOTE: >0.08 gets fast", (530, y0+165), FONT, 0.6, (0,165,255), 2)

    # Steering bar (-0.5..+0.5)
    cv2.putText(canvas, "Steering bar (-0.5..+0.5)", (18, y0+210), FONT, 0.6, (180,180,180), 1)
    cx, yb = 18+250, y0+230
    cv2.line(canvas, (cx-250, yb), (cx+250, yb), (80,80,80), 10)
    # map steer in [-0.5,+0.5] to bar [-1,+1]
    steer_norm = np.clip(steer / STEER_FULL_SCALE, -1.0, 1.0)
    sx = int(cx + 250 * steer_norm)
    cv2.circle(canvas, (sx, yb), 10, (255,255,0), -1)
    # tick marks
    for t in (-1, -0.5, 0, 0.5, 1):
        tx = int(cx + 250*t)
        cv2.line(canvas, (tx, yb-15), (tx, yb+15), (120,120,120), 2)
    cv2.putText(canvas, "left", (cx-280, yb+35), FONT, 0.6, (160,160,160), 1)
    cv2.putText(canvas, "right", (cx+220, yb+35), FONT, 0.6, (160,160,160), 1)

    # Help
    help_y = H - 20
    help_text = "W/S speed  A/D steer  X center  SPACE stop  R reset  ESC quit"
    cv2.putText(canvas, help_text, (18, help_y), FONT, 0.6, (200,200,200), 1)

def send(speed, steer):
    try:
        car.read_write_std(np.array([speed, steer], dtype=np.float64), LEDs)
    except Exception:
        pass

try:
    last_send = time.time()
    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    while running:
        # Draw HUD
        draw_hud(canvas, speed, steer)
        cv2.imshow(WIN, canvas)

        # Keys are read from the OpenCV window; click it to focus.
        key = cv2.waitKey(10) & 0xFF
        if key == 27:  # ESC
            running = False
        elif key in (ord('w'), ord('W')):
            speed += DSPEED
        elif key in (ord('s'), ord('S')):
            speed -= DSPEED
        elif key in (ord('a'), ord('A')):
            steer -= DSTEER
        elif key in (ord('d'), ord('D')):
            steer += DSTEER
        elif key in (ord('x'), ord('X')):
            steer = 0.0
        elif key == 32:  # SPACE
            speed = 0.0
        elif key in (ord('r'), ord('R')):
            speed, steer = 0.0, 0.0

        # Soft-limit steering to [-0.5, +0.5]
        if steer > STEER_FULL_SCALE:
            steer = STEER_FULL_SCALE
        if steer < -STEER_FULL_SCALE:
            steer = -STEER_FULL_SCALE

        # modest loop rate
        now = time.time()
        if now - last_send >= 0.02:  # ~50 Hz
            send(speed, steer)
            last_send = now

finally:
    # Zero outputs for safety
    try:
        car.read_write_std(np.array([0.0, 0.0], dtype=np.float64), np.zeros(8))
    except Exception:
        pass
    cv2.destroyAllWindows()
    try:
        car.terminate()
    except Exception:
        pass
