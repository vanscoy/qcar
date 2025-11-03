from Quanser.product_QCar import QCar
from Quanser.q_essential import Camera3D
import cv2
import numpy as np
import time

# Initialize car and camera
myCar = QCar()
frontCam = Camera3D(mode='RGB',
                    frame_width_RGB=1280,
                    frame_height_RGB=720,
                    frame_rate_RGB=20.0,
                    device_id='0')

# Parameters
WINDOW = "QCar Stop Sign Demo"
base_speed = 0.15        # Forward speed
stop_duration = 3.0      # Seconds to stop at a stop sign

# Improved HSV thresholds for red
RED1_LOWER = (0, 70, 50)
RED1_UPPER = (10, 255, 255)
RED2_LOWER = (160, 70, 50)
RED2_UPPER = (180, 255, 255)

MIN_STOP_AREA = 500       # Smaller to catch distant/far stop signs

stopping = False
stop_timer = 0.0

def detect_stop_sign(frame):
    """Detect red octagon stop sign, draw a box, and return True if found"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Red masks
    mask1 = cv2.inRange(hsv, RED1_LOWER, RED1_UPPER)
    mask2 = cv2.inRange(hsv, RED2_LOWER, RED2_UPPER)
    mask = mask1 + mask2

    # Reduce noise
    mask = cv2.GaussianBlur(mask, (5,5), 0)
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # For debugging: see what the mask looks like
    if False:  # set True to debug
        cv2.imshow("Mask", mask)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) < MIN_STOP_AREA:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        if 7 <= len(approx) <= 9:  # tolerate slightly imperfect octagons
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
            cv2.putText(frame, "STOP SIGN", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            return True
    return False

# ---------- Main Loop ----------
try:
    while True:
        frontCam.read_RGB()
        img = frontCam.image_buffer_RGB
        if img is None or img.size == 0:
            continue

        display = img.copy()
        stop_seen = detect_stop_sign(display)

        # Stop sign logic
        if stop_seen and not stopping:
            stopping = True
            stop_timer = time.time()
            speed = 0.0
        elif stopping:
            speed = 0.0
            if time.time() - stop_timer >= stop_duration:
                stopping = False
                speed = base_speed
        else:
            speed = base_speed  # drive forward

        # Display
        cv2.imshow(WINDOW, display)
        if (cv2.waitKey(1) & 0xFF) == 27:  # ESC to quit
            break

        # Drive
        mtr_cmd = np.array([speed, 0.0], dtype=np.float64)  # forward only
        LEDs = np.array([0,0,0,0, 0,0,1,1], dtype=np.float64)
        myCar.read_write_std(mtr_cmd, LEDs)

        time.sleep(0.05)

finally:
    cv2.destroyAllWindows()
    try: myCar.terminate()
    except Exception: pass
    if hasattr(frontCam, "terminate"):
        frontCam.terminate()
