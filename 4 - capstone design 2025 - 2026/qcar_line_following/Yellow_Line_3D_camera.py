# QCar + front 3D camera, simple yellow follow on bottom 40% ROI
from Quanser.product_QCar import QCar
from Quanser.q_essential import Camera3D
import cv2, time, numpy as np

myCar = QCar()
frontCam = Camera3D(mode='RGB', frame_width_RGB=1280, frame_height_RGB=720, frame_rate_RGB=20.0, device_id='0')

# --- Tunables ---
bottom_frac = 0.40            # use bottom 40% of the image (full width)
target_offset_right = 640      # desired x-position = 50 px from RIGHT edge of full frame
speed = 0.075
steering_gain = 0.005
max_steering_angle = 45       # HUD only

# Simple HSV yellow (robust but still minimal). If you prefer grayscale threshold, see comment below.
LOWER_YELLOW = np.array([15, 80, 80], dtype=np.uint8)
UPPER_YELLOW = np.array([40, 255, 255], dtype=np.uint8)
KERNEL = np.ones((5,5), np.uint8)
OPEN_ITERS = 1                # light denoise; set 0 to disable

frame_count, fps, last_time = 0, 0, time.time()

def get_line_info_bottom(image):
    h, w, _ = image.shape
    y0 = int(h * (1.0 - bottom_frac))     # start row of bottom 40%
    roi = image[y0:h, 0:w]                 # FULL WIDTH, bottom 40%

    # Yellow mask in HSV (swap to grayscale threshold if you like—see below)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_YELLOW, UPPER_YELLOW)
    if OPEN_ITERS:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL, iterations=OPEN_ITERS)

    # If you want pure grayscale instead, replace the 4 lines above with:
    # gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # _, mask = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    largest = max(cnts, key=cv2.contourArea)
    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None

    cx = int(M["m10"]/M["m00"])           # in ROI coords (0..w-1)
    cy = int(M["m01"]/M["m00"])           # in ROI coords
    # Map to full-frame for drawing
    contour_full = largest + np.array([0, y0])
    centroid_full = (cx, y0 + cy)
    return {"contour": contour_full, "centroid": centroid_full, "cx_full": cx, "roi_origin": (0, y0), "roi_size": (w, h-y0)}

try:
    while True:
        t0 = time.time()
        frontCam.read_RGB()
        img = frontCam.image_buffer_RGB
        if img is None or img.shape[0] == 0 or img.shape[1] == 0:
            print("Warning: Camera returned invalid image data.")
            continue

        info = get_line_info_bottom(img)
        h, w, _ = img.shape
        display = img.copy()

        # FPS
        frame_count += 1
        now = time.time()
        if now - last_time >= 1.0:
            fps = frame_count; frame_count = 0; last_time = now

        # Default straight
        steering = 0.0

        if info is not None:
            # Desired x = N pixels from RIGHT edge of FULL FRAME
            desired_x = w - target_offset_right
            error = desired_x - info["cx_full"]   # +error => target is to the right of detected cx
            steering = np.clip(error * steering_gain, -1.0, 1.0)

            # Draw ROI box, contour, centroid, and target dot
            (rx0, ry0) = info["roi_origin"]; (rw, rh) = info["roi_size"]
            cv2.rectangle(display, (rx0, ry0), (rx0+rw-1, ry0+rh-1), (0,255,0), 2)
            cv2.drawContours(display, [info["contour"]], -1, (255,0,0), 2)
            cv2.circle(display, info["centroid"], 8, (255,0,0), -1)
            cv2.circle(display, (desired_x, ry0 + rh//2), 8, (0,0,255), -1)
        # else: steering stays 0.0 (go straight if no line found)

        # HUD
        calc_ms = (time.time() - t0) * 1000.0
        cv2.putText(display, f'FPS:{fps}  Calc:{calc_ms:.1f} ms', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.putText(display, f'Steering:{steering:+.3f}  Gain:{steering_gain}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
        cv2.putText(display, f'Angle:{steering*max_steering_angle:+.1f} deg', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,128,255), 2)

        # Show
        cv2.namedWindow('Front 3D Camera View', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Front 3D Camera View', 1280, 720)
        cv2.imshow('Front 3D Camera View', display)
        if (cv2.waitKey(1) & 0xFF) == 27:
            print("Kill switch activated: ESC pressed.")
            break

        # Drive
        mtr_cmd = np.array([speed, steering], dtype=np.float64)  # [throttle, steer]
        LEDs = np.array([0,0,0,0, 0,0,1,1], dtype=np.float64)
        myCar.read_write_std(mtr_cmd, LEDs)

        time.sleep(0.05)
finally:
    cv2.destroyAllWindows()
    myCar.terminate()
    frontCam.terminate()
