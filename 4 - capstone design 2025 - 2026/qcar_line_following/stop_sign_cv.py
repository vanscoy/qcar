from Quanser.product_QCar import QCar
from Quanser.q_essential import Camera3D
import cv2, time, numpy as np

myCar = QCar()
frontCam = Camera3D(mode='RGB',
                    frame_width_RGB=1280,
                    frame_height_RGB=720,
                    frame_rate_RGB=20.0,
                    device_id='0')

# ---------- QCar SDK objects ----------
# Assuming frontCam and myCar are already initialized before this script
# frontCam: camera object
# myCar: QCar control object

# ---------- Parameters ----------
SUPPRESS_GLARE = False
GLARE_LOWER = (0, 0, 240)
GLARE_UPPER = (180, 60, 255)

WINDOW = 'Front 3D Camera View'
SHOW_MASK = False  # True to debug lane mask
bottom_frac = 0.4
MIN_CONTOUR_AREA = 200
SEP_MIN_PX = 50
MIN_PTS = 5
KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
OPEN_ITERS = 1
CLOSE_ITERS = 1
STEER_CLIP = 0.5
steering_gain = 0.01
max_steering_angle = 25  # degrees
base_speed = 0.15  # normal driving speed

frame_count, fps, last_time = 0, 0, time.time()

# ---------- Stop sign HSV thresholds ----------
RED1_LOWER = (0, 120, 70)
RED1_UPPER = (10, 255, 255)
RED2_LOWER = (170, 120, 70)
RED2_UPPER = (180, 255, 255)

# ---------- Helper functions ----------
def find_contours(bin_img):
    res = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(res) == 2:
        cnts, hier = res
    else:
        _, cnts, hier = res
    return cnts, hier

def make_lane_mask_hsv(roi_bgr):
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    # Yellow mask
    Y1_LOWER = (20, 100, 100)
    Y1_UPPER = (30, 255, 255)
    mask_yellow = cv2.inRange(hsv, np.array(Y1_LOWER), np.array(Y1_UPPER))
    # White mask
    W_LOWER = (0, 0, 200)
    W_UPPER = (180, 30, 255)
    mask_white = cv2.inRange(hsv, np.array(W_LOWER), np.array(W_UPPER))
    if SUPPRESS_GLARE:
        glare = cv2.inRange(hsv, np.array(GLARE_LOWER), np.array(GLARE_UPPER))
        mask_white = cv2.bitwise_and(mask_white, cv2.bitwise_not(glare))
    mask = cv2.bitwise_or(mask_yellow, mask_white)
    # Morphology
    if OPEN_ITERS:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL, iterations=OPEN_ITERS)
    if CLOSE_ITERS:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL, iterations=CLOSE_ITERS)
    return mask

def contour_centroid(contour):
    pts = contour.reshape(-1, 2)
    if pts.shape[0] < MIN_PTS:
        return None
    cx = int(pts[:,0].mean())
    cy = int(pts[:,1].mean())
    return cx, cy

def get_lanes_bottom40(image_bgr):
    h, w, _ = image_bgr.shape
    y0 = int(h * (1.0 - bottom_frac))
    roi = image_bgr[y0:h, 0:w]
    mask = make_lane_mask_hsv(roi)
    if SHOW_MASK:
        cv2.imshow("mask (bottom40%)", mask)
    cnts, _ = find_contours(mask)
    picks = []
    for c in cnts:
        if cv2.contourArea(c) < MIN_CONTOUR_AREA:
            continue
        cen = contour_centroid(c)
        if cen is None:
            continue
        cx, cy = cen
        picks.append({
            "contour_full": c + np.array([0, y0]),
            "cx_roi": cx,
            "cy_roi": cy
        })
    if not picks:
        return None
    picks.sort(key=lambda d: d["cx_roi"])
    right_two = picks[-2:] if len(picks) >= 2 else picks[-1:]
    if len(right_two) == 2 and abs(right_two[1]["cx_roi"] - right_two[0]["cx_roi"]) < SEP_MIN_PX:
        right_two = [right_two[1]]
    if len(right_two) == 2:
        mid_x_roi = int((right_two[0]["cx_roi"] + right_two[1]["cx_roi"]) * 0.5)
        control_x_full = mid_x_roi
        which = "pair"
    else:
        control_x_full = right_two[0]["cx_roi"]
        which = "single"
    return {
        "roi_origin": (0, y0),
        "roi_size": (w, h - y0),
        "blobs": picks,
        "right_two": right_two,
        "control_x_full": control_x_full,
        "which": which
    }

def detect_stop_sign(image_bgr):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, RED1_LOWER, RED1_UPPER)
    mask2 = cv2.inRange(hsv, RED2_LOWER, RED2_UPPER)
    mask = mask1 + mask2
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    stop_detected = False
    for cnt in contours:
        if cv2.contourArea(cnt) < 2000:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        if len(approx) == 8:
            stop_detected = True
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(image_bgr, (x,y), (x+w, y+h), (0,255,0), 3)
            cv2.putText(image_bgr, "STOP SIGN", (x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    return stop_detected

def safe_terminate_camera3d(cam):
    for name in ("terminate_RGB", "stop_RGB", "stop_rgb", "stop"):
        if hasattr(cam, name):
            try: getattr(cam, name)()
            except Exception: pass
    try:
        cam.terminate()
    except Exception: pass

# ---------- Main Loop ----------
try:
    while True:
        t0 = time.time()
        frontCam.read_RGB()
        img = frontCam.image_buffer_RGB
        if img is None or img.size == 0:
            print("Warning: Camera returned invalid image data.")
            continue
        display = img.copy()
        info = get_lanes_bottom40(img)

        frame_count += 1
        now = time.time()
        if now - last_time >= 1.0:
            fps = frame_count
            frame_count = 0
            last_time = now

        desired_x = img.shape[1] // 2
        steering = 0.0

        # Lane following
        if info is not None:
            rx0, ry0 = info["roi_origin"]
            rw, rh = info["roi_size"]
            cv2.rectangle(display, (rx0, ry0), (rx0+rw-1, ry0+rh-1), (0,255,0), 2)
            for b in info["blobs"]:
                cv2.drawContours(display, [b["contour_full"]], -1, (255,200,0), 2)
                cv2.circle(display, (b["cx_roi"], b["cy_roi"] + ry0), 6, (255,0,0), -1)
            chosen = info["right_two"] if isinstance(info["right_two"], list) else [info["right_two"]]
            for b in chosen:
                cv2.drawContours(display, [b["contour_full"]], -1, (255,0,255), 3)
                cv2.circle(display, (b["cx_roi"], b["cy_roi"] + ry0), 8, (255,0,255), -1)
            ctrl_x = int(info["control_x_full"])
            cv2.circle(display, (ctrl_x, ry0+rh//2), 8, (0,255,255), -1)
            cv2.circle(display, (desired_x, ry0+rh//2), 8, (0,0,255), -1)
            error = desired_x - ctrl_x
            steering = float(np.clip(error * steering_gain, -STEER_CLIP, STEER_CLIP))
        else:
            cv2.putText(display, "NO YELLOW/WHITE LANE FOUND (bottom 40%)", (10,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

        # Stop sign detection
        stop_seen = detect_stop_sign(display)
        if stop_seen:
            cv2.putText(display, "STOP!", (10,120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
            speed = 0.0
        else:
            speed = base_speed

        # Display info
        calc_ms = (time.time() - t0)*1000.0
        cv2.putText(display, f'FPS:{fps}  Calc:{calc_ms:.1f} ms', (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.putText(display, f'Steer:{steering:+.3f}  Gain:{steering_gain}', (10,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
        cv2.putText(display, f'Angle:{steering*max_steering_angle:+.1f} deg', (10,90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,128,255), 2)

        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.imshow(WINDOW, display)
        if (cv2.waitKey(1) & 0xFF) == 27:
            print("Kill switch: ESC pressed.")
            break

        # Drive
        mtr_cmd = np.array([speed, steering], dtype=np.float64)
        LEDs = np.array([0,0,0,0, 0,0,1,1], dtype=np.float64)
        myCar.read_write_std(mtr_cmd, LEDs)

        time.sleep(0.05)

finally:
    cv2.destroyAllWindows()
    try: myCar.terminate()
    except Exception: pass
    safe_terminate_camera3d(frontCam)
