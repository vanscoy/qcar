# QCar lane keeping + stop sign detection
from Quanser.product_QCar import QCar
from Quanser.q_essential import Camera3D
import cv2, time, numpy as np

myCar = QCar()
frontCam = Camera3D(mode='RGB',
                    frame_width_RGB=1280,
                    frame_height_RGB=720,
                    frame_rate_RGB=20.0,
                    device_id='0')

# ---------- Lane-following tunables ----------
bottom_frac = 0.40
MIN_CONTOUR_AREA = 60
SEP_MIN_PX = 30
MIN_PTS = 20

speed = 0.078
steering_gain = 0.0012
STEER_CLIP = 0.5
max_steering_angle = 28.0

KERNEL = np.ones((5,5), np.uint8)
OPEN_ITERS = 1
CLOSE_ITERS = 1

Y1_LOWER = (15, 90, 80)
Y1_UPPER = (45, 255, 255)
Y2_LOWER = (15, 40, 60)
Y2_UPPER = (45, 255, 200)

WHITE_S_MAX = 60
WHITE_V_MIN = 200
W_LOWER = (0, 0, WHITE_V_MIN)
W_UPPER = (180, WHITE_S_MAX, 255)

SUPPRESS_GLARE = False
GLARE_LOWER = (0, 0, 240)
GLARE_UPPER = (180, 60, 255)

WINDOW = 'Front 3D Camera View'
SHOW_MASK = False

frame_count, fps, last_time = 0, 0, time.time()

# ---------- Stop sign parameters ----------
RED1_LOWER = (0, 70, 50)
RED1_UPPER = (10, 255, 255)
RED2_LOWER = (160, 70, 50)
RED2_UPPER = (180, 255, 255)
MIN_STOP_AREA = 500
base_speed = speed
stop_duration = 3.0
stopping = False
stop_timer = 0.0

# ---------- Helpers ----------
def find_contours(bin_img):
    res = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return res[0] if len(res) == 2 else res[1]

def make_lane_mask_hsv(roi_bgr):
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, np.array(Y1_LOWER), np.array(Y1_UPPER))
    m2 = cv2.inRange(hsv, np.array(Y2_LOWER), np.array(Y2_UPPER))
    mask_yellow = cv2.bitwise_or(m1, m2)
    mask_white = cv2.inRange(hsv, np.array(W_LOWER), np.array(W_UPPER))
    if SUPPRESS_GLARE:
        glare = cv2.inRange(hsv, np.array(GLARE_LOWER), np.array(GLARE_UPPER))
        mask_white = cv2.bitwise_and(mask_white, cv2.bitwise_not(glare))
    mask = cv2.bitwise_or(mask_yellow, mask_white)
    if OPEN_ITERS:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL, iterations=OPEN_ITERS)
    if CLOSE_ITERS:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL, iterations=CLOSE_ITERS)
    return mask

def contour_centroid(contour):
    pts = contour.reshape(-1, 2)
    if pts.shape[0] < MIN_PTS: return None
    return int(pts[:,0].mean()), int(pts[:,1].mean())

def get_lanes_bottom40(image_bgr):
    h, w, _ = image_bgr.shape
    y0 = int(h * (1.0 - bottom_frac))
    roi = image_bgr[y0:h, 0:w]
    mask = make_lane_mask_hsv(roi)
    if SHOW_MASK: cv2.imshow("mask (bottom40%)", mask)
    cnts = find_contours(mask)
    picks = []
    for c in cnts:
        if cv2.contourArea(c) < MIN_CONTOUR_AREA: continue
        cen = contour_centroid(c)
        if cen is None: continue
        cx, cy = cen
        picks.append({"contour_full": c + np.array([0,y0]), "cx_roi": cx, "cy_roi": cy})
    if not picks: return None
    picks.sort(key=lambda d: d["cx_roi"])
    right_two = picks[-2:] if len(picks) >= 2 else picks[-1:]
    if len(right_two) == 2 and abs(right_two[1]["cx_roi"] - right_two[0]["cx_roi"]) < SEP_MIN_PX:
        right_two = [right_two[1]]
    ctrl_x = int((right_two[0]["cx_roi"] + right_two[1]["cx_roi"])*0.5) if len(right_two) == 2 else right_two[0]["cx_roi"]
    return {"roi_origin": (0,y0), "roi_size": (w,h-y0), "blobs": picks, "right_two": right_two, "control_x_full": ctrl_x}

def detect_stop_sign(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, RED1_LOWER, RED1_UPPER)
    mask2 = cv2.inRange(hsv, RED2_LOWER, RED2_UPPER)
    mask = mask1 + mask2
    mask = cv2.GaussianBlur(mask, (5,5), 0)
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) < MIN_STOP_AREA: continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04*peri, True)
        if 7 <= len(approx) <= 9:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)
            cv2.putText(frame, "STOP SIGN", (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            return True
    return False

def safe_terminate_camera3d(cam):
    for name in ("terminate_RGB","stop_RGB","stop_rgb","stop"):
        if hasattr(cam,name):
            try: getattr(cam,name)()
            except: pass
    try: cam.terminate()
    except: pass

# ---------- Main Loop ----------
try:
    while True:
        t0 = time.time()
        frontCam.read_RGB()
        img = frontCam.image_buffer_RGB
        if img is None or img.size == 0: continue

        display = img.copy()
        info = get_lanes_bottom40(img)

        frame_count += 1
        now = time.time()
        if now - last_time >= 1.0:
            fps = frame_count
            frame_count = 0
            last_time = now

        # ---------- Lane following ----------
        desired_x = img.shape[1]//2
        steering = 0.0
        if info is not None:
            rx0, ry0 = info["roi_origin"]
            rw, rh = info["roi_size"]
            cv2.rectangle(display, (rx0,ry0),(rx0+rw-1,ry0+rh-1),(0,255,0),2)
            for b in info["blobs"]:
                cv2.drawContours(display, [b["contour_full"]], -1, (255,200,0), 2)
                cv2.circle(display, (b["cx_roi"], b["cy_roi"]+ry0), 6, (255,0,0), -1)
            chosen = info["right_two"] if isinstance(info["right_two"], list) else [info["right_two"]]
            for b in chosen:
                cv2.drawContours(display, [b["contour_full"]], -1, (255,0,255), 3)
                cv2.circle(display, (b["cx_roi"], b["cy_roi"]+ry0), 8, (255,0,255), -1)
            ctrl_x = int(info["control_x_full"])
            cv2.circle(display, (ctrl_x, ry0+rh//2), 8, (0,255,255), -1)
            cv2.circle(display, (desired_x, ry0+rh//2), 8, (0,0,255), -1)
            error = desired_x - ctrl_x
            steering = float(np.clip(error*steering_gain, -STEER_CLIP, STEER_CLIP))

        # ---------- Stop sign detection ----------
        stop_seen = detect_stop_sign(display)
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
            speed = base_speed

        # ---------- Display info ----------
        calc_ms = (time.time() - t0)*1000.0
        cv2.putText(display, f'FPS:{fps}  Calc:{calc_ms:.1f} ms', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0),2)
        cv2.putText(display, f'Steer:{steering:+.3f}', (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0),2)
        cv2.putText(display, f'Angle:{steering*max_steering_angle:+.1f} deg', (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,128,255),2)
        cv2.imshow(WINDOW, display)
        if (cv2.waitKey(1) & 0xFF)==27: break

        # ---------- Drive ----------
        mtr_cmd = np.array([speed, steering], dtype=np.float64)
        LEDs = np.array([0,0,0,0, 0,0,1,1], dtype=np.float64)
        myCar.read_write_std(mtr_cmd, LEDs)

        time.sleep(0.05)

finally:
    cv2.destroyAllWindows()
    try: myCar.terminate()
    except: pass
    safe_terminate_camera3d(frontCam)
