# QCar lane keeping using HSV (yellow OR white) over bottom 40% only
from Quanser.product_QCar import QCar
from Quanser.q_essential import Camera3D
import cv2, time, numpy as np

myCar = QCar()
frontCam = Camera3D(mode='RGB',
                    frame_width_RGB=1280,
                    frame_height_RGB=720,
                    frame_rate_RGB=20.0,
                    device_id='0')

# ---------- Tunables ----------
bottom_frac = 0.40      # use bottom 40% of frame for detection + centroiding
MIN_CONTOUR_AREA = 60   # ignore tiny specks
SEP_MIN_PX = 30         # min x-separation to treat two blobs as distinct
MIN_PTS = 20            # min points in a contour to trust centroid

# Steering
speed = 0.078
steering_gain = 0.0012
STEER_CLIP = 0.5
max_steering_angle = 28.0  # just for display

# Morphology
KERNEL = np.ones((5,5), np.uint8)
OPEN_ITERS = 1
CLOSE_ITERS = 1

# HSV thresholds
# Yellow (two bands to catch saturated + slightly washed)
Y1_LOWER = (15, 90, 80)
Y1_UPPER = (45, 255, 255)
Y2_LOWER = (15, 40, 60)
Y2_UPPER = (45, 255, 200)

# White lane paint: low saturation & high value (reject grey floor)
WHITE_S_MAX = 60
WHITE_V_MIN = 200
W_LOWER = (0, 0, WHITE_V_MIN)
W_UPPER = (180, WHITE_S_MAX, 255)

# Optional glare suppression (disabled by default)
SUPPRESS_GLARE = False
GLARE_LOWER = (0, 0, 240)
GLARE_UPPER = (180, 60, 255)

frame_count, fps, last_time = 0, 0, time.time()
WINDOW = 'Front 3D Camera View'
SHOW_MASK = False  # set True to see the binary mask window

# ---------- Helpers ----------
def find_contours(bin_img):
    res = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(res) == 2:
        cnts, hier = res
    else:
        _, cnts, hier = res
    return cnts, hier

def make_lane_mask_hsv(roi_bgr):
    """
    Binary mask where YELLOW or WHITE lane paint is 255, everything else 0.
    """
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

    # Yellow masks
    m1 = cv2.inRange(hsv, np.array(Y1_LOWER, np.uint8), np.array(Y1_UPPER, np.uint8))
    m2 = cv2.inRange(hsv, np.array(Y2_LOWER, np.uint8), np.array(Y2_UPPER, np.uint8))
    mask_yellow = cv2.bitwise_or(m1, m2)

    # White mask (low S, high V)
    mask_white = cv2.inRange(hsv, np.array(W_LOWER, np.uint8), np.array(W_UPPER, np.uint8))
    if SUPPRESS_GLARE:
        glare = cv2.inRange(hsv, np.array(GLARE_LOWER, np.uint8), np.array(GLARE_UPPER, np.uint8))
        mask_white = cv2.bitwise_and(mask_white, cv2.bitwise_not(glare))

    mask = cv2.bitwise_or(mask_yellow, mask_white)

    # Clean up morphology
    if OPEN_ITERS:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL, iterations=OPEN_ITERS)
    if CLOSE_ITERS:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL, iterations=CLOSE_ITERS)

    return mask

def contour_centroid(contour):
    pts = contour.reshape(-1, 2)
    if pts.shape[0] < MIN_PTS:
        return None
    cx = int(pts[:, 0].mean())
    cy = int(pts[:, 1].mean())
    return cx, cy

def get_lanes_bottom40(image_bgr):
    """
    Detect lanes in the bottom 40% of the frame using HSV (yellow|white).
    - Finds all blobs in that ROI
    - Picks two right-most and returns their mid-x (or single right-most)
    """
    h, w, _ = image_bgr.shape
    y0 = int(h * (1.0 - bottom_frac))     # start of bottom 40%
    roi = image_bgr[y0:h, 0:w]
    roi_h, roi_w = roi.shape[:2]

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
            "contour_full": c + np.array([0, y0]),  # shift to full-frame coords
            "cx_roi": cx,
            "cy_roi": cy
        })

    if not picks:
        return None

    # Choose two right-most blobs (or one)
    picks.sort(key=lambda d: d["cx_roi"])
    right_two = picks[-2:] if len(picks) >= 2 else picks[-1:]
    if len(right_two) == 2 and abs(right_two[1]["cx_roi"] - right_two[0]["cx_roi"]) < SEP_MIN_PX:
        right_two = [right_two[1]]

    if len(right_two) == 2:
        mid_x_roi = int((right_two[0]["cx_roi"] + right_two[1]["cx_roi"]) * 0.5)
        control_x_full = mid_x_roi  # ROI spans full width; x is same in full frame
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

def safe_terminate_camera3d(cam):
    for name in ("terminate_RGB", "stop_RGB", "stop_rgb", "stop"):
        if hasattr(cam, name):
            try: getattr(cam, name)()
            except Exception: pass
    try:
        cam.terminate()
    except AttributeError as e:
        if "video3d" not in str(e):
            raise
    except Exception:
        pass

# ---------- Main Loop ----------
try:
    while True:
        t0 = time.time()
        frontCam.read_RGB()
        img = frontCam.image_buffer_RGB
        if img is None or img.size == 0:
            print("Warning: Camera returned invalid image data.")
            continue

        h, w, _ = img.shape
        display = img.copy()
        info = get_lanes_bottom40(img)

        frame_count += 1
        now = time.time()
        if now - last_time >= 1.0:
            fps = frame_count
            frame_count = 0
            last_time = now

        desired_x = w // 2
        steering = 0.0

        if info is not None:
            (rx0, ry0) = info["roi_origin"]
            (rw, rh) = info["roi_size"]
            cv2.rectangle(display, (rx0, ry0), (rx0 + rw - 1, ry0 + rh - 1), (0, 255, 0), 2)

            # draw all blobs
            for b in info["blobs"]:
                cv2.drawContours(display, [b["contour_full"]], -1, (255, 200, 0), 2)
                cv2.circle(display, (b["cx_roi"], b["cy_roi"] + ry0), 6, (255, 0, 0), -1)

            # highlight the chosen right-side pair/single
            chosen = info["right_two"] if isinstance(info["right_two"], list) else [info["right_two"]]
            for b in chosen:
                cv2.drawContours(display, [b["contour_full"]], -1, (255, 0, 255), 3)
                cv2.circle(display, (b["cx_roi"], b["cy_roi"] + ry0), 8, (255, 0, 255), -1)

            ctrl_x = int(info["control_x_full"])
            cv2.circle(display, (ctrl_x, ry0 + rh // 2), 8, (0, 255, 255), -1)
            cv2.circle(display, (desired_x, ry0 + rh // 2), 8, (0, 0, 255), -1)

            error = desired_x - ctrl_x
            steering = float(np.clip(error * steering_gain, -STEER_CLIP, STEER_CLIP))
        else:
            cv2.putText(display, "NO YELLOW/WHITE LANE FOUND (bottom 40%)", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        calc_ms = (time.time() - t0) * 1000.0
        cv2.putText(display, f'FPS:{fps}  Calc:{calc_ms:.1f} ms', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.putText(display, f'Steer:{steering:+.3f}  Gain:{steering_gain}', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
        cv2.putText(display, f'Angle:{steering*max_steering_angle:+.1f} deg', (10, 90),
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
