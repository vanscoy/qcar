# QCar + front 3D camera, simple yellow follow on bottom 40% ROI
from Quanser.product_QCar import QCar
from Quanser.q_essential import Camera3D
import cv2, time, numpy as np

myCar = QCar()
frontCam = Camera3D(mode='RGB',
                    frame_width_RGB=1280,
                    frame_height_RGB=720,
                    frame_rate_RGB=20.0,
                    device_id='0')

# --- Tunables ---
bottom_frac = 0.40             # use bottom 40% of the image (full width)
band_frac   = 0.20      # NEW: use only bottom 20% of that ROI for centroid
MIN_BAND_PTS = 30       # fallback threshold for too-few points
target_offset_right = 1000      # 640 to be in the middle
speed = 0.078
steering_gain = 0.0012
max_steering_angle = 28

LOWER_YELLOW = np.array([15, 80, 80], dtype=np.uint8)
UPPER_YELLOW = np.array([40, 255, 255], dtype=np.uint8)
KERNEL = np.ones((5,5), np.uint8)
OPEN_ITERS = 1

frame_count, fps, last_time = 0, 0, time.time()
WINDOW = 'Front 3D Camera View'

# method for finding yellow in image
def make_yellow_mask(roi):
    # 1) HSV thresholds: normal yellow AND a paler variant (shade)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    #            H    S    V
    lower1 = (15,  90,  80)   # normal yellow (good saturation)
    upper1 = (45, 255, 255)
    lower2 = (15,  40,  60)   # pale yellow in shadow (lower S/V)
    upper2 = (45, 255, 200)

    m1 = cv2.inRange(hsv, lower1, upper1)
    m2 = cv2.inRange(hsv, lower2, upper2)
    mask_hsv = cv2.bitwise_or(m1, m2)

    # 2) Kill near-white/glare (very bright + desaturated)
    white_glare = cv2.inRange(hsv, (0, 0, 220), (180, 60, 255))
    mask = cv2.bitwise_and(mask_hsv, cv2.bitwise_not(white_glare))

    # 3) (Optional) reinforce yellow using LAB 'b' channel (yellow has high b)
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    # Otsu gets a good threshold for B in current lighting
    _, b_bin = cv2.threshold(B, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.bitwise_and(mask, b_bin)

    # 4) Clean up small specks
    KERNEL = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL, iterations=1)
    return mask

# method for finding bottom image and contour
def get_line_info_bottom(image):
    h, w, _ = image.shape
    y0 = int(h * (1.0 - bottom_frac))       # ROI = bottom 40% of full frame
    roi = image[y0:h, 0:w]

    # OLD: HSV yellow mask
    #hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    #mask = cv2.inRange(hsv, LOWER_YELLOW, UPPER_YELLOW)
    #if OPEN_ITERS:
    #    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL, iterations=OPEN_ITERS)

    mask = make_yellow_mask(roi)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    largest = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(largest) < 50:  # ignore tiny blobs
        return None

    # ---- Compute centroid using only the bottom band of the contour ----
    # contour points in ROI coords
    pts = largest.reshape(-1, 2)
    roi_h = roi.shape[0]
    band_y_start = int(roi_h * (1.0 - band_frac))        # bottom 20% start row (in ROI coords)
    band_pts = pts[pts[:, 1] >= band_y_start]

    if band_pts.shape[0] >= MIN_BAND_PTS:
        # centroid from bottom band only
        cx_band = float(band_pts[:, 0].mean())
        cy_band = float(band_pts[:, 1].mean())
        cx, cy = int(cx_band), int(cy_band)
    else:
        # Fallback: use bottom-most N points if band is too sparse
        N = min(50, pts.shape[0])
        idx = np.argsort(pts[:, 1])[-N:]   # take lowest (largest y)
        sel = pts[idx]
        cx, cy = int(sel[:,0].mean()), int(sel[:,1].mean())

    # Map for drawing on full frame
    contour_full = largest + np.array([0, y0])    # whole contour (for overlay)
    centroid_full = (cx, y0 + cy)                 # centroid of bottom band

    return {
        "contour": contour_full,
        "centroid": centroid_full,
        "cx_full": cx,                   # NOTE: already in full-frame X since ROI spans full width
        "roi_origin": (0, y0),
        "roi_size": (w, h - y0),
        "band_y_start_full": y0 + band_y_start  # for drawing the band box if you want
    }


def safe_terminate_camera3d(cam):
    """Avoid AttributeError when in RGB-only mode."""
    # If the API exposes an RGB-specific terminator, use it.
    for name in ("terminate_RGB", "stop_RGB", "stop_rgb", "stop"):
        if hasattr(cam, name):
            try: getattr(cam, name)()
            except Exception: pass
    # Fall back to generic terminate, but swallow the 'video3d' attribute error.
    try:
        cam.terminate()
    except AttributeError as e:
        # Typical when terminate() tries to access cam.video3d in RGB-only mode
        if "video3d" not in str(e):
            raise
    except Exception:
        pass

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

        frame_count += 1
        now = time.time()
        if now - last_time >= 1.0:
            fps = frame_count; frame_count = 0; last_time = now

        steering = 0.0
        if info is not None:
            desired_x = w - target_offset_right   # 640 px from right edge
            error = desired_x - info["cx_full"]
            steering = np.clip(error * steering_gain, -0.5, 0.5)  # apparently its clipped between -0.5 and 0.5 instead of -1 to 1. Who does that anyway like wtf

            (rx0, ry0) = info["roi_origin"]; (rw, rh) = info["roi_size"]
            cv2.rectangle(display, (rx0, ry0), (rx0+rw-1, ry0+rh-1), (0,255,0), 2)
            cv2.drawContours(display, [info["contour"]], -1, (255,0,0), 2)
            cv2.circle(display, info["centroid"], 8, (255,0,0), -1)
            cv2.circle(display, (desired_x, ry0 + rh//2), 8, (0,0,255), -1)
            #For the bottom 20%
            cv2.rectangle(display, (rx0, info["band_y_start_full"]),
                (rx0 + rw - 1, ry0 + rh - 1), (0, 200, 200), 2)  # bottom-20% band box
        else:
            cv2.putText(display, "NO YELLOW DETECTED", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        calc_ms = (time.time() - t0) * 1000.0
        cv2.putText(display, f'FPS:{fps}  Calc:{calc_ms:.1f} ms', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.putText(display, f'Steering:{steering:+.3f}  Gain:{steering_gain}', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
        cv2.putText(display, f'Angle:{steering*max_steering_angle:+.1f} deg', (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,128,255), 2)


        # Display (avoid resizing if VNC/Qt is flaky)
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.imshow(WINDOW, display)
        if (cv2.waitKey(1) & 0xFF) == 27:
            print("Kill switch activated: ESC pressed.")
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
    # <-- FIXED termination for RGB-only mode
    safe_terminate_camera3d(frontCam)
