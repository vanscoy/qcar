# qcar_linefollow_visual_steps.py
# QCar front-camera line-following VISUALIZATION ONLY (step-by-step)
# - Car speed = 0, steer = 0 (for taking pictures)
# - Shows five windows:
#     1) Original   : full RGB, nothing drawn
#     2) ROI_Only   : only bottom 40%, rest black, no annotations
#     3) YellowMask : ROI + yellow pixels highlighted (yellow color)
#     4) GrayMask   : AFTER YellowMask -> yellow=white, rest=black
#     5) Final      : grayscale mask ROI + band, contour, centroid, targets, error, steer, angle
#
# Keys:
#   ESC : quit
#   S/s : save all 5 images as lf_XX_*.png

from Quanser.product_QCar import QCar
from Quanser.q_essential import Camera3D
import cv2
import numpy as np
import time

# ---------------- Tunables (match your FSM) ----------------
BOTTOM_FRAC   = 0.40   # bottom 40% of image
BAND_FRAC     = 0.20   # bottom 20% of that ROI
MIN_BAND_PTS  = 30

TARGET_OFFSET_RIGHT = 1000      # same as FSM
Kp = 0.0012                     # same as FSM
STEER_CLIP = 0.5                # same as FSM
MAX_STEER_ANGLE_DEG = 28        # mapping you used

KERNEL5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# ---------------- Yellow mask (same logic as FSM) ----------------
def make_yellow_mask(roi_bgr):
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

    lower1 = (15,  90,  80); upper1 = (45, 255, 255)
    lower2 = (15,  40,  60); upper2 = (45, 255, 200)

    m1 = cv2.inRange(hsv, lower1, upper1)
    m2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(m1, m2)

    white_glare = cv2.inRange(hsv, (0, 0, 220), (180, 60, 255))
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(white_glare))

    lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)
    _, b_bin = cv2.threshold(lab[:, :, 2], 0, 255,
                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.bitwise_and(mask, b_bin)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL5, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL5, iterations=1)
    return mask

def main():
    car = QCar()

    cam = Camera3D(mode='RGB',
                   frame_width_RGB=1280,
                   frame_height_RGB=720,
                   frame_rate_RGB=20.0,
                   device_id='0')

    # Windows
    for name in ["Original", "ROI_Only", "YellowMask", "GrayMask", "Final"]:
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, 960, 540)

    shot_idx = 0

    try:
        while True:
            cam.read_RGB()
            rgb = cam.image_buffer_RGB

            if rgb is None or rgb.size == 0:
                key = (cv2.waitKey(1) & 0xFF)
                if key == 27: break
                continue

            h, w, _ = rgb.shape

            # Target column (same meaning as FSM)
            x_target = w - TARGET_OFFSET_RIGHT

            # ---------------- Step 1: Original ----------------
            img_original = rgb.copy()

            # ROI
            y0 = int(h * (1.0 - BOTTOM_FRAC))
            roi = rgb[y0:h, :]

            # ---------------- Step 2: ROI_Only ----------------
            img_roi = np.zeros_like(rgb)
            img_roi[y0:h, :] = roi.copy()

            # Yellow mask
            mask = make_yellow_mask(roi)

            # ---------------- Step 3: YellowMask (yellow pixels shown) ----------------
            img_yellow = np.zeros_like(rgb)
            yellow_vis_roi = np.zeros_like(roi)
            yellow_vis_roi[mask > 0] = (0, 255, 255)  # yellow
            img_yellow[y0:h, :] = yellow_vis_roi

            # ---------------- Step 4: GrayMask AFTER yellowmask ----------------
            # yellow->white, everything else->black
            img_graymask = np.zeros_like(rgb)
            gray_roi = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # 0/255 -> black/white
            img_graymask[y0:h, :] = gray_roi

            # ---------------- Step 5 base: Final grayscale mask ----------------
            img_final = img_graymask.copy()

            # Contour / centroid from bottom band
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

            if cnts:
                largest = max(cnts, key=cv2.contourArea)
                if cv2.contourArea(largest) >= 50:
                    pts = largest.reshape(-1, 2)
                    roi_h = roi.shape[0]
                    band_y_start = int(roi_h * (1.0 - BAND_FRAC))
                    band_pts = pts[pts[:, 1] >= band_y_start]

                    # centroid ONLY from bottom band
                    if band_pts.shape[0] >= MIN_BAND_PTS:
                        cx = int(float(band_pts[:, 0].mean()))
                        cy = int(float(band_pts[:, 1].mean()))
                    else:
                        N = min(50, pts.shape[0])
                        sel = pts[np.argsort(pts[:, 1])[-N:]]
                        cx, cy = int(sel[:, 0].mean()), int(sel[:, 1].mean())

                    cx_full = cx
                    cy_full = y0 + cy
                    contour_full = largest + np.array([0, y0])
                    band_y_start_full = y0 + band_y_start

                    # correct error: e = x_target - Cx
                    error = x_target - cx_full

                    # steer + angle
                    steer = float(np.clip(Kp * error, -STEER_CLIP, STEER_CLIP))
                    angle_deg = steer * MAX_STEER_ANGLE_DEG

                    # ---- Draw on FINAL ----
                    cv2.rectangle(img_final, (0, y0), (w-1, h-1), (0,255,0), 2)
                    cv2.rectangle(img_final, (0, band_y_start_full), (w-1, h-1), (0,200,200), 2)

                    cv2.drawContours(img_final, [contour_full], -1, (255,0,0), 2)
                    cv2.circle(img_final, (cx_full, cy_full), 7, (255,0,0), -1)

                    # arrow y in middle of band
                    arrow_y = int((band_y_start_full + h - 1) / 2)

                    # x_target marker
                    cv2.circle(img_final, (x_target, arrow_y), 7, (0,255,0), -1)
                    cv2.putText(img_final, "x_target", (x_target-45, arrow_y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

                    # error arrow (x_target -> Cx)
                    cv2.arrowedLine(img_final,
                                    (x_target, arrow_y),
                                    (cx_full, arrow_y),
                                    (0,0,255), 2, tipLength=0.08)

                    # labels
                    cv2.putText(img_final, f"Cx={cx_full}", (cx_full-30, cy_full-12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

                    cv2.putText(img_final, f"e={error:+d}px", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
                    cv2.putText(img_final, f"steer={steer:+.3f}, angle={angle_deg:+.1f} deg",
                                (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
                else:
                    cv2.putText(img_final, "NO VALID CONTOUR", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            else:
                cv2.putText(img_final, "NO YELLOW FOUND", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

            # Show windows
            cv2.imshow("Original",   img_original)
            cv2.imshow("ROI_Only",   img_roi)
            cv2.imshow("YellowMask", img_yellow)
            cv2.imshow("GrayMask",   img_graymask)
            cv2.imshow("Final",      img_final)

            # keep car stopped
            mtr_cmd = np.array([0.0, 0.0], dtype=np.float64)
            LEDs    = np.zeros(8, dtype=np.float64)
            car.read_write_std(mtr_cmd, LEDs)

            key = (cv2.waitKey(1) & 0xFF)
            if key == 27:
                break
            elif key in (ord('s'), ord('S')):
                base = f"lf_{shot_idx:02d}"
                cv2.imwrite(base + "_original.png",  img_original)
                cv2.imwrite(base + "_roi.png",       img_roi)
                cv2.imwrite(base + "_yellow.png",    img_yellow)
                cv2.imwrite(base + "_graymask.png",  img_graymask)
                cv2.imwrite(base + "_final.png",     img_final)
                print(f"Saved images for index {shot_idx:02d}")
                shot_idx += 1

            time.sleep(0.01)

    finally:
        try: cv2.destroyAllWindows()
        except Exception: pass
        try: cam.terminate()
        except Exception: pass
        try: car.terminate()
        except Exception: pass

if __name__ == "__main__":
    main()
