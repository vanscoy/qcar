# qcar_linefollow_visual_steps.py
# QCar front-camera line-following VISUALIZATION ONLY (step-by-step)
# - Car speed = 0, steer = 0 (for taking pictures)
# - Shows four windows:
#     1) Original  : full RGB, nothing drawn
#     2) ROI_Only  : only bottom 40%, rest black, no annotations
#     3) YellowMask: ROI + yellow pixels highlighted, no band/contour/arrow
#     4) Final     : LAB grayscale ROI + band, contour, centroid, centerline & error arrow
#
# Keys:
#   ESC : quit
#   S/s : save all 4 images as lf_XX_*.png

from Quanser.product_QCar import QCar
from Quanser.q_essential import Camera3D
import cv2
import numpy as np
import time

# ---------------- Line-follow tunables (same as your FSM) ----------------
BOTTOM_FRAC   = 0.40   # bottom 40% of image
BAND_FRAC     = 0.20   # bottom 20% of that ROI
MIN_BAND_PTS  = 30

KERNEL5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# ---------------- Yellow mask (same logic as your FSM) ----------------
def make_yellow_mask(roi_bgr):
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

    # Two slightly different yellow ranges, OR'ed
    lower1 = (15,  90,  80)
    upper1 = (45, 255, 255)
    lower2 = (15,  40,  60)
    upper2 = (45, 255, 200)

    m1 = cv2.inRange(hsv, lower1, upper1)
    m2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(m1, m2)

    # Kill white glare
    white_glare = cv2.inRange(hsv, (0, 0, 220), (180, 60, 255))
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(white_glare))

    # LAB b-channel refiner (like your FSM)
    lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)
    _, b_bin = cv2.threshold(lab[:, :, 2], 0, 255,
                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.bitwise_and(mask, b_bin)

    # Clean up
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL5, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL5, iterations=1)
    return mask, lab

def main():
    # Init car (but we will NOT move it)
    car = QCar()

    # Front RGB-only camera
    cam = Camera3D(mode='RGB',
                   frame_width_RGB=1280,
                   frame_height_RGB=720,
                   frame_rate_RGB=20.0,
                   device_id='0')

    cv2.namedWindow("Original",   cv2.WINDOW_NORMAL)
    cv2.namedWindow("ROI_Only",   cv2.WINDOW_NORMAL)
    cv2.namedWindow("YellowMask", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Final",      cv2.WINDOW_NORMAL)

    cv2.resizeWindow("Original",   960, 540)
    cv2.resizeWindow("ROI_Only",   960, 540)
    cv2.resizeWindow("YellowMask", 960, 540)
    cv2.resizeWindow("Final",      960, 540)

    shot_idx = 0

    try:
        while True:
            cam.read_RGB()
            rgb = cam.image_buffer_RGB

            if rgb is None or rgb.size == 0:
                key = (cv2.waitKey(1) & 0xFF)
                if key == 27:
                    break
                continue

            h, w, _ = rgb.shape
            center_x = w // 2

            # ---- Step 1: Original (no annotations) ----
            img_original = rgb.copy()

            # ---- Basic ROI calculations (always, even if no yellow) ----
            y0 = int(h * (1.0 - BOTTOM_FRAC))  # top of ROI
            roi = rgb[y0:h, :]

            # ---- Step 2: ROI_Only (bottom 40% only, no annotations) ----
            img_roi = np.zeros_like(rgb)
            img_roi[y0:h, :] = roi.copy()

            # ---- Mask + LAB (for later steps) ----
            mask, lab = make_yellow_mask(roi)

            # ---- Step 3: YellowMask (ROI + yellow highlighted, no band/contour/arrow) ----
            img_yellow = np.zeros_like(rgb)
            yellow_vis_roi = np.zeros_like(roi)
            yellow_vis_roi[mask > 0] = (0, 255, 255)  # yellow colored
            img_yellow[y0:h, :] = yellow_vis_roi

            # ---- Step 4 base: LAB grayscale ROI ----
            img_final = np.zeros_like(rgb)
            b_chan = lab[:, :, 2]
            b_norm = cv2.normalize(b_chan, None, 0, 255,
                                   cv2.NORM_MINMAX).astype(np.uint8)
            gray3 = cv2.cvtColor(b_norm, cv2.COLOR_GRAY2BGR)
            img_final[y0:h, :] = gray3

            # ---- Contour / centroid / band for final step ----
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

            if cnts:
                largest = max(cnts, key=cv2.contourArea)
                if cv2.contourArea(largest) >= 50:
                    pts = largest.reshape(-1, 2)
                    roi_h = roi.shape[0]
                    band_y_start = int(roi_h * (1.0 - BAND_FRAC))

                    # Points in the bottom band
                    band_pts = pts[pts[:, 1] >= band_y_start]

                    if band_pts.shape[0] >= MIN_BAND_PTS:
                        cx = int(float(band_pts[:, 0].mean()))
                        cy = int(float(band_pts[:, 1].mean()))
                    else:
                        # fallback: bottom-most ~50 points
                        N = min(50, pts.shape[0])
                        sel = pts[np.argsort(pts[:, 1])[-N:]]
                        cx, cy = int(sel[:, 0].mean()), int(sel[:, 1].mean())

                    cx_full = cx
                    cy_full = y0 + cy
                    contour_full = largest + np.array([0, y0])
                    band_y_start_full = y0 + band_y_start

                    error = cx_full - center_x

                    # --- Draw *only* on FINAL image ---
                    # ROI box
                    cv2.rectangle(img_final, (0, y0),
                                  (w - 1, h - 1),
                                  (0, 255, 0), 2)

                    # Band box
                    cv2.rectangle(img_final, (0, band_y_start_full),
                                  (w - 1, h - 1),
                                  (0, 200, 200), 2)

                    # Contour & centroid
                    cv2.drawContours(img_final, [contour_full], -1,
                                     (255, 0, 0), 2)
                    cv2.circle(img_final, (cx_full, cy_full), 7,
                               (255, 0, 0), -1)

                    # Centerline
                    cv2.line(img_final, (center_x, 0),
                             (center_x, h - 1),
                             (255, 255, 255), 1)

                    # Error arrow (center -> centroid) at mid-band height
                    arrow_y = int((band_y_start_full + h - 1) / 2)
                    cv2.arrowedLine(img_final,
                                    (center_x, arrow_y),
                                    (cx_full, arrow_y),
                                    (0, 0, 255), 2,
                                    tipLength=0.08)

                    # Text
                    cv2.putText(img_final, f"err = {error:+d} px",
                                (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (0, 255, 255), 2)
                else:
                    cv2.putText(img_final, "NO VALID CONTOUR",
                                (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (0, 0, 255), 2)
            else:
                cv2.putText(img_final, "NO YELLOW FOUND",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 0, 255), 2)

            # ---- Show windows ----
            cv2.imshow("Original",   img_original)
            cv2.imshow("ROI_Only",   img_roi)
            cv2.imshow("YellowMask", img_yellow)
            cv2.imshow("Final",      img_final)

            # Keep car stopped
            mtr_cmd = np.array([0.0, 0.0], dtype=np.float64)
            LEDs    = np.zeros(8, dtype=np.float64)
            car.read_write_std(mtr_cmd, LEDs)

            key = (cv2.waitKey(1) & 0xFF)
            if key == 27:  # ESC
                print("ESC -> exit")
                break
            elif key in (ord('s'), ord('S')):
                # Save all 4 images
                base = f"lf_{shot_idx:02d}"
                cv2.imwrite(base + "_original.png",   img_original)
                cv2.imwrite(base + "_roi.png",        img_roi)
                cv2.imwrite(base + "_yellow.png",     img_yellow)
                cv2.imwrite(base + "_final.png",      img_final)
                print(f"Saved images for index {shot_idx:02d}")
                shot_idx += 1

            time.sleep(0.01)

    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        try:
            cam.terminate()
        except Exception:
            pass
        try:
            car.terminate()
        except Exception:
            pass

if __name__ == "__main__":
    main()
