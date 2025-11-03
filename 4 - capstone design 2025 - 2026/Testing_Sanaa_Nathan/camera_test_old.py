from Quanser.q_essential import Camera3D
import Quanser.q_interpretation as qi
import time, numpy as np, cv2

FPS = 30
SIM_TIME = 30.0
IMG_W, IMG_H = 1280, 720
MIN_Z, MAX_Z = 1.0, 3.0

cam = Camera3D(mode='RGB&DEPTH', frame_width_RGB=IMG_W, frame_height_RGB=IMG_H)

t0 = time.time()
try:
    while (time.time() - t0) < SIM_TIME:          # <-- LOOP starts here
        ok_rgb = cam.read_RGB()
        ok_d   = cam.read_depth(dataMode='m')

        if not ok_rgb or not ok_d:
            print(f"[WARN] read_RGB={ok_rgb}, read_depth={ok_d}")
            cv2.waitKey(1)
            continue                               # <-- INSIDE the loop

        rgb = cam.image_buffer_RGB
        depth_m = cam.image_buffer_depth_m
        if rgb is None or depth_m is None or rgb.size == 0 or depth_m.size == 0:
            print("[WARN] Empty buffers")
            cv2.waitKey(1)
            continue                               # <-- INSIDE the loop

        # --- your processing/visualization here ---
        binary01 = qi.binary_thresholding(depth_m, MIN_Z, MAX_Z).astype(np.uint8)
        mask255 = (binary01 * 255).astype(np.uint8)
        masked = cv2.bitwise_and(rgb, rgb, mask=mask255)

        cv2.imshow("RGB x MASK", cv2.resize(masked, (640, 360)))
        if (cv2.waitKey(1) & 0xFF) in (27, ord('q')):
            break

except KeyboardInterrupt:
    print("User interrupted")
finally:
    cam.terminate()
    cv2.destroyAllWindows()
