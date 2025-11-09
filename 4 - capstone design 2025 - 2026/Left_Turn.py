# Follow-left-line implementation (based on Right_Turn logic)
# Uses left camera, crops region to ignore the yellow line, and follows the largest contour.
from Quanser.product_QCar import QCar
from Quanser.q_essential import Camera2D
import cv2
import numpy as np
import time
import math

# Camera and control params
leftCam = Camera2D(camera_id="1", frame_width=640, frame_height=480, frame_rate=30.0)

# Pixel offsets / tuning
# target_offset is horizontal shift from centre: negative moves target left
target_offset = -50
# steering gain tuned for the new crop/target location
steering_gain = 0.009
max_steering_angle = 28
steering_invert = False

# Speed mapping (same idea as Right_Turn)
SPEED_MAX = 0.078
SPEED_MIN = 0.068
SPEED_KP = 0.00007407

# HUD constants
HUD_X = 10
HUD_Y = 20
HUD_LINE_H = 24

# simple encoder helper fallback (copied minimal behavior)
class speedCalc:
    def __init__(self, qCar, t=None, counts_per_rev=31844, wheel_diameter_m=0.066):
        self.qCar = qCar
        self.t = time.time() if t is None else t
        self.counts_per_rev = float(counts_per_rev)
        self.wheel_diameter_m = float(wheel_diameter_m)
        try:
            v = self.qCar.read_encoder()
            if isinstance(v, (list, tuple, np.ndarray)):
                self.begin_encoder = int(v[0])
            else:
                self.begin_encoder = int(v)
        except Exception:
            self.begin_encoder = None

    def encoder_speed(self):
        now = time.time()
        totalTime = now - self.t
        self.t = now
        if totalTime <= 0:
            return 0.0
        try:
            v = self.qCar.read_encoder()
            if isinstance(v, (list, tuple, np.ndarray)):
                currentEncoder = int(v[0])
            else:
                currentEncoder = int(v)
        except Exception:
            return 0.0

        if self.begin_encoder is None:
            self.begin_encoder = currentEncoder
            return 0.0

        encoderChange = currentEncoder - self.begin_encoder
        self.begin_encoder = currentEncoder
        dist = (encoderChange / self.counts_per_rev) * (self.wheel_diameter_m * np.pi)
        return dist / totalTime

    def encoder_dist(self):
        try:
            v = self.qCar.read_encoder()
            if isinstance(v, (list, tuple, np.ndarray)):
                currentEncoder = int(v[0])
            else:
                currentEncoder = int(v)
        except Exception:
            return 0.0

        if self.begin_encoder is None:
            self.begin_encoder = currentEncoder
            return 0.0

        encoderChange = currentEncoder - self.begin_encoder
        self.begin_encoder = currentEncoder
        dist = (encoderChange / self.counts_per_rev) * (self.wheel_diameter_m * np.pi)
        return dist

    def encoder_cur(self):
        return self.qCar.read_encoder()

# Instantiate car helper
myCar = QCar()
speed_calc = speedCalc(myCar)
total_distance_m = 0.0

try:
    while True:
        start_calc = time.time()
        leftCam.read()
        img = leftCam.image_data
        if img is None or img.size == 0:
            print("Warning: left camera returned invalid image.")
            continue

        h, w, _ = img.shape
        display_img = img.copy()

        # Crop: focus on left-side area and ignore bottom 10% (to avoid yellow line)
        # Remove right 20% to emphasize left-most white line
        crop_x = 0
        crop_w = int(w * 0.8)  # keep left 80%
        crop_y = h // 2
        crop_h = int(h - int(h * 0.1)) - crop_y  # upper bound excludes bottom 10%
        # guard crop bounds
        crop_h = max(1, crop_h)
        right_crop = img[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]

        # Simple white detection: grayscale + high threshold
        gray = cv2.cvtColor(right_crop, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        steering = 0.0
        centroid_y_for_speed = None

        if contours:
            # pick the largest contour unconditionally
            largest = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                # translate centroid to full-image coordinates
                centroid_full = (crop_x + cx, crop_y + cy)
                # draw contour and centroid
                contour_full = largest + np.array([crop_x, crop_y])
                cv2.drawContours(display_img, [contour_full], -1, (255, 0, 0), 2)
                cv2.circle(display_img, centroid_full, 8, (255, 0, 0), -1)

                # compute target (in full-image coords)
                target_x = int(w * 0.5) + target_offset
                # move target_y up "quite a bit" so the car aims further ahead
                # choose a point roughly in the upper half of the processing region
                target_y = int(h * 0.35)  # moved up compared to right-turn logic
                # compute vertical error (same control strategy as right-turn)
                dy = int(centroid_full[1]) - int(target_y)
                if abs(dy) > 10:
                    steering = float(np.clip(dy * steering_gain, -0.5, 0.5))
                else:
                    steering = 0.0
                centroid_y_for_speed = int(centroid_full[1])

                # draw red target
                cv2.circle(display_img, (target_x, target_y), 10, (0,0,255), -1)
        else:
            steering = 0.0
            centroid_y_for_speed = None

        # apply steering invert if set
        steering_cmd = -steering if steering_invert else steering

        # dynamic speed mapping based on vertical alignment (same mapping)
        if centroid_y_for_speed is not None:
            prop = abs(target_y - int(centroid_y_for_speed)) + 1
            dynamic_speed = float(np.clip(SPEED_MAX - (SPEED_KP * float(prop)), SPEED_MIN, SPEED_MAX))
        else:
            dynamic_speed = float(SPEED_MIN)

        mtr_cmd = np.array([dynamic_speed, steering_cmd])
        LEDs = np.array([0,0,0,0,0,0,1,1])
        try:
            myCar.read_write_std(mtr_cmd, LEDs)
        except Exception:
            pass

        # encoder-derived stats (best-effort)
        try:
            speed_m_s = speed_calc.encoder_speed()
            dist_delta = speed_calc.encoder_dist()
            total_distance_m += (dist_delta / 31844.0) * (0.066 * math.pi) if isinstance(dist_delta, (int, float)) else 0.0
        except Exception:
            speed_m_s = 0.0

        # HUD
        cv2.putText(display_img, f'Steering: {steering:.3f}  Gain: {steering_gain}', (HUD_X, HUD_Y + HUD_LINE_H * 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        angle = steering * max_steering_angle
        cv2.putText(display_img, f'Angle: {angle:.1f} deg', (HUD_X, HUD_Y + HUD_LINE_H * 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,128,255), 2)
        cv2.putText(display_img, f'Speed cmd: {dynamic_speed:.3f} m/s  enc_m/s: {speed_m_s:.3f}',
                    (HUD_X, HUD_Y + HUD_LINE_H * 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)
        cv2.putText(display_img, f'Distance: {total_distance_m:.3f} m', (HUD_X, HUD_Y + HUD_LINE_H * 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        # Show region outline for debugging
        cv2.rectangle(display_img, (crop_x, crop_y), (crop_x + crop_w - 1, crop_y + crop_h - 1), (0,255,255), 2)

        cv2.namedWindow('Left Camera View', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Left Camera View', 1280, 960)
        cv2.imshow('Left Camera View', display_img)
        key = cv2.waitKey(1)
        if key == 27:
            break
        if key == ord('i'):
            steering_invert = not steering_invert
            print(f"Steering invert toggled: {steering_invert}")

        time.sleep(0.05)

finally:
    cv2.destroyAllWindows()
    try:
        myCar.terminate()
    except Exception:
        pass
    try:
        leftCam.terminate()
    except Exception:
        pass