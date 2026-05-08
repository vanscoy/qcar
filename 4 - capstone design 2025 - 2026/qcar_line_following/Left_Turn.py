# Import QCar class for robot control
from Quanser.product_QCar import QCar
# Import Camera2D class for image capture
from Quanser.q_essential import Camera2D
# Import OpenCV for image processing
import cv2
# Import NumPy for numerical operations
import numpy as np
# Import time for delays and timing
import time

# Diagnostic flag: when True, print and save myCar.read() and dir(myCar) once at startup
# Default False; use --diagnostics to enable at startup
DIAGNOSTICS_ON = False

FALLBACK_RET0_AS_RPS = False


myCar = QCar()
# Create left camera object (user requested camera_id=2)
leftCam = Camera2D(camera_id="2", frame_width=640, frame_height=480, frame_rate=30.0)

# Desired pixel offset from right edge for line following
target_offset = 50
# Forward speed of the robot (lower value for slower movement)
speed = 0.07
steering_gain = 0.02  # Gain used for steering calculation (increased per user request)
max_steering_angle = 28  # Maximum steering angle in degrees (mechanical limit)
# Runtime steering invert: when True steering is multiplied by -1 before sending
steering_invert = True  # inverted steering applied via runtime flag; set False to flip direction

SPEED_MAX = 0.078
SPEED_MIN = 0.072
# Recompute SPEED_KP so SPEED_MIN = SPEED_MAX - SPEED_KP * prop_max
# where prop = abs(target_y - centroid_y) + 1 and target/centroid are in pixels.
# Current frame height = 480, vertical crop from 45%->65% => crop_h = 96 px.
# Maximum |target_y-centroid_y| ≈ crop_h/2 = 48, so prop_max ≈ 48 + 1 = 49.
# Therefore SPEED_KP = (SPEED_MAX - SPEED_MIN) / prop_max = 0.006 / 49 ≈ 8.16e-05
SPEED_KP = 8.16e-05


Y_IGNORE_THRESHOLD = 100
X_IGNORE_THRESHOLD = 100

last_accepted_centroid = None

# Frame counter and FPS calculation
frame_count = 0
fps = 0
last_time = time.time()
# HUD layout constants (top-left column)
HUD_X = 10
HUD_Y = 20
HUD_LINE_H = 24


class speedCalc:
    """Helper to compute speed (m/s) and distance (m) using a single encoder
    reading from the QCar. This is defensive: if read_encoder() isn't available
    it will return zeros instead of raising.
    """
    def __init__(self, qCar, t=None, counts_per_rev=31844, wheel_diameter_m=0.066):
        self.qCar = qCar
        self.t = time.time() if t is None else t
        self.counts_per_rev = float(counts_per_rev)
        self.wheel_diameter_m = float(wheel_diameter_m)
        # seed begin_encoder if possible
        try:
            v = self.qCar.read_encoder()
            if isinstance(v, (list, tuple, np.ndarray)):
                self.begin_encoder = int(v[0])
            else:
                self.begin_encoder = int(v)
        except Exception:
            self.begin_encoder = None

    def elapsed_time(self):
        return time.time() - self.t

    def encoder_speed(self):
        """Return speed in m/s measured since last call. Updates internal timer
        and begin_encoder so consecutive calls return relative speed.
        """
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
        # distance per count = (pi * diameter) / counts_per_rev
        dist = (encoderChange / self.counts_per_rev) * (self.wheel_diameter_m * np.pi)
        return dist / totalTime

    def encoder_dist(self):
        """Return incremental distance (m) since last call and update the
        stored encoder baseline.
        """
        try:
            v = self.qCar.read_encoder()
            if isinstance(v, (list, tuple, np.ndarray)):
                currentEncoder = int(v[0])
            else:
                currentEncoder = int(v)
        except Exception:
            return 0.0

        if self.begin_encoder is None:
            # Seed baseline on first successful read and return zero distance
            self.begin_encoder = currentEncoder
            return 0.0

        encoderChange = currentEncoder - self.begin_encoder
        self.begin_encoder = currentEncoder
        # use configured counts_per_rev and wheel_diameter_m for distance
        dist = (encoderChange / self.counts_per_rev) * (self.wheel_diameter_m * np.pi)
        return dist

    def encoder_cur(self):
        return self.qCar.read_encoder()


# instantiate simplified encoder helper and distance accumulator
speed_calc = speedCalc(myCar)
total_distance_m = 0.0

def get_right_line_offset(image):
    h, w, _ = image.shape  # Get image dimensions
    # Use the same crop fractions as the main loop so detections lie inside the drawn box
    crop_x = int(w * 0.3)       # remove left 30%
    right_crop = int(w * 0.8)   # remove right 20%
    top_crop = int(h * 0.45)
    bottom_crop = int(h * 0.65)

    # Crop the image to the processing rectangle and run the same threshold/contour pipeline
    proc = image[top_crop:bottom_crop, crop_x:right_crop]
    gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    overlay_info = None
    if contours:
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M.get('m00', 0) > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            # Convert contour and centroid back to full-image coordinates
            contour_full = largest + np.array([crop_x, top_crop])
            centroid_full = (crop_x + cx, top_crop + cy)
            overlay_info = {
                'contour': contour_full,
                'centroid': centroid_full,
                'offset': crop_x + cx
            }
    return overlay_info, thresh

try:
    while True:  # Main control loop

        # Capture image from left camera
        leftCam.read()
        img = leftCam.image_data

        # Check for valid image data
        if img is None or img.shape[0] == 0 or img.shape[1] == 0:
            print("Warning: Camera returned invalid image data.")
            time.sleep(0.05)
            continue

        h, w, _ = img.shape  # Get image dimensions
        display_img = img.copy()  # Show full camera view

        # Draw processing-area outline (keep vertical band from 45%->65%, remove left 30% and right 20%)
        crop_x = int(w * 0.3)
        right_crop = int(w * 0.8)
        crop_y = int(h * 0.45)  # top of the kept vertical band (45% down)
        bottom_crop = int(h * 0.65)  # bottom moved down to 65% to include more lower frame
        crop_w = right_crop - crop_x
        crop_h = bottom_crop - crop_y

        overlay_info, thresh = get_right_line_offset(img)  # Get overlay and thresh from crop

        cv2.rectangle(display_img, (crop_x, crop_y), (crop_x + crop_w - 1, crop_y + crop_h - 1), (0, 255, 255), 2)
        cv2.line(display_img, (crop_x, crop_y), (crop_x, crop_y + crop_h - 1), (0,0,255), 2)
        cv2.line(display_img, (crop_x + crop_w - 1, crop_y), (crop_x + crop_w - 1, crop_y + crop_h - 1), (0,255,0), 2)

        # Draw overlays and frame info
        if overlay_info is not None:
            # Compute on-screen target X (full-image coords)
            target_x = int(w * 0.5) + target_offset

            # Get centroid of the largest contour
            centroid_x, centroid_y = overlay_info['centroid']
            # Target vertically centered in the crop, moved up 15 pixels from previous setting
            target_y = crop_y + (crop_h // 2)

            # Simple Y-based P-control: steer proportionally to vertical offset
            dy = int(centroid_y) - int(target_y)
            if abs(dy) > 10:
                steering = float(np.clip(dy * steering_gain, -0.5, 0.5))
                control_mode = 'Y'
            else:
                steering = 0.0
                control_mode = 'aligned'

            centroid_y_for_speed = int(centroid_y)

            # Draw contour and centroid
            try:
                cv2.drawContours(display_img, [overlay_info['contour']], -1, (255,0,0), 2)
                cv2.circle(display_img, overlay_info['centroid'], 10, (255,0,0), -1)
            except Exception:
                pass

            # Draw target position as red dot
            cv2.circle(display_img, (target_x, target_y), 10, (0,0,255), -1)
            try:
                cv2.putText(display_img, f'dy: {(int(centroid_y)-int(target_y)):+d} ctrl:{control_mode}',
                            (HUD_X, HUD_Y + HUD_LINE_H * 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 2)
            except Exception:
                pass
        else:
            steering = 0  # No line detected, go straight
            centroid_y_for_speed = None


        # Prepare motor command (apply runtime invert if enabled)
        steering_cmd = -steering if steering_invert else steering

        # Dynamic forward speed based on vertical alignment (pixel Y distance)
        if 'centroid_y_for_speed' in locals() and centroid_y_for_speed is not None:
            prop = abs(target_y - int(centroid_y_for_speed)) + 1
            dynamic_speed = float(np.clip(SPEED_MAX - (SPEED_KP * float(prop)), SPEED_MIN, SPEED_MAX))
        else:
            dynamic_speed = float(SPEED_MIN)

        mtr_cmd = np.array([dynamic_speed, steering_cmd])
        LEDs = np.array([0, 0, 0, 0, 0, 0, 1, 1])

        # Send motor command
        try:
            myCar.read_write_std(mtr_cmd, LEDs)
        except Exception:
            pass

        # Encoders are disabled in this trimmed script; provide zeroed values
        speed_m_s = 0.0
        dist_delta = 0.0
        total_distance_m = 0.0

        # HUD overlays
        cv2.putText(display_img, f'Frames: {frame_count}  FPS: {fps}', (HUD_X, HUD_Y + HUD_LINE_H * 0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.putText(display_img, f'Steering: {steering:.3f}  Gain: {steering_gain}', (HUD_X, HUD_Y + HUD_LINE_H * 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        angle = steering * max_steering_angle
        cv2.putText(display_img, f'Angle: {angle:.1f} deg', (HUD_X, HUD_Y + HUD_LINE_H * 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,128,255), 2)
        # show commanded speed (encoders removed)
        cv2.putText(display_img, f'Speed cmd: {dynamic_speed:.3f} m/s', (HUD_X, HUD_Y + HUD_LINE_H * 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)

        # Resize and show window
        cv2.namedWindow('Left Camera View', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Left Camera View', 1280, 960)
        cv2.imshow('Left Camera View', display_img)

        # Show a black & white thresholded view for debugging
        try:
            cv2.namedWindow('Thresh', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Thresh', 640, 480)
            cv2.imshow('Thresh', thresh)
        except Exception:
            pass

        key = cv2.waitKey(1)
        # Kill switch: ESC key (27) to exit
        if key == 27:
            print("Kill switch activated: ESC pressed.")
            break
        # (steering invert toggle removed) -- steering_invert is fixed by variable above

        # Send motor command once per loop (safe write)
        LEDs = np.array([0, 0, 0, 0, 0, 0, 1, 1])
        try:
            myCar.read_write_std(mtr_cmd, LEDs)
        except Exception:
            pass

        time.sleep(0.05)  # Small delay for control loop timing
finally:
    cv2.destroyAllWindows()  # Close all OpenCV windows
    myCar.terminate()  # Terminate QCar connection
    leftCam.terminate()  # Terminate camera connection
