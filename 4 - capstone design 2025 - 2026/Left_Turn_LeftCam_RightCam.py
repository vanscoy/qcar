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
# If no discrete encoder counts are exposed by the model/driver, fall back to using
# the first element returned by read_write_std() as a motor velocity (revolutions/sec).
# Set to False to disable this heuristic.
FALLBACK_RET0_AS_RPS = False


# NOTE: Diagnostics/probe helper functions were removed to keep the runtime script
# small and focused. The remaining helper functions `read_encoder_velocity` and
# `read_encoders` are retained because they're used at runtime to obtain encoder
# values when available.

# Parse CLI args for diagnostics toggle

# Create QCar object for robot control
myCar = QCar()
# Create right camera object
rightCam = Camera2D(camera_id="0", frame_width=640, frame_height=480, frame_rate=30.0)
# Create left camera object (for the initial phase)
leftCam = Camera2D(camera_id="2", frame_width=640, frame_height=480, frame_rate=30.0)

# Desired pixel offset from right edge for line following
target_offset = 50
# Forward speed of the robot (lower value for slower movement)
speed = 0.072
steering_gain = 0.009  # Gain used for steering calculation (increased per user request)
max_steering_angle = 28  # Maximum steering angle in degrees (mechanical limit)
# Runtime steering invert toggle: when True steering is multiplied by -1 before sending
steering_invert = False
# Speed control constants for gain-based slowing during large turns
# Updated per user: increase min speed to avoid stalling and bump max speed
# New values: SPEED_MIN = 0.068, SPEED_MAX = 0.072
# Kp chosen so that 0.068 = 0.072 - Kp * (135) -> Kp ~= 2.96296e-05
SPEED_MAX = 0.072
SPEED_MIN = 0.068
SPEED_KP = 2.96296e-05

# Contour acceptance thresholds (pixels). If a new detection differs from the
# last accepted centroid by more than these thresholds it will be considered
# spurious/incorrect. Both X and Y must be within the thresholds to accept.
Y_IGNORE_THRESHOLD = 100
X_IGNORE_THRESHOLD = 100

# Last accepted centroid (full-image coords) as (x, y). Initialized to None
# and set on first valid detection. Used to filter spurious contours (e.g.,
# large yellow lines) and to provide a fallback when ignored contours appear.
last_accepted_centroid = None

# Frame counter and FPS calculation
frame_count = 0
fps = 0
last_time = time.time()
# HUD layout constants (top-left column)
HUD_X = 10
HUD_Y = 20
HUD_LINE_H = 24

# Duration (seconds) for the initial left-camera phase
LEFT_PHASE_TIME = 10.0

# ...existing code...

# --- Simplified encoder helper (user-provided, defensive) -------------------
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


# Note: removed low-level encoder probing helpers to keep runtime simple.
# If you need advanced probing later, reintroduce a minimal helper that
# calls myCar.read_encoder() or parses read_write_std() returns.

# Function to find the x-position of the detected line in the right crop
def get_right_line_offset(image):
    h, w, _ = image.shape  # Get image dimensions
    # Crop lower half but remove left 20% of image for line detection
    crop_x = int(w * 0.2)  # remove left 20%
    lower_half = image[h//2:h, crop_x:]  # lower half with left 20% removed
    gray = cv2.cvtColor(lower_half, cv2.COLOR_BGR2GRAY)  # Convert cropped image to grayscale
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)  # Threshold to highlight bright lines
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours
    overlay_info = None
    if contours:  # If any contours are found
        largest = max(contours, key=cv2.contourArea)  # Select the largest contour (assumed to be the line)
        M = cv2.moments(largest)  # Calculate moments for the largest contour
        if M['m00'] > 0:  # Prevent division by zero
            cx = int(M['m10'] / M['m00'])  # Compute center x-position of the contour (in right_crop)
            cy = int(M['m01'] / M['m00'])  # Compute center y-position of the contour (in right_crop)
            # Prepare overlay info in full-image coordinates
            contour_full = largest + np.array([crop_x, h//2])
            centroid_full = (crop_x + cx, h//2 + cy)
            overlay_info = {
                'contour': contour_full,
                'centroid': centroid_full,
                # store offset as full-image X so main loop can compare directly to target_x
                'offset': crop_x + cx
            }
            return overlay_info  # Return overlay info
    return None  # Return None if no line is found


# Function to find the line in the left-camera crop (white-only mask, narrow vertical band)
def get_left_line_offset(image):
    h, w, _ = image.shape
    # remove left 20% and right 20% -> keep middle 60% horizontally
    crop_x = int(w * 0.2)
    right_crop = int(w * 0.8)
    # vertical band around center: 45% -> 65% (widened to capture more below)
    top_crop = int(h * 0.45)
    bottom_crop = int(h * 0.65)
    crop = image[top_crop:bottom_crop, crop_x:right_crop]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200], dtype=np.uint8)
    upper_white = np.array([180, 60, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_white, upper_white)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay_info = None
    if contours:
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M.get('m00', 0) > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            contour_full = largest + np.array([crop_x, top_crop])
            centroid_full = (crop_x + cx, top_crop + cy)
            overlay_info = {'contour': contour_full, 'centroid': centroid_full, 'offset': crop_x + cx}
    return overlay_info, mask

try:
    kill = False

    # --- Phase 1: left-camera closed-loop follow for LEFT_PHASE_TIME seconds ---
    left_phase_start = time.time()
    while time.time() - left_phase_start < LEFT_PHASE_TIME and not kill:
        start_calc = time.time()
        leftCam.read()
        img = leftCam.image_data

        if img is None or img.shape[0] == 0 or img.shape[1] == 0:
            print("Warning: Left camera returned invalid image data.")
            time.sleep(0.05)
            continue

        overlay_info, thresh = get_left_line_offset(img)

        h, w, _ = img.shape
        display_img = img.copy()

        # Update frame counter and FPS
        frame_count += 1
        current_time = time.time()
        if current_time - last_time >= 1.0:
            fps = frame_count
            frame_count = 0
            last_time = current_time

    # Draw processing-area outline (vertical band, middle 60% horizontally)
    crop_x = int(w * 0.2)
    right_crop = int(w * 0.8)
    crop_y = int(h * 0.45)
    bottom_crop = int(h * 0.65)
        crop_w = right_crop - crop_x
        crop_h = bottom_crop - crop_y
        cv2.rectangle(display_img, (crop_x, crop_y), (crop_x + crop_w - 1, crop_y + crop_h - 1), (0, 255, 255), 2)
        cv2.line(display_img, (crop_x, crop_y), (crop_x, crop_y + crop_h - 1), (0,0,255), 2)
        cv2.line(display_img, (crop_x + crop_w - 1, crop_y), (crop_x + crop_w - 1, crop_y + crop_h - 1), (0,255,0), 2)

        if overlay_info is not None:
            target_x = int(w * 0.5) + target_offset
            centroid_x, centroid_y = overlay_info['centroid']
            target_y = crop_y + (crop_h // 2)
            dy = int(centroid_y) - int(target_y)
            if abs(dy) > 10:
                steering = float(np.clip(dy * steering_gain, -0.5, 0.5))
                control_mode = 'Y'
            else:
                steering = 0.0
                control_mode = 'aligned'

            centroid_y_for_speed = int(centroid_y)

            try:
                cv2.drawContours(display_img, [overlay_info['contour']], -1, (255,0,0), 2)
                cv2.circle(display_img, overlay_info['centroid'], 10, (255,0,0), -1)
            except Exception:
                pass

            cv2.circle(display_img, (target_x, target_y), 10, (0,0,255), -1)
            try:
                cv2.putText(display_img, f'dy: {(int(centroid_y)-int(target_y)):+d} ctrl:{control_mode}',
                            (HUD_X, HUD_Y + HUD_LINE_H * 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 2)
            except Exception:
                pass
        else:
            steering = 0
            centroid_y_for_speed = None

        calc_time_ms = (time.time() - start_calc) * 1000
        steering_cmd = -steering if steering_invert else steering
        if 'centroid_y_for_speed' in locals() and centroid_y_for_speed is not None:
            prop = abs(target_y - int(centroid_y_for_speed)) + 1
            dynamic_speed = float(np.clip(SPEED_MAX - (SPEED_KP * float(prop)), SPEED_MIN, SPEED_MAX))
        else:
            dynamic_speed = float(SPEED_MIN)

        mtr_cmd = np.array([dynamic_speed, steering_cmd])
        LEDs = np.array([0, 0, 0, 0, 0, 0, 1, 1])
        try:
            myCar.read_write_std(mtr_cmd, LEDs)
        except Exception:
            pass

        # HUD overlays
        cv2.putText(display_img, f'Left Phase Time: {time.time()-left_phase_start:.1f}/{LEFT_PHASE_TIME}s', (HUD_X, HUD_Y + HUD_LINE_H * 0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.namedWindow('Left Camera View', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Left Camera View', 1280, 960)
        cv2.imshow('Left Camera View', display_img)
        try:
            cv2.namedWindow('Left Thresh', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Left Thresh', 640, 480)
            cv2.imshow('Left Thresh', thresh)
        except Exception:
            pass

        key = cv2.waitKey(1)
        if key == 27:
            kill = True
            break
        if key == ord('i'):
            steering_invert = not steering_invert
            print(f"Steering invert toggled: {steering_invert}")

        time.sleep(0.05)

    if kill:
        # user requested exit during left phase
        pass

    # --- Phase 2: right-camera main loop (existing behavior) ---
    while True:  # Main control loop
        start_calc = time.time()  # Start timing calculation
        rightCam.read()  # Capture image from right camera
        img = rightCam.image_data  # Get image data from camera object
        # Check for valid image data
        if img is None or img.shape[0] == 0 or img.shape[1] == 0:
            print("Warning: Camera returned invalid image data.")  # Warn if image is invalid
            continue  # Skip to next loop iteration

        overlay_info = get_right_line_offset(img)  # Get overlay info from cropped lower half

        h, w, _ = img.shape  # Get image dimensions
        display_img = img.copy()  # Show full camera view

        # Update frame counter and FPS
        frame_count += 1
        current_time = time.time()
        if current_time - last_time >= 1.0:
            fps = frame_count
            frame_count = 0
            last_time = current_time

        # Draw processing-area outline (lower half, left 20% cropped)
        crop_x = int(w * 0.2)
        crop_y = h // 2
        crop_w = w - crop_x
        crop_h = h - crop_y
        # draw a thin yellow rectangle showing the processing region
        cv2.rectangle(display_img, (crop_x, crop_y), (crop_x + crop_w - 1, crop_y + crop_h - 1), (0, 255, 255), 2)
        # visual debug: draw left (red) and right (green) vertical guide lines and a label
        cv2.line(display_img, (crop_x, crop_y), (crop_x, crop_y + crop_h - 1), (0,0,255), 2)
        cv2.line(display_img, (crop_x + crop_w - 1, crop_y), (crop_x + crop_w - 1, crop_y + crop_h - 1), (0,255,0), 2)
        #cv2.putText(display_img, f'Crop x:{crop_x} w:{crop_w}', (crop_x + 8, crop_y + 22),
                    #cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        # Draw overlays and frame info
        if overlay_info is not None:
            # Compute on-screen target X (full-image coords)
            h, w, _ = display_img.shape
            target_x = int(w * 0.5) + target_offset

            # Get centroid of the largest contour (no more "nias" acceptance logic)
            centroid_x, centroid_y = overlay_info['centroid']
            # Move the red target down by 30 pixels (was -15, now +15)
            target_y = h // 2 + (h // 4) + 15

            # Simple Y-based P-control: steer proportionally to vertical offset
            dy = int(centroid_y) - int(target_y)
            if abs(dy) > 10:
                steering = float(np.clip(dy * steering_gain, -0.5, 0.5))
                control_mode = 'Y'
            else:
                steering = 0.0
                control_mode = 'aligned'

            # Use this centroid to map to dynamic speed
            centroid_y_for_speed = int(centroid_y)

            # Draw contour and centroid (always draw the largest detection)
            try:
                cv2.drawContours(display_img, [overlay_info['contour']], -1, (255,0,0), 2)
                cv2.circle(display_img, overlay_info['centroid'], 10, (255,0,0), -1)  # Blue centroid dot
            except Exception:
                pass

            # Draw target position as red dot (center X + offset)
            cv2.circle(display_img, (target_x, target_y), 10, (0,0,255), -1)
            # Draw vertical-error info on HUD
            try:
                cv2.putText(display_img, f'dy: {(int(centroid_y)-int(target_y)):+d} ctrl:{control_mode}',
                            (HUD_X, HUD_Y + HUD_LINE_H * 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 2)
            except Exception:
                pass
        else:
            steering = 0  # No line detected, go straight
            centroid_y_for_speed = None

        # Calculate computation time
        calc_time_ms = (time.time() - start_calc) * 1000

        # Prepare motor command (apply runtime invert if enabled)
        steering_cmd = -steering if steering_invert else steering

        # Dynamic forward speed based on vertical alignment (pixel Y distance)
        # Use centroid_y_for_speed (set from accepted detection or last accepted) if available
        if 'centroid_y_for_speed' in locals() and centroid_y_for_speed is not None:
            prop = abs(target_y - int(centroid_y_for_speed)) + 1
            dynamic_speed = float(np.clip(SPEED_MAX - (SPEED_KP * float(prop)), SPEED_MIN, SPEED_MAX))
        else:
            # No reliable centroid available: be conservative and go at minimum speed
            dynamic_speed = float(SPEED_MIN)

        mtr_cmd = np.array([dynamic_speed, steering_cmd])  # Create motor command array: [speed, steering]
        LEDs = np.array([0, 0, 0, 0, 0, 0, 1, 1])  # Set LED pattern (example)

        # Send motor command (don't attempt complex encoder probing here)
        try:
            myCar.read_write_std(mtr_cmd, LEDs)
        except Exception:
            pass

        # Use simplified encoder helper for speed (m/s) and incremental distance (m)
        try:
            speed_m_s = speed_calc.encoder_speed()
            dist_delta = speed_calc.encoder_dist()
            dist_hardcode = speed_calc.encoder_cur()
            total_distance_m = (dist_hardcode/31844) * (0.066*3.14)
        except Exception:
            speed_m_s = 0.0
            dist_delta = 0.0

        # Put frame count, FPS, and computation time on image (neatly stacked)
        cv2.putText(display_img, f'Frames: {frame_count}  FPS: {fps}', (HUD_X, HUD_Y + HUD_LINE_H * 0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(display_img, f'Calc Time: {calc_time_ms:.1f} ms', (HUD_X, HUD_Y + HUD_LINE_H * 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        cv2.putText(display_img, f'Steering: {steering:.3f}  Gain: {steering_gain}', (HUD_X, HUD_Y + HUD_LINE_H * 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        angle = steering * max_steering_angle
        cv2.putText(display_img, f'Angle: {angle:.1f} deg', (HUD_X, HUD_Y + HUD_LINE_H * 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,128,255), 2)

        # Show applied forward speed on HUD (from dynamic mapping) and encoder speed
        try:
            cv2.putText(display_img, f'Speed cmd: {dynamic_speed:.3f} m/s  enc_m/s: {speed_m_s:.3f}', (HUD_X, HUD_Y + HUD_LINE_H * 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)
        except Exception:
            pass

        # Cumulative distance (meters) from simplified helper
        cv2.putText(display_img, f'Distance: {total_distance_m:.3f} m', (HUD_X, HUD_Y + HUD_LINE_H * 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        # Resize window for larger display
        cv2.namedWindow('Right Camera View', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Right Camera View', 1280, 960)
        cv2.imshow('Right Camera View', display_img)  # Show full camera view with overlays
        key = cv2.waitKey(1)  # Wait for key press (1 ms)
        # Kill switch: ESC key (27) to exit
        if key == 27:
            print("Kill switch activated: ESC pressed.")  # Print message if ESC is pressed
            break  # Exit control loop
        # Toggle steering invert with 'i'
        if key == ord('i'):
            steering_invert = not steering_invert
            print(f"Steering invert toggled: {steering_invert}")

        # Send motor command once per loop (safe write)
        LEDs = np.array([0, 0, 0, 0, 0, 0, 1, 1])  # LED pattern
        try:
            myCar.read_write_std(mtr_cmd, LEDs)
        except Exception:
            pass

        time.sleep(0.05)  # Small delay for control loop timing
finally:
    cv2.destroyAllWindows()  # Close all OpenCV windows
    myCar.terminate()  # Terminate QCar connection
    try:
        rightCam.terminate()
    except Exception:
        pass
    try:
        leftCam.terminate()
    except Exception:
        pass
