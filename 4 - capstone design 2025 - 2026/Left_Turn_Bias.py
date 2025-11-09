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
# Create left camera object (user requested camera_id=2)
leftCam = Camera2D(camera_id="2", frame_width=640, frame_height=480, frame_rate=30.0)

# Desired pixel offset from right edge for line following
target_offset = 50
# Forward speed of the robot (lower value for slower movement)
speed = 0.072
steering_gain = 0.009  # Gain used for steering calculation (increased per user request)
max_steering_angle = 28  # Maximum steering angle in degrees (mechanical limit)
# Runtime steering invert toggle: when True steering is multiplied by -1 before sending
# Invert steering by default for this left-camera configuration. Press 'i' at runtime to toggle.
steering_invert = True
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
    # Crop the middle 50% vertically (remove top 25% and bottom 25%)
    crop_x = int(w * 0.2)  # remove left 20%
    # keep the middle 50% vertically (remove top 25% and bottom 25%)
    top_crop = h // 4
    bottom_crop = (3 * h) // 4
    lower_half = image[top_crop:bottom_crop, crop_x:]  # middle half with left 20% removed
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
            contour_full = largest + np.array([crop_x, top_crop])
            centroid_full = (crop_x + cx, top_crop + cy)
            overlay_info = {
                'contour': contour_full,
                'centroid': centroid_full,
                # store offset as full-image X so main loop can compare directly to target_x
                'offset': crop_x + cx
            }
    # Always return (overlay_info, thresh) so caller can visualize the thresholded image
    return overlay_info, thresh

try:
    while True:  # Main control loop
        start_calc = time.time()  # Start timing calculation

        # Capture image from left camera
        leftCam.read()
        img = leftCam.image_data

        # Check for valid image data
        if img is None or img.shape[0] == 0 or img.shape[1] == 0:
            print("Warning: Camera returned invalid image data.")
            time.sleep(0.05)
            continue

        overlay_info, thresh = get_right_line_offset(img)  # Get overlay and threshold image from crop

        h, w, _ = img.shape  # Get image dimensions
        display_img = img.copy()  # Show full camera view

        # Update frame counter and FPS
        frame_count += 1
        current_time = time.time()
        if current_time - last_time >= 1.0:
            fps = frame_count
            frame_count = 0
            last_time = current_time

        # Draw processing-area outline (middle 50% vertically, left 20% cropped)
        crop_x = int(w * 0.2)
        crop_y = h // 4  # top of the middle 50% crop
        crop_w = w - crop_x
        crop_h = h // 2
        cv2.rectangle(display_img, (crop_x, crop_y), (crop_x + crop_w - 1, crop_y + crop_h - 1), (0, 255, 255), 2)
        cv2.line(display_img, (crop_x, crop_y), (crop_x, crop_y + crop_h - 1), (0,0,255), 2)
        cv2.line(display_img, (crop_x + crop_w - 1, crop_y), (crop_x + crop_w - 1, crop_y + crop_h - 1), (0,255,0), 2)

        # Draw overlays and frame info
        if overlay_info is not None:
            # Compute on-screen target X (full-image coords)
            target_x = int(w * 0.5) + target_offset

            # Get centroid of the largest contour (full-image coords)
            centroid_x, centroid_y = overlay_info['centroid']

            # Target vertically centered in the kept crop band
            target_y = crop_y + (crop_h // 2)

            # Robustness: accept a new detection only if it's near the last accepted one
            # or within image bounds. If not, ignore it and keep following the last accepted
            # (presumed-blue) contour's Y coordinate.
            ACCEPT_DIST = 100  # pixels tolerance in both x and y
            SEARCH_BOX = 100   # size of purple search box (pixels)
            # bounding box where contours are considered valid (user-provided)
            VALID_TL = (128, 240)
            VALID_BR = (639, 479)

            # Check whether this new centroid is inside the valid region
            in_bounds = (VALID_TL[0] <= centroid_x <= VALID_BR[0]) and (VALID_TL[1] <= centroid_y <= VALID_BR[1])

            accept = False
            if last_accepted_centroid is None:
                # First valid detection becomes the baseline
                accept = True if in_bounds else False
            else:
                last_x, last_y = last_accepted_centroid
                dx = abs(int(centroid_x) - int(last_x))
                dy = abs(int(centroid_y) - int(last_y))
                # Accept only if both dx and dy are within tolerance and within bounds
                accept = in_bounds and (dx <= ACCEPT_DIST and dy <= ACCEPT_DIST)

            if accept:
                # Update baseline to this new (presumed-blue) contour
                last_accepted_centroid = (int(centroid_x), int(centroid_y))
                used_centroid_x, used_centroid_y = last_accepted_centroid
                centroid_y_for_speed = int(used_centroid_y)
                # Draw accepted contour in blue
                try:
                    cv2.drawContours(display_img, [overlay_info['contour']], -1, (255,0,0), 2)
                    cv2.circle(display_img, (used_centroid_x, used_centroid_y), 10, (255,0,0), -1)
                except Exception:
                    pass
                control_note = 'ACCEPT'
            else:
                # Ignored detection (likely yellow or spurious). Draw it in yellow but
                # continue to follow the previously accepted contour's Y if available.
                try:
                    cv2.drawContours(display_img, [overlay_info['contour']], -1, (0,255,255), 2)
                except Exception:
                    pass
                control_note = 'IGNORED(yellow)'
                if last_accepted_centroid is not None:
                    used_centroid_x, used_centroid_y = last_accepted_centroid
                    centroid_y_for_speed = int(used_centroid_y)
                    # Draw a purple search box (100x100) around the last accepted centroid
                    half = SEARCH_BOX // 2
                    tl = (max(0, used_centroid_x - half), max(0, used_centroid_y - half))
                    br = (min(w-1, used_centroid_x + half), min(h-1, used_centroid_y + half))
                    try:
                        cv2.rectangle(display_img, tl, br, (128, 0, 128), 2)
                    except Exception:
                        pass
                else:
                    # No baseline available: treat as no detection
                    centroid_y_for_speed = None

            # Compute steering using the chosen centroid Y (either new accepted or last accepted)
            if centroid_y_for_speed is not None:
                dy_ctrl = int(centroid_y_for_speed) - int(target_y)
                if abs(dy_ctrl) > 10:
                    steering = float(np.clip(dy_ctrl * steering_gain, -0.5, 0.5))
                    control_mode = 'Y'
                else:
                    steering = 0.0
                    control_mode = 'aligned'
            else:
                # No reliable centroid to follow
                steering = 0.0
                control_mode = 'no_target'

            # Draw target position as red dot and status
            try:
                cv2.circle(display_img, (target_x, target_y), 10, (0,0,255), -1)
                note = control_note if 'control_note' in locals() else ''
                dy_text = f'{(int(centroid_y_for_speed)-int(target_y)):+d}' if centroid_y_for_speed is not None else ' N/A'
                cv2.putText(display_img, f'dy: {dy_text} ctrl:{control_mode} note:{note}',
                            (HUD_X, HUD_Y + HUD_LINE_H * 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 2)
            except Exception:
                pass
        else:
            # No contour detected: if we have a last accepted centroid, continue following its Y
            if last_accepted_centroid is not None:
                centroid_y_for_speed = int(last_accepted_centroid[1])
                dy_ctrl = int(centroid_y_for_speed) - int(crop_y + (crop_h // 2))
                if abs(dy_ctrl) > 10:
                    steering = float(np.clip(dy_ctrl * steering_gain, -0.5, 0.5))
                    control_mode = 'Y(last)'
                else:
                    steering = 0.0
                    control_mode = 'aligned'
            else:
                steering = 0  # No line detected at all, go straight
                centroid_y_for_speed = None

        # Calculate computation time
        calc_time_ms = (time.time() - start_calc) * 1000

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

        # Use simplified encoder helper for speed and incremental distance
        try:
            speed_m_s = speed_calc.encoder_speed()
            dist_delta = speed_calc.encoder_dist()
            dist_hardcode = speed_calc.encoder_cur()
            total_distance_m = (dist_hardcode/31844) * (0.066*3.14)
        except Exception:
            speed_m_s = 0.0
            dist_delta = 0.0

        # HUD overlays
        cv2.putText(display_img, f'Frames: {frame_count}  FPS: {fps}', (HUD_X, HUD_Y + HUD_LINE_H * 0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(display_img, f'Calc Time: {calc_time_ms:.1f} ms', (HUD_X, HUD_Y + HUD_LINE_H * 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        cv2.putText(display_img, f'Steering: {steering:.3f}  Gain: {steering_gain}', (HUD_X, HUD_Y + HUD_LINE_H * 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        angle = steering * max_steering_angle
        cv2.putText(display_img, f'Angle: {angle:.1f} deg', (HUD_X, HUD_Y + HUD_LINE_H * 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,128,255), 2)
        try:
            cv2.putText(display_img, f'Speed cmd: {dynamic_speed:.3f} m/s  enc_m/s: {speed_m_s:.3f}', (HUD_X, HUD_Y + HUD_LINE_H * 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)
        except Exception:
            pass
        cv2.putText(display_img, f'Distance: {total_distance_m:.3f} m', (HUD_X, HUD_Y + HUD_LINE_H * 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        # Resize and show window
        cv2.namedWindow('Left Camera View', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Left Camera View', 1280, 960)
        cv2.imshow('Left Camera View', display_img)

        # Show thresholded view (binary) for debugging; resize to a visible size
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
        # Toggle steering invert with 'i'
        if key == ord('i'):
            steering_invert = not steering_invert
            print(f"Steering invert toggled: {steering_invert}")

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
