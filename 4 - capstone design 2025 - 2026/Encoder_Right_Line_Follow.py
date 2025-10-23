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

# Desired pixel offset from right edge for line following
target_offset = 50
# Forward speed of the robot (lower value for slower movement)
speed = 0.075
steering_gain = 0.005  # Gain used for steering calculation
max_steering_angle = 28  # Maximum steering angle in degrees (mechanical limit)
# Runtime steering invert toggle: when True steering is multiplied by -1 before sending
steering_invert = False

# Frame counter and FPS calculation
frame_count = 0
fps = 0
last_time = time.time()

# Encoder tracking (single scalar motor encoder)
prev_enc_count = None
prev_enc_time = time.time()
enc_rate = 0.0
# Cumulative encoder counts observed since script start (raw units)
cumulative_counts = 0

# Encoder / wheel constants (QCar E8T-720-125): 720 counts per rev (single-ended)
# Quadrature mode = 4x -> 2880 counts/rev. Adjust if your setup differs.
ENC_COUNTS_PER_REV = 30584  # measured by motor-driven calibration (raw units from read_write_std)
# Wheel radius in meters (derived from tire diameter). User provided diameter 0.066 m -> radius 0.033 m
WHEEL_RADIUS_M = 0.066 / 2.0

# ...existing code...

def read_encoder_velocity(car):
    """Try to read hardware-provided encoder velocities (counts/s) from the car.
    Returns (left_counts_per_s, right_counts_per_s) or None.
    """
    # Try QCar-specific read methods/attributes first
    try:
        # 1) read_encoder() method (common on some QCar APIs)
        fn = getattr(car, 'read_encoder', None)
        if callable(fn):
            try:
                v = fn()
                if isinstance(v, (list, tuple, np.ndarray)) and len(v) >= 2:
                    return float(v[0]), float(v[1])
                if isinstance(v, dict):
                    for k in ('encoders', 'encoder_counts', 'encoderCounts', 'velocity'):
                        if k in v and isinstance(v[k], (list, tuple, np.ndarray)) and len(v[k]) >= 2:
                            return float(v[k][0]), float(v[k][1])
            except Exception:
                pass

        # 2) mtr_encoder attribute (some APIs expose this)
        if hasattr(car, 'mtr_encoder'):
            v = getattr(car, 'mtr_encoder')
            if isinstance(v, (list, tuple, np.ndarray)) and len(v) >= 2:
                return float(v[0]), float(v[1])
            if isinstance(v, dict):
                for k in ('encoders', 'encoder_counts', 'encoderCounts', 'velocity'):
                    if k in v and isinstance(v[k], (list, tuple, np.ndarray)) and len(v[k]) >= 2:
                        return float(v[k][0]), float(v[k][1])

        # 3) read_std() may return a dict with velocities
        fn = getattr(car, 'read_std', None)
        if callable(fn):
            try:
                data = fn()
                if isinstance(data, dict):
                    for k in ('velocity', 'velocities', 'encoder_velocity', 'encoder_velocities', 'other'):
                        v = data.get(k) if k in data else None
                        if v is None:
                            continue
                        if isinstance(v, (list, tuple, np.ndarray)) and len(v) >= 2:
                            return float(v[0]), float(v[1])
                        if isinstance(v, dict):
                            for subk in ('encoders','encoder_counts','encoderCounts','velocity'):
                                if subk in v and isinstance(v[subk], (list, tuple, np.ndarray)) and len(v[subk]) >= 2:
                                    return float(v[subk][0]), float(v[subk][1])
            except Exception:
                pass
    except Exception:
        pass
    # No hardware velocity found
    return None

def read_encoders(car):
    """Try several common QCar encoder access patterns and return (left, right) counts or None.
    This function is defensive: it won't raise if the API doesn't expose encoders the same way.
    """
    try:
        # 1) prefer read_encoder() if available
        fn = getattr(car, 'read_encoder', None)
        if callable(fn):
            try:
                val = fn()
                # support single-channel encoders (motor encoder only) as well as dual-channel
                if isinstance(val, (list, tuple, np.ndarray)):
                    if len(val) >= 2:
                        return int(val[0]), int(val[1])
                    if len(val) == 1:
                        return int(val[0]), None
                if isinstance(val, dict):
                    for k in ('encoders', 'encoder_counts', 'encoderCounts', 'mtr_encoder'):
                        if k in val and isinstance(val[k], (list, tuple, np.ndarray)) and len(val[k]) >= 2:
                            return int(val[k][0]), int(val[k][1])
            except Exception:
                pass

        # 2) attribute mtr_encoder (common on this QCar API)
        if hasattr(car, 'mtr_encoder'):
            val = getattr(car, 'mtr_encoder')
            if isinstance(val, (list, tuple, np.ndarray)):
                if len(val) >= 2:
                    return int(val[0]), int(val[1])
                if len(val) == 1:
                    return int(val[0]), None
            if isinstance(val, dict):
                for k in ('encoders', 'encoder_counts', 'encoderCounts'):
                    if k in val and isinstance(val[k], (list, tuple, np.ndarray)) and len(val[k]) >= 2:
                        return int(val[k][0]), int(val[k][1])

        # 3) try read_std() which often returns a data dict
        fn = getattr(car, 'read_std', None)
        if callable(fn):
            try:
                data = fn()
                if isinstance(data, dict):
                    for k in ('encoders', 'encoder_counts', 'encoderCounts', 'mtr_encoder'):
                        if k in data and isinstance(data[k], (list, tuple, np.ndarray)):
                            if len(data[k]) >= 2:
                                return int(data[k][0]), int(data[k][1])
                            if len(data[k]) == 1:
                                return int(data[k][0]), None
            except Exception:
                pass

        # 4) fallback: try several common method/attr names
        for name in ('get_encoders', 'read_encoders', 'getEncoderCounts', 'readEncoderCounts'):
            fn = getattr(car, name, None)
            if callable(fn):
                try:
                    val = fn()
                except Exception:
                    val = None
                if val is None:
                    continue
                if isinstance(val, (list, tuple, np.ndarray)) and len(val) >= 2:
                    return int(val[0]), int(val[1])

        for name in ('encoder_counts', 'encoders', 'encoderCounts'):
            val = getattr(car, name, None)
            if val is None:
                continue
            if isinstance(val, (list, tuple, np.ndarray)) and len(val) >= 2:
                return int(val[0]), int(val[1])

        # 5) as a last resort, some APIs return a dict from read_std() or similar
        if hasattr(car, 'read_std') and callable(getattr(car, 'read_std')):
            try:
                data = car.read_std()
                if isinstance(data, dict):
                    for k in ('encoders', 'encoder_counts', 'encoderCounts'):
                        if k in data and isinstance(data[k], (list, tuple, np.ndarray)) and len(data[k]) >= 2:
                            return int(data[k][0]), int(data[k][1])
            except Exception:
                pass
    except Exception:
        # be conservative: never let encoder probing crash the main loop
        return None
    return None

# Function to find the x-position of the detected line in the right crop
def get_right_line_offset(image):
    h, w, _ = image.shape  # Get image dimensions
    # Crop lower half full-width (uncrop left side) for line detection
    lower_half = image[h//2:h, :]  # Only lower half (full width)
    right_crop = lower_half[:, :]  # Full-width lower half
    gray = cv2.cvtColor(right_crop, cv2.COLOR_BGR2GRAY)  # Convert cropped image to grayscale
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)  # Threshold to highlight bright lines
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours
    overlay_info = None
    if contours:  # If any contours are found
        largest = max(contours, key=cv2.contourArea)  # Select the largest contour (assumed to be the line)
        M = cv2.moments(largest)  # Calculate moments for the largest contour
        if M['m00'] > 0:  # Prevent division by zero
            cx = int(M['m10'] / M['m00'])  # Compute center x-position of the contour (in right_crop)
            cy = int(M['m01'] / M['m00'])  # Compute center y-position of the contour (in right_crop)
            # Prepare overlay info for full image
            contour_full = largest + np.array([0, h//2])
            centroid_full = (cx, h//2+cy)
            overlay_info = {
                'contour': contour_full,
                'centroid': centroid_full,
                'offset': cx
            }
            return overlay_info  # Return overlay info
    return None  # Return None if no line is found

try:
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

        # Draw processing-area outline (lower half, full width)
        crop_x = 0
        crop_y = h // 2
        crop_w = w
        crop_h = h - crop_y
        # draw a thin yellow rectangle showing the processing region
        cv2.rectangle(display_img, (crop_x, crop_y), (crop_x + crop_w - 1, crop_y + crop_h - 1), (0, 255, 255), 2)
        # visual debug: draw left (red) and right (green) vertical guide lines and a label
        cv2.line(display_img, (crop_x, crop_y), (crop_x, crop_y + crop_h - 1), (0,0,255), 2)
        cv2.line(display_img, (crop_x + crop_w - 1, crop_y), (crop_x + crop_w - 1, crop_y + crop_h - 1), (0,255,0), 2)
        cv2.putText(display_img, f'Crop x:{crop_x} w:{crop_w}', (crop_x + 8, crop_y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        # Draw overlays and frame info
        if overlay_info is not None:
            # Compute on-screen target X (in full-image coords) and convert to crop coords
            h, w, _ = display_img.shape
            target_x = int(w * 0.5) + target_offset
            # crop_x is 0 for our lower-half full-width crop, so target_x in crop coords == target_x
            target_x_in_crop = target_x

            # Calculate error as (target - centroid_in_crop). Positive => centroid left of target.
            error = target_x_in_crop - overlay_info['offset']
            steering = float(np.clip(error * steering_gain, -0.5, 0.5))  # Hardware-safe clamp

            # Draw overlays on full image
            cv2.drawContours(display_img, [overlay_info['contour']], -1, (255,0,0), 2)
            cv2.circle(display_img, overlay_info['centroid'], 10, (255,0,0), -1)  # Blue centroid dot
            # Draw target position as red dot (center X + offset)
            target_y = h // 2 + (h // 4)  # Middle of cropped lower half
            cv2.circle(display_img, (target_x, target_y), 10, (0,0,255), -1)
        else:
            steering = 0  # No line detected, go straight

        # Calculate computation time
        calc_time_ms = (time.time() - start_calc) * 1000

        # Prepare motor command (apply runtime invert if enabled)
        steering_cmd = -steering if steering_invert else steering
        mtr_cmd = np.array([speed, steering_cmd])  # Create motor command array: [speed, steering]
        LEDs = np.array([0, 0, 0, 0, 0, 0, 1, 1])  # Set LED pattern (example)

        # Send motor command and attempt to capture a single motor encoder scalar
        motor_enc = None
        hw_counts_per_s = None
        enc_source = 'none'
        now_t = time.time()
        try:
            # We expect the encoder counts as a scalar in ret[2]. Read that directly.
            ret = myCar.read_write_std(mtr_cmd, LEDs)
            # Typical return: (current, batteryVoltage, encoderCounts)
            if isinstance(ret, (list, tuple)) and len(ret) >= 3:
                enc_ret = ret[2]
                try:
                    motor_enc = int(enc_ret)
                    enc_source = 'read_write_std[2] (scalar)'
                except Exception:
                    motor_enc = None
            else:
                motor_enc = None
        except Exception:
            motor_enc = None

        # If hardware-provided velocity is available via other methods, prefer first channel
        vel_try = read_encoder_velocity(myCar)
        if vel_try is not None:
            try:
                hw_counts_per_s = float(vel_try[0]) if isinstance(vel_try, (list, tuple, np.ndarray)) else float(vel_try)
                enc_source = 'read_encoder() or read_std() velocities'
            except Exception:
                pass

        # Compute encoder rate from counts if we have motor_enc
        if motor_enc is not None:
            if prev_enc_count is None:
                prev_enc_count = motor_enc
                prev_enc_time = now_t
                enc_rate = 0.0
            else:
                dt = now_t - prev_enc_time
                if dt <= 0:
                    enc_rate = 0.0
                else:
                    enc_rate = (motor_enc - prev_enc_count) / dt
                    delta_counts = motor_enc - prev_enc_count
                    enc_rate = (delta_counts) / dt
                    # accumulate raw counts (can be negative when reversing)
                    cumulative_counts += delta_counts
                prev_enc_count = motor_enc
                prev_enc_time = now_t
        else:
            # No encoder counts available from read_write_std; leave motor_enc=None.
            pass

        # Put frame count, FPS, and computation time on image
        cv2.putText(display_img, f'Frames: {frame_count}  FPS: {fps}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.putText(display_img, f'Calc Time: {calc_time_ms:.1f} ms', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        cv2.putText(display_img, f'Steering: {steering:.3f}  Gain: {steering_gain}', (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
        angle = steering * max_steering_angle
        cv2.putText(display_img, f'Angle: {angle:.1f} deg', (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,128,255), 2)

        # Encoder display (single forward/combined encoder)
        # The hardware exposes a single scalar encoder-like value that appears to
        # represent net forward rotation (combined/aggregate). We label it as
        # 'ForwardEnc (combined)' to reduce confusion.
        forward_cnt = motor_enc if motor_enc is not None else 0
        cv2.putText(display_img, f'ForwardEnc (combined): {forward_cnt}', (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)

        # Determine counts/s: prefer hardware velocity if available, otherwise derived enc_rate
        counts_per_s = float(hw_counts_per_s) if hw_counts_per_s is not None else float(enc_rate if enc_rate is not None else 0.0)
        cv2.putText(display_img, f'EncCounts/s: {counts_per_s:.1f}', (10, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)

        # Conversions: counts/s -> RPM, rad/s, m/s (forward/combined)
        rpm_forward = (counts_per_s / ENC_COUNTS_PER_REV) * 60.0
        rad_s_forward = (counts_per_s / ENC_COUNTS_PER_REV) * 2.0 * np.pi
        vel_m_s_forward = rad_s_forward * WHEEL_RADIUS_M

        cv2.putText(display_img, f'RPM: {rpm_forward:.1f}', (10, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)
        cv2.putText(display_img, f'Rad/s: {rad_s_forward:.2f}', (10, 210),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)
        cv2.putText(display_img, f'm/s: {vel_m_s_forward:.3f}', (10, 230),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)

        # Show which source we used for encoder/velocity
        #cv2.putText(display_img, f'Enc source: {enc_source}', (10, 250),
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,200,100), 2)

        # Cumulative distance (meters) computed from accumulated raw counts
        try:
            revs_total = cumulative_counts / float(ENC_COUNTS_PER_REV) if ENC_COUNTS_PER_REV != 0 else 0.0
            dist_m = revs_total * (2.0 * np.pi * WHEEL_RADIUS_M)
        except Exception:
            dist_m = 0.0
        cv2.putText(display_img, f'Distance: {dist_m:.3f} m', (10, 250),
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
    rightCam.terminate()  # Terminate camera connection
