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
# Adaptive speed tuning: when steering is large (close to ±1) reduce forward speed
# to avoid aggressive turns causing the car to run off course. The commanded speed
# will be scaled in proportion to (1 - SPEED_REDUCTION_FACTOR * |steering|)
# and clamped to [MIN_SPEED, speed].
MIN_SPEED = 0.02
# Fraction of speed to remove at maximum steering magnitude (0..1)
SPEED_REDUCTION_FACTOR = 0.75

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

# Tracking bias: prefer contours close to the previously-detected contour.
# This helps ignore spurious detections far from the last known line location.
# TRACK_DIST_SCALE (pixels) controls how quickly distance penalizes area.
prev_line_centroid = None  # (x,y) in right-crop coordinates from previous frame
TRACK_DIST_SCALE = 50.0
MIN_CONTOUR_AREA = 100  # ignore tiny contours

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
    # Crop lower half and right 30% of the image for line detection
    lower_half = image[h//2:h, :]  # Only lower half
    right_crop = lower_half[:, int(w*0.7):w]  # Crop right 30% of lower half
    gray = cv2.cvtColor(right_crop, cv2.COLOR_BGR2GRAY)  # Convert cropped image to grayscale
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)  # Threshold to highlight bright lines
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours
    overlay_info = None
    if not contours:
        return None

    # Score contours using area and distance to previous centroid (if known).
    best_score = -1.0
    best = None
    # Compute previous centroid in right_crop coordinates if available
    prev = None
    try:
        global prev_line_centroid
        prev = prev_line_centroid
    except Exception:
        prev = None

    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_CONTOUR_AREA:
            continue
        M = cv2.moments(c)
        if M['m00'] == 0:
            continue
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        # distance to previous centroid (in pixels)
        if prev is None:
            distance = 0.0
        else:
            dx = float(cx - prev[0])
            dy = float(cy - prev[1])
            distance = (dx*dx + dy*dy) ** 0.5
        # score: area reduced by distance; dividing by (1 + distance/scale) keeps units sane
        score = float(area) / (1.0 + (distance / TRACK_DIST_SCALE))
        if score > best_score:
            best_score = score
            best = (c, cx, cy, area, distance)

    if best is None:
        return None

    largest, cx, cy, area, distance = best
    # Prepare overlay info in full-image coordinates
    contour_full = largest + np.array([int(w*0.7), h//2])
    centroid_full = (int(w*0.7) + cx, h//2 + cy)
    overlay_info = {
        'contour': contour_full,
        'centroid': centroid_full,
        'offset': cx,
        'centroid_local': (cx, cy),
        'area': area,
        'distance_to_prev': distance,
        'score': best_score,
    }
    return overlay_info  # Return overlay info

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

        # Draw overlays and frame info
        if overlay_info is not None:
            error = target_offset - overlay_info['offset']  # Calculate error from desired offset
            steering = np.clip(error * steering_gain, -1, 1)  # Reduced gain for smoother turns
            # Draw overlays on full image
            cv2.drawContours(display_img, [overlay_info['contour']], -1, (255,0,0), 2)
            cv2.circle(display_img, overlay_info['centroid'], 10, (255,0,0), -1)  # Blue centroid dot
            # Draw target position as red dot
            h, w, _ = display_img.shape
            target_x = int(w * 0.7) + target_offset
            target_y = h // 2 + (h // 4)  # Middle of cropped lower half
            cv2.circle(display_img, (target_x, target_y), 10, (0,0,255), -1)
            # Update previous centroid (local coordinates within right-crop)
            try:
                prev_line_centroid = overlay_info.get('centroid_local', prev_line_centroid)
            except Exception:
                prev_line_centroid = prev_line_centroid
            # Debug: show contour score and distance-to-prev
            try:
                score = overlay_info.get('score', 0.0)
                distp = overlay_info.get('distance_to_prev', 0.0)
                cv2.putText(display_img, f'Score: {score:.1f}  DistPrev: {distp:.1f}', (10, 320),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,180,0), 2)
            except Exception:
                pass
        else:
            steering = 0  # No line detected, go straight

        # Calculate computation time
        calc_time_ms = (time.time() - start_calc) * 1000

        # Prepare motor command: scale speed down when steering magnitude is high
        adaptive_speed = float(speed) * (1.0 - SPEED_REDUCTION_FACTOR * abs(steering))
        # Clamp so we never command trivially small or negative speeds
        if adaptive_speed < MIN_SPEED:
            adaptive_speed = MIN_SPEED
        if adaptive_speed > speed:
            adaptive_speed = float(speed)
        mtr_cmd = np.array([adaptive_speed, steering])  # Create motor command array: [speed, steering]
        LEDs = np.array([0, 0, 0, 0, 0, 0, 1, 1])  # Set LED pattern (example)

        # Send motor command and attempt to capture a single motor encoder scalar
        motor_enc = None
        hw_counts_per_s = None
        enc_source = 'none'
        now_t = time.time()
        try:
            ret = myCar.read_write_std(mtr_cmd, LEDs)
            # Typical return: (current, batteryVoltage, encoderCounts)
            if isinstance(ret, (list, tuple)) and len(ret) >= 3:
                enc_ret = ret[2]
                if isinstance(enc_ret, (list, tuple, np.ndarray)):
                    if len(enc_ret) >= 1:
                        try:
                            motor_enc = int(enc_ret[0])
                            enc_source = 'read_write_std[2][0]'
                        except Exception:
                            motor_enc = None
                else:
                    try:
                        motor_enc = int(enc_ret)
                        enc_source = 'read_write_std[2] (scalar)'
                    except Exception:
                        motor_enc = None
            # If no counts found but fallback enabled, consider ret[0] as velocity (rev/s)
            if motor_enc is None and FALLBACK_RET0_AS_RPS and isinstance(ret, (list, tuple)) and len(ret) >= 1:
                try:
                    rps = float(ret[0])
                    hw_counts_per_s = rps * ENC_COUNTS_PER_REV
                    enc_source = 'read_write_std[0] (assumed rps)'
                except Exception:
                    pass
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
                    delta_counts = motor_enc - prev_enc_count
                    enc_rate = (delta_counts) / dt
                    # accumulate raw counts (can be negative when reversing)
                    cumulative_counts += delta_counts
                prev_enc_count = motor_enc
                prev_enc_time = now_t
        else:
            # fallback to read_encoders() first channel if available
            e = read_encoders(myCar)
            if e is not None:
                try:
                    if isinstance(e, (list, tuple, np.ndarray)) and len(e) >= 1:
                        motor_enc = int(e[0])
                        enc_source = 'read_encoders()'
                except Exception:
                    motor_enc = None

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
        cv2.putText(display_img, f'Counts/s Forward: {counts_per_s:.1f}', (10, 170),
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

    # Enc source overlay removed to declutter HUD

        # Show commanded/adaptive speed (magnitude is forward fractional command)
        cv2.putText(display_img, f'Cmd Speed: {adaptive_speed:.3f}', (10, 295),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,0), 2)

        # Cumulative distance (meters) computed from accumulated raw counts
        try:
            revs_total = cumulative_counts / float(ENC_COUNTS_PER_REV) if ENC_COUNTS_PER_REV != 0 else 0.0
            dist_m = revs_total * (2.0 * np.pi * WHEEL_RADIUS_M)
        except Exception:
            dist_m = 0.0
        cv2.putText(display_img, f'Distance: {dist_m:.3f} m', (10, 270),
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

        # Send motor command once per loop (safe write)
        LEDs = np.array([0, 0, 0, 0, 0, 0, 1, 1])  # LED pattern
        try:
            # Use adaptive speed when sending the motor command
            myCar.read_write_std(np.array([adaptive_speed, steering]), LEDs)
        except Exception:
            pass

        time.sleep(0.05)  # Small delay for control loop timing
finally:
    cv2.destroyAllWindows()  # Close all OpenCV windows
    myCar.terminate()  # Terminate QCar connection
    rightCam.terminate()  # Terminate camera connection
