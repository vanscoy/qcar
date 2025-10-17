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
DIAGNOSTICS_ON = True

def run_diagnostics_once(car):
    """Print and save a compact diagnostic snapshot of the car object and read() output.
    This helps map encoder keys and hardware velocity fields to wheel channels.
    """
    try:
        info_lines = []
        info_lines.append('=== myCar dir() ===')
        try:
            names = dir(car)
            info_lines.extend(names)
        except Exception as e:
            info_lines.append(f'dir() failed: {e}')

        info_lines.append('\n=== myCar.read() snapshot ===')
        try:
            data = None
            if hasattr(car, 'read') and callable(getattr(car, 'read')):
                data = car.read()
            info_lines.append(repr(data))
            # If dict-like, show keys to help mapping
            if isinstance(data, dict):
                info_lines.append('\nmyCar.read() keys:')
                info_lines.extend([str(k) for k in data.keys()])
        except Exception as e:
            info_lines.append(f'read() failed: {e}')

        out = '\n'.join(info_lines)
        print(out)
        try:
            with open('encoder_diag.txt', 'w') as f:
                f.write(out)
            print('Diagnostic written to encoder_diag.txt')
        except Exception as e:
            print('Failed to write diagnostic file:', e)
    except Exception:
        pass

def run_extended_encoder_diagnostics(car):
    """Call several encoder-related methods/attributes and append their outputs to encoder_diag.txt.
    This helps identify where encoder counts live on this QCar instance.
    """
    tries = []
    def safe_call(name):
        try:
            if hasattr(car, name):
                attr = getattr(car, name)
                if callable(attr):
                    try:
                        val = attr()
                        return f'CALL {name} -> {repr(val)}'
                    except Exception as e:
                        return f'CALL {name} raised {e}'
                else:
                    return f'ATTR {name} -> {repr(attr)}'
            else:
                return f'NO {name}'
        except Exception as e:
            return f'ERR {name} -> {e}'

    candidates = [
        # Avoid calling read_write_std without arguments; we'll call it explicitly with a small test command below
        # 'read_write_std',
        'read_encoder',
        'mtr_encoder',
        'read_std',
        'read_encoder_channels_throttle',
        'read_encoder_buffer_throttle',
        'read_other_channels_accelerometer',
    ]
    for c in candidates:
        tries.append(safe_call(c))

    out = '\n=== Extended encoder diagnostics ===\n' + '\n'.join(tries) + '\n'
    print(out)
    try:
        with open('encoder_diag.txt', 'a') as f:
            f.write('\n' + out)
        print('Appended extended diagnostics to encoder_diag.txt')
    except Exception as e:
        print('Failed to append extended diagnostics:', e)

    # Targeted call: call read_write_std WITH a small test command to capture its real return structure.
    try:
        mtr_cmd_test = np.array([0.0, 0.0])
        LEDs_test = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        try:
            ret = car.read_write_std(mtr_cmd_test, LEDs_test)
            txt = f'CALL read_write_std(mtr_cmd, LEDs) -> {repr(ret)}'
        except Exception as e:
            txt = f'CALL read_write_std(mtr_cmd, LEDs) raised {e}'
        print(txt)
        with open('encoder_diag.txt', 'a') as f:
            f.write('\n' + txt + '\n')
        print('Appended read_write_std(mtr_cmd, LEDs) result to encoder_diag.txt')
    except Exception as e:
        print('Failed targeted read_write_std diagnostic:', e)

# Create QCar object for robot control
myCar = QCar()
# Create right camera object
rightCam = Camera2D(camera_id="0", frame_width=640, frame_height=480, frame_rate=30.0)

# Run diagnostics once (prints dir(myCar) and myCar.read() snapshot) if enabled
if DIAGNOSTICS_ON:
    run_diagnostics_once(myCar)
    # Append extended diagnostics to help find encoder fields
    try:
        run_extended_encoder_diagnostics(myCar)
    except Exception:
        pass

# Desired pixel offset from right edge for line following
target_offset = 50
# Forward speed of the robot (lower value for slower movement)
speed = 0.075
steering_gain = 0.005  # Gain used for steering calculation
max_steering_angle = 28  # Maximum steering angle in degrees (mechanical limit)

# Frame counter and FPS calculation
frame_count = 0
fps = 0
last_time = time.time()

# Encoder tracking (robust to different QCar API names)
prev_enc_counts = None
prev_enc_time = time.time()
enc_rates = (0.0, 0.0)

# Encoder / wheel constants (QCar E8T-720-125): 720 counts per rev (single-ended)
# Quadrature mode = 4x -> 2880 counts/rev. Adjust if your setup differs.
ENC_COUNTS_PER_REV = 2880
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
                if isinstance(val, (list, tuple, np.ndarray)) and len(val) >= 2:
                    return int(val[0]), int(val[1])
                if isinstance(val, dict):
                    for k in ('encoders', 'encoder_counts', 'encoderCounts', 'mtr_encoder'):
                        if k in val and isinstance(val[k], (list, tuple, np.ndarray)) and len(val[k]) >= 2:
                            return int(val[k][0]), int(val[k][1])
            except Exception:
                pass

        # 2) attribute mtr_encoder (common on this QCar API)
        if hasattr(car, 'mtr_encoder'):
            val = getattr(car, 'mtr_encoder')
            if isinstance(val, (list, tuple, np.ndarray)) and len(val) >= 2:
                return int(val[0]), int(val[1])
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
                        if k in data and isinstance(data[k], (list, tuple, np.ndarray)) and len(data[k]) >= 2:
                            return int(data[k][0]), int(data[k][1])
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
    if contours:  # If any contours are found
        largest = max(contours, key=cv2.contourArea)  # Select the largest contour (assumed to be the line)
        M = cv2.moments(largest)  # Calculate moments for the largest contour
        if M['m00'] > 0:  # Prevent division by zero
            cx = int(M['m10'] / M['m00'])  # Compute center x-position of the contour (in right_crop)
            cy = int(M['m01'] / M['m00'])  # Compute center y-position of the contour (in right_crop)
            # Prepare overlay info for full image
            contour_full = largest + np.array([int(w*0.7), h//2])
            centroid_full = (int(w*0.7)+cx, h//2+cy)
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
        else:
            steering = 0  # No line detected, go straight

        # Calculate computation time
        calc_time_ms = (time.time() - start_calc) * 1000

        # Prepare motor command
        mtr_cmd = np.array([speed, steering])  # Create motor command array: [speed, steering]
        LEDs = np.array([0, 0, 0, 0, 0, 0, 1, 1])  # Set LED pattern (example)

        # Send motor command and attempt to capture encoder counts returned by read_write_std
        enc = None
        hw_vel = None
        now_t = time.time()
        try:
            ret = myCar.read_write_std(mtr_cmd, LEDs)
            # Typical return: (current, batteryVoltage, encoderCounts)
            if isinstance(ret, (list, tuple)) and len(ret) >= 3:
                enc_ret = ret[2]
                if isinstance(enc_ret, (list, tuple, np.ndarray)) and len(enc_ret) >= 2:
                    enc = (int(enc_ret[0]), int(enc_ret[1]))
        except Exception:
            # If read_write_std fails to return enc, fall back to probing later
            enc = None

        # If hardware-provided velocity is available via other methods, try to get it
        hw_vel = read_encoder_velocity(myCar)

        # Compute encoder rates from counts if we have them
        if enc is not None:
            if prev_enc_counts is None:
                prev_enc_counts = enc
                prev_enc_time = now_t
                enc_rates = (0.0, 0.0)
            else:
                dt = now_t - prev_enc_time
                if dt <= 0:
                    enc_rates = (0.0, 0.0)
                else:
                    enc_rates = ((enc[0] - prev_enc_counts[0]) / dt,
                                 (enc[1] - prev_enc_counts[1]) / dt)
                prev_enc_counts = enc
                prev_enc_time = now_t
        else:
            # last resort: probe using API if read_write_std did not give encoderCounts
            enc = read_encoders(myCar) or (None, None)

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

        # Encoder display
        left_cnt = enc[0] if enc[0] is not None else 0
        right_cnt = enc[1] if enc[1] is not None else 0
        cv2.putText(display_img, f'Enc L: {left_cnt}  R: {right_cnt}', (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)
        # Prefer hardware-provided velocity if available
        if hw_vel is not None:
            counts_per_s_l, counts_per_s_r = hw_vel
        else:
            counts_per_s_l, counts_per_s_r = enc_rates

        cv2.putText(display_img, f'Counts/s L: {counts_per_s_l:.1f}  R: {counts_per_s_r:.1f}', (10, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)

        # Conversions: counts/s -> RPM, rad/s, m/s
        rpm_l = (counts_per_s_l / ENC_COUNTS_PER_REV) * 60.0
        rpm_r = (counts_per_s_r / ENC_COUNTS_PER_REV) * 60.0
        rad_s_l = (counts_per_s_l / ENC_COUNTS_PER_REV) * 2.0 * np.pi
        rad_s_r = (counts_per_s_r / ENC_COUNTS_PER_REV) * 2.0 * np.pi
        vel_m_s_l = rad_s_l * WHEEL_RADIUS_M
        vel_m_s_r = rad_s_r * WHEEL_RADIUS_M

        cv2.putText(display_img, f'RPM L:{rpm_l:.1f} R:{rpm_r:.1f}', (10, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)
        cv2.putText(display_img, f'Rad/s L:{rad_s_l:.2f} R:{rad_s_r:.2f}', (10, 210),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)
        cv2.putText(display_img, f'm/s L:{vel_m_s_l:.3f} R:{vel_m_s_r:.3f}', (10, 230),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)

        # Resize window for larger display
        cv2.namedWindow('Right Camera View', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Right Camera View', 1280, 960)
        cv2.imshow('Right Camera View', display_img)  # Show full camera view with overlays
        key = cv2.waitKey(1)  # Wait for key press (1 ms)
        # Kill switch: ESC key (27) to exit
        if key == 27:
            print("Kill switch activated: ESC pressed.")  # Print message if ESC is pressed
            break  # Exit control loop

        mtr_cmd = np.array([speed, steering])  # Create motor command array: [speed, steering]
        LEDs = np.array([0, 0, 0, 0, 0, 0, 1, 1])  # Set LED pattern (example)
        myCar.read_write_std(mtr_cmd, LEDs)  # Send speed/steering/LED command to QCar

        time.sleep(0.05)  # Small delay for control loop timing
finally:
    cv2.destroyAllWindows()  # Close all OpenCV windows
    myCar.terminate()  # Terminate QCar connection
    rightCam.terminate()  # Terminate camera connection
    




