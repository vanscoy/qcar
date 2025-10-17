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
# If no discrete encoder counts are exposed by the model/driver, fall back to using
# the first element returned by read_write_std() as a motor velocity (revolutions/sec).
# Set to False to disable this heuristic.
FALLBACK_RET0_AS_RPS = False

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


def run_active_encoder_probe(car, duration=2.0, speed=0.05, sample_dt=0.1):
    """Active probe: briefly command the motors to move and sample encoder-related outputs.
    Appends timestamped samples to encoder_diag.txt so we can see whether encoder values change.
    Uses a small speed and short duration to minimize movement.
    """
    try:
        start = time.time()
        samples = []
        mtr_cmd = np.array([speed, 0.0])
        stop_cmd = np.array([0.0, 0.0])
        LEDs = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        # Send command and sample repeatedly for duration
        while time.time() - start < duration:
            t = time.time()
            try:
                ret = None
                try:
                    ret = car.read_write_std(mtr_cmd, LEDs)
                except Exception as e:
                    ret = f'read_write_std raised {e}'
                # probe read_encoder and mtr_encoder
                r_enc = None
                try:
                    fn = getattr(car, 'read_encoder', None)
                    if callable(fn):
                        r_enc = fn()
                except Exception as e:
                    r_enc = f'read_encoder raised {e}'
                try:
                    m_enc = getattr(car, 'mtr_encoder', None)
                except Exception as e:
                    m_enc = f'mtr_encoder access raised {e}'
                samples.append((t, repr(ret), repr(r_enc), repr(m_enc)))
            except Exception:
                pass
            # small sleep between samples
            time.sleep(sample_dt)
        # Stop motors
        try:
            car.read_write_std(stop_cmd, LEDs)
        except Exception:
            pass

        # Append samples to encoder_diag.txt
        try:
            with open('encoder_diag.txt', 'a') as f:
                f.write('\n=== Active encoder probe samples ===\n')
                for s in samples:
                    ts = s[0]
                    f.write(f'{ts}: read_write_std -> {s[1]} | read_encoder -> {s[2]} | mtr_encoder -> {s[3]}\n')
            print('Appended active encoder probe samples to encoder_diag.txt')
        except Exception as e:
            print('Failed to append active probe samples:', e)
    except Exception as e:
        print('Active encoder probe failed:', e)


def run_wheel_mapping_probe(car, phases=None, phase_time=0.6, speed=0.06):
    """Automated wheel mapping routine.
    phases: list of (name, steering) tuples. Each phase runs the given steering value
    at `speed` for `phase_time` seconds while sampling the encoder scalar from
    read_write_std(...)[2] or read_encoder(). Appends a human-readable summary
    to encoder_diag.txt.
    """
    try:
        if phases is None:
            # forward, hard-left, hard-right (short duration each)
            phases = [('forward', 0.0), ('left', -1.0), ('right', 1.0)]

        LEDs = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        summary_lines = []
        summary_lines.append('=== Wheel mapping probe ===')

        for name, steer in phases:
            # sample initial encoder values (try read_write_std()[2] then read_encoder())
            enc0_left = None
            enc0_right = None
            try:
                ret0 = car.read_write_std(np.array([0.0, 0.0]), LEDs)
                if isinstance(ret0, (list, tuple)) and len(ret0) >= 3:
                    enc_array0 = ret0[2]
                    if isinstance(enc_array0, (list, tuple, np.ndarray)) and len(enc_array0) >= 2:
                        enc0_left = int(enc_array0[0]); enc0_right = int(enc_array0[1])
                    elif isinstance(enc_array0, (int, float)):
                        enc0_left = int(enc_array0); enc0_right = None
            except Exception:
                pass
            try:
                r0 = getattr(car, 'read_encoder', None)
                if callable(r0):
                    v = r0()
                    if isinstance(v, (list, tuple, np.ndarray)) and len(v) >= 2:
                        enc0_left = int(v[0]); enc0_right = int(v[1])
                    elif isinstance(v, (list, tuple, np.ndarray)) and len(v) == 1:
                        enc0_left = int(v[0]); enc0_right = None
                    elif isinstance(v, int):
                        enc0_left = int(v); enc0_right = None
            except Exception:
                pass

            # run phase
            cmd = np.array([speed, steer])
            start = time.time()
            # While driving, capture the last non-empty read_write_std return (enc values are often present
            # only when the motor command is non-zero). We'll store the last_ret to extract encoder counts.
            last_ret = None
            while time.time() - start < phase_time:
                try:
                    ret = car.read_write_std(cmd, LEDs)
                    last_ret = ret
                except Exception:
                    pass
                time.sleep(0.05)

            # derive final encoder values from the last non-empty ret while moving; fall back to read_encoder()
            enc1_left = None
            enc1_right = None
            try:
                if last_ret is not None and isinstance(last_ret, (list, tuple)) and len(last_ret) >= 3:
                    enc_array1 = last_ret[2]
                    if isinstance(enc_array1, (list, tuple, np.ndarray)) and len(enc_array1) >= 2:
                        enc1_left = int(enc_array1[0]); enc1_right = int(enc_array1[1])
                    elif isinstance(enc_array1, (int, float)):
                        enc1_left = int(enc_array1); enc1_right = None
            except Exception:
                pass
            if enc1_left is None and enc1_right is None:
                try:
                    r1 = getattr(car, 'read_encoder', None)
                    if callable(r1):
                        v = r1()
                        if isinstance(v, (list, tuple, np.ndarray)) and len(v) >= 2:
                            enc1_left = int(v[0]); enc1_right = int(v[1])
                        elif isinstance(v, (list, tuple, np.ndarray)) and len(v) == 1:
                            enc1_left = int(v[0])
                        elif isinstance(v, int):
                            enc1_left = int(v)
                except Exception:
                    pass

            # compute deltas
            delta_left = None
            delta_right = None
            try:
                if enc0_left is not None and enc1_left is not None:
                    delta_left = enc1_left - enc0_left
                if enc0_right is not None and enc1_right is not None:
                    delta_right = enc1_right - enc0_right
            except Exception:
                delta_left = None
                delta_right = None

            summary_lines.append(
                f'Phase {name}: steer={steer} enc_left_before={enc0_left} enc_left_after={enc1_left} delta_left={delta_left} | '
                f'enc_right_before={enc0_right} enc_right_after={enc1_right} delta_right={delta_right}'
            )

            # small pause between phases
            try:
                car.read_write_std(np.array([0.0, 0.0]), LEDs)
            except Exception:
                pass
            time.sleep(0.15)

        # Stop motors (safety)
        try:
            car.read_write_std(np.array([0.0, 0.0]), LEDs)
        except Exception:
            pass

        # Append summary to diagnostic file
        try:
            with open('encoder_diag.txt', 'a') as f:
                f.write('\n' + '\n'.join(summary_lines) + '\n')
            print('\n'.join(summary_lines))
            print('Appended wheel mapping summary to encoder_diag.txt')
        except Exception as e:
            print('Failed to append wheel mapping summary:', e)
    except Exception as e:
        print('Wheel mapping probe failed:', e)
def run_per_motor_probe(car, speed=0.06, duration=0.6, sample_dt=0.05):
    """Focused per-motor probe: drive two conservative commands and log per-channel encoder changes.
    This helps map encoder channel index -> left/right wheel without needing model changes.
    """
    LEDs = np.array([0, 0, 0, 0, 0, 0, 0, 0])

    def sample_enc():
        # Return (enc_left, enc_right) or (val, None) for single-channel
        try:
            r = car.read_write_std(np.array([0.0, 0.0]), LEDs)
            if isinstance(r, (list, tuple)) and len(r) >= 3:
                a = r[2]
                if isinstance(a, (list, tuple, np.ndarray)) and len(a) >= 2:
                    return int(a[0]), int(a[1])
                if isinstance(a, (int, float)):
                    return int(a), None
        except Exception:
            pass
        try:
            fn = getattr(car, 'read_encoder', None)
            if callable(fn):
                v = fn()
                if isinstance(v, (list, tuple, np.ndarray)) and len(v) >= 2:
                    return int(v[0]), int(v[1])
                if isinstance(v, (list, tuple, np.ndarray)) and len(v) == 1:
                    return int(v[0]), None
                if isinstance(v, int):
                    return int(v), None
        except Exception:
            pass
        return None, None

    try:
        with open('encoder_diag.txt', 'a') as f:
            f.write('\n=== Per-motor probe ===\n')

        # Phase A: conservative command that likely moves motor A
        before_a = sample_enc()
        cmdA = np.array([speed, 0.0])
        start = time.time()
        last_ret = None
        while time.time() - start < duration:
            try:
                ret = car.read_write_std(cmdA, LEDs)
                last_ret = ret
            except Exception:
                pass
            time.sleep(sample_dt)
        # Prefer encoder values from the last non-empty ret captured during motion
        after_a = None
        try:
            if last_ret is not None and isinstance(last_ret, (list, tuple)) and len(last_ret) >= 3:
                a = last_ret[2]
                if isinstance(a, (list, tuple, np.ndarray)) and len(a) >= 2:
                    after_a = (int(a[0]), int(a[1]))
                elif isinstance(a, (int, float)):
                    after_a = (int(a), None)
        except Exception:
            after_a = None
        if after_a is None:
            after_a = sample_enc()
        with open('encoder_diag.txt', 'a') as f:
            f.write(f'MotorA cmd {cmdA.tolist()} enc_before={before_a} enc_after={after_a} delta={(None if before_a[0] is None or after_a[0] is None else after_a[0]-before_a[0], None if before_a[1] is None or after_a[1] is None else after_a[1]-before_a[1])}\n')

        # Stop briefly
        try:
            car.read_write_std(np.array([0.0, 0.0]), LEDs)
        except Exception:
            pass
        time.sleep(0.15)

        # Phase B: conservative command that likely moves motor B
        before_b = sample_enc()
        cmdB = np.array([0.0, speed])
        start = time.time()
        last_ret = None
        while time.time() - start < duration:
            try:
                ret = car.read_write_std(cmdB, LEDs)
                last_ret = ret
            except Exception:
                pass
            time.sleep(sample_dt)
        after_b = None
        try:
            if last_ret is not None and isinstance(last_ret, (list, tuple)) and len(last_ret) >= 3:
                a = last_ret[2]
                if isinstance(a, (list, tuple, np.ndarray)) and len(a) >= 2:
                    after_b = (int(a[0]), int(a[1]))
                elif isinstance(a, (int, float)):
                    after_b = (int(a), None)
        except Exception:
            after_b = None
        if after_b is None:
            after_b = sample_enc()
        with open('encoder_diag.txt', 'a') as f:
            f.write(f'MotorB cmd {cmdB.tolist()} enc_before={before_b} enc_after={after_b} delta={(None if before_b[0] is None or after_b[0] is None else after_b[0]-before_b[0], None if before_b[1] is None or after_b[1] is None else after_b[1]-before_b[1])}\n')

        # Stop motors
        try:
            car.read_write_std(np.array([0.0, 0.0]), LEDs)
        except Exception:
            pass
        with open('encoder_diag.txt', 'a') as f:
            f.write('Appended per-motor probe results.\n')
        print('Per-motor probe appended to encoder_diag.txt')
    except Exception as e:
        print('Per-motor probe failed:', e)

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
    # Perform an active probe: briefly command the motors and log encoder-related samples
    try:
        run_active_encoder_probe(myCar, duration=2.0, speed=0.05, sample_dt=0.1)
    except Exception:
        pass
    # Perform a short automated wheel mapping probe to attribute encoder increments to steering phases
    try:
        run_wheel_mapping_probe(myCar, phases=[('forward',0.0), ('left', -1.0), ('right', 1.0)], phase_time=0.6, speed=0.06)
    except Exception:
        pass
    # Perform a per-motor probe to disambiguate encoder channel mapping
    try:
        run_per_motor_probe(myCar, speed=0.06, duration=0.5, sample_dt=0.05)
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
        enc_source = 'none'
        now_t = time.time()
        try:
            ret = myCar.read_write_std(mtr_cmd, LEDs)
            # Typical return: (current, batteryVoltage, encoderCounts)
            if isinstance(ret, (list, tuple)):
                # Common patterns: (float, voltage, counts) or (velocity, voltage, counts)
                # If third element present, treat as counts or single-channel count
                if len(ret) >= 3:
                    enc_ret = ret[2]
                    if isinstance(enc_ret, (list, tuple, np.ndarray)):
                        if len(enc_ret) >= 2:
                            enc = (int(enc_ret[0]), int(enc_ret[1]))
                            enc_source = 'read_write_std[2] (counts)'
                        elif len(enc_ret) == 1:
                            enc = (int(enc_ret[0]), None)
                            enc_source = 'read_write_std[2] (counts, single)'
                    else:
                        # third element might be a scalar int count
                        try:
                            enc_val = int(enc_ret)
                            enc = (enc_val, None)
                            enc_source = 'read_write_std[2] (counts scalar)'
                        except Exception:
                            pass
                # If no counts found but fallback enabled, consider ret[0] as velocity (rev/s)
                if enc is None and FALLBACK_RET0_AS_RPS and len(ret) >= 1:
                    try:
                        # ret[0] may be a small float representing rev/s or rad/s. We'll assume rev/s.
                        rps = float(ret[0])
                        # Convert rev/s to rad/s and m/s
                        rad_s = rps * 2.0 * np.pi
                        vel_m_s = rad_s * WHEEL_RADIUS_M
                        # store as hw_vel (counts/s not available)
                        hw_vel = (rps * ENC_COUNTS_PER_REV, None)
                        enc_source = 'read_write_std[0] (assumed rps)'
                    except Exception:
                        pass
        except Exception:
            # If read_write_std fails to return enc, fall back to probing later
            enc = None

        # If hardware-provided velocity is available via other methods, try to get it
        vel_try = read_encoder_velocity(myCar)
        if vel_try is not None:
            hw_vel = vel_try
            enc_source = 'read_encoder() or read_std() velocities'

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
                    # handle single-channel counts (right may be None)
                    left_rate = (enc[0] - (prev_enc_counts[0] if prev_enc_counts[0] is not None else 0)) / dt
                    if enc[1] is None or prev_enc_counts[1] is None:
                        right_rate = 0.0
                    else:
                        right_rate = (enc[1] - prev_enc_counts[1]) / dt
                    enc_rates = (left_rate, right_rate)
                prev_enc_counts = enc
                prev_enc_time = now_t
        else:
            # last resort: probe using API if read_write_std did not give encoderCounts
            enc = read_encoders(myCar) or (None, None)
            if enc is not None:
                enc_source = 'read_encoders()'

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
        # If enc is single-channel (enc = (val, None)) treat as motor encoder scalar
        left_cnt = enc[0] if enc and enc[0] is not None else 0
        right_cnt = enc[1] if enc and enc[1] is not None else None
        if right_cnt is None:
            cv2.putText(display_img, f'MotorEnc: {left_cnt}', (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)
        else:
            cv2.putText(display_img, f'Enc L: {left_cnt}  R: {right_cnt}', (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)
        # Prefer hardware-provided velocity if available
        if hw_vel is not None:
            counts_per_s_l, counts_per_s_r = hw_vel
        else:
            counts_per_s_l, counts_per_s_r = enc_rates

        # Ensure we have numeric rates for computation (use 0.0 when missing)
        counts_per_s_l = float(counts_per_s_l) if counts_per_s_l is not None else 0.0
        counts_per_s_r = float(counts_per_s_r) if counts_per_s_r is not None else 0.0

        # Show 'N/A' for right channel if absent; if only a motor scalar exists, label accordingly
        if enc and enc[1] is None:
            cv2.putText(display_img, f'Counts/s Motor: {counts_per_s_l:.1f}', (10, 170),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)
        else:
            r_display = f'{counts_per_s_r:.1f}' if enc and enc[1] is not None else 'N/A'
            cv2.putText(display_img, f'Counts/s L: {counts_per_s_l:.1f}  R: {r_display}', (10, 170),
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

        # Show which source we used for encoder/velocity
        cv2.putText(display_img, f'Enc source: {enc_source}', (10, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,200,100), 2)

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

