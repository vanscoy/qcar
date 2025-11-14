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
speed = 0.072
steering_gain = 0.009  # Gain used for steering calculation (increased per user request)
max_steering_angle = 28  # Maximum steering angle in degrees (mechanical limit)
# Runtime steering invert: when True steering is multiplied by -1 before sending
steering_invert = True

SPEED_MAX = 0.078
SPEED_MIN = 0.072
SPEED_KP = 2.96296e-05


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

def get_right_line_offset(image):
    h, w, _ = image.shape  # Get image dimensions
    # remove left 20% and right 20% -> keep the middle 60% horizontally
    crop_x = int(w * 0.2)  # left boundary (remove left 20%)
    right_crop = int(w * 0.8)  # right boundary (remove right 20%)
    # keep vertical band from 45% -> 55% of the frame (narrower band to reduce spurious lines)
    top_crop = int(h * 0.45)
    bottom_crop = int(h * 0.55)  # bottom moved up to 55% (remove bottom 45%)
    lower_half = image[top_crop:bottom_crop, crop_x:right_crop]
    gray = cv2.cvtColor(lower_half, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay_info = None
    if contours:
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M.get('m00', 0) > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
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
        start_calc = time.time()  # Start timing calculation

        # Capture image from left camera
        leftCam.read()
        img = leftCam.image_data

        # Check for valid image data
        if img is None or img.shape[0] == 0 or img.shape[1] == 0:
            print("Warning: Camera returned invalid image data.")
            time.sleep(0.05)
            continue

        overlay_info, thresh = get_right_line_offset(img)  # Get overlay and thresh from crop

        h, w, _ = img.shape  # Get image dimensions
        display_img = img.copy()  # Show full camera view

        # Update frame counter and FPS
        frame_count += 1
        current_time = time.time()
        if current_time - last_time >= 1.0:
            fps = frame_count
            frame_count = 0
            last_time = current_time

        # Draw processing-area outline (keep vertical band from 45%->55%, remove left/right 20%)
        crop_x = int(w * 0.2)
        right_crop = int(w * 0.8)
        crop_y = int(h * 0.45)  # top of the kept vertical band (45% down)
        bottom_crop = int(h * 0.55)  # bottom moved up to 55% to remove bottom 45%
        crop_w = right_crop - crop_x
        crop_h = bottom_crop - crop_y
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

        # Encoders are disabled in this trimmed script; provide zeroed values
        speed_m_s = 0.0
        dist_delta = 0.0
        total_distance_m = 0.0

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
