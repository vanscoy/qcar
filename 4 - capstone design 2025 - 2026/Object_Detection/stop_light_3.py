import time
import math
import enum
import numpy as np
import cv2

from qvl.qlabs import QuanserInteractiveLabs
from qvl.qcar2 import QLabsQCar2

# ------------------------------------------------------------
# Camera + Car Init (RGB + Depth)
# ------------------------------------------------------------
myCar = QCar()
frontCam = Camera3D(mode='RGB&DEPTH',
                    frame_width_RGB=1280,
                    frame_height_RGB=720,
                    frame_rate_RGB=20.0,
                    device_id='0')

# ------------------------------------------------------------
# Parameters
# ------------------------------------------------------------
WINDOW = "QCar Traffic Light Demo"
WINDOWS = ["Upper ROI", "Red Mask", "Yellow Mask", 
           "Combined Mask", WINDOW]

base_speed = 0.08
stop_duration = 3.0

# Crop to upper portion (top 40% of frame)
UPPER_ROI_RATIO = 0.4

# HSV thresholds for RED traffic light
RED1_LOWER = np.array([0, 150, 150])
RED1_UPPER = np.array([10, 255, 255])
RED2_LOWER = np.array([170, 150, 150])
RED2_UPPER = np.array([179, 255, 255])

# HSV thresholds for YELLOW traffic light
YELLOW_LOWER = np.array([15, 100, 100])
YELLOW_UPPER = np.array([40, 255, 255])

# Traffic light detection parameters
MIN_LIGHT_AREA = 200
MAX_LIGHT_AREA = 15000
MIN_ASPECT_RATIO = 0.3  # Lights are more vertical/circular
MAX_ASPECT_RATIO = 3.0

MAX_LIGHT_DISTANCE_M = 5.0  # Detect lights up to 5 meters away

# State
stopping = False
stop_timer = 0.0
last_detected_color = None
last_light_seen_time = 0.0
CLEAR_DURATION = 1.0  # Must not see light for 1 second before resuming


# ------------------------------------------------------------
# Helper: safe depth -> float
# ------------------------------------------------------------
def read_distance_scalar(depth_img, cx, cy):
    """
    Safely read a single distance value from depth_img at (cx, cy)
    and return a Python float or math.nan if invalid.
    """
    try:
        if depth_img is None:
            return math.nan
        h, w = depth_img.shape[:2]
        if cx < 0 or cy < 0 or cx >= w or cy >= h:
            return math.nan

        val = depth_img[cy, cx]

        if isinstance(val, np.ndarray):
            if val.size == 0:
                return math.nan
            val = val.flat[0]

        dist = float(val)
        if not math.isfinite(dist) or dist <= 0.0:
            return math.nan
        return dist
    except Exception:
        return math.nan


# ------------------------------------------------------------
# Traffic Light Detection with Depth Check
# ------------------------------------------------------------
def detect_traffic_light(frame, depth_img):
    """
    Detect red or yellow traffic lights in the upper portion of the frame.
    Returns: (detected, color) where color is 'red' or 'yellow' or None
    """
    global last_detected_color
    
    # Crop to upper region
    height = frame.shape[0]
    upper_height = int(height * UPPER_ROI_RATIO)
    upper_roi = frame[0:upper_height, :]
    upper_depth = depth_img[0:upper_height, :] if depth_img is not None else None
    
    # Show upper ROI
    cv2.imshow("Upper ROI", upper_roi)
    
    # Convert to HSV
    hsv = cv2.cvtColor(upper_roi, cv2.COLOR_BGR2HSV)
    
    # Create masks for red and yellow
    red_mask1 = cv2.inRange(hsv, RED1_LOWER, RED1_UPPER)
    red_mask2 = cv2.inRange(hsv, RED2_LOWER, RED2_UPPER)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    
    yellow_mask = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)
    
    cv2.imshow("Red Mask", red_mask)
    cv2.imshow("Yellow Mask", yellow_mask)
    
    # Combined mask
    combined_mask = cv2.bitwise_or(red_mask, yellow_mask)
    cv2.imshow("Combined Mask", combined_mask)
    
    # Noise filtering
    combined_mask = cv2.GaussianBlur(combined_mask, (5, 5), 0)
    kernel = np.ones((3, 3), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    
    detected_lights = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_LIGHT_AREA or area > MAX_LIGHT_AREA:
            continue
        
        x, y, w, h = cv2.boundingRect(cnt)
        ar = float(w) / float(h) if h != 0 else 0.0
        if ar < MIN_ASPECT_RATIO or ar > MAX_ASPECT_RATIO:
            continue
        
        # Depth check
        cx, cy = x + w // 2, y + h // 2
        dist = read_distance_scalar(upper_depth, cx, cy) if upper_depth is not None else math.nan
        
        # Skip if too far or invalid
        if not math.isnan(dist) and dist > MAX_LIGHT_DISTANCE_M:
            continue
        
        # Determine color by checking which mask has more pixels in this region
        roi_red = red_mask[y:y+h, x:x+w]
        roi_yellow = yellow_mask[y:y+h, x:x+w]
        
        red_pixels = cv2.countNonZero(roi_red) if roi_red.size > 0 else 0
        yellow_pixels = cv2.countNonZero(roi_yellow) if roi_yellow.size > 0 else 0
        
        # Determine dominant color
        if red_pixels > yellow_pixels and red_pixels > 0:
            color = 'red'
            box_color = (0, 0, 255)  # Red in BGR
        elif yellow_pixels > red_pixels and yellow_pixels > 0:
            color = 'yellow'
            box_color = (0, 255, 255)  # Yellow in BGR
        else:
            continue
        
        detected_lights.append({
            'x': x,
            'y': y,
            'w': w,
            'h': h,
            'color': color,
            'distance': dist,
            'box_color': box_color
        })
    
    # Draw detections on the full frame
    for light in detected_lights:
        x, y, w, h = light['x'], light['y'], light['w'], light['h']
        color_name = light['color'].upper()
        dist = light['distance']
        
        # Draw rectangle on upper portion of original frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), light['box_color'], 3)
        
        # Add label
        dist_str = "{:.2f}m".format(dist) if not math.isnan(dist) else "N/A"
        label = "{} LIGHT {}".format(color_name, dist_str)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, light['box_color'], 2)
        
        # Update last detected color
        last_detected_color = light['color']
    
    # Return True if red or yellow light detected
    if detected_lights:
        colors = [light['color'] for light in detected_lights]
        if 'red' in colors or 'yellow' in colors:
            return True, colors[0]
    
    return False, None


# ------------------------------------------------------------
# Main Loop
# ------------------------------------------------------------
try:
    while True:
        frontCam.read_RGB()
        frontCam.read_depth(dataMode='m')

        img = frontCam.image_buffer_RGB
        depth = frontCam.image_buffer_depth_m

        if img is None or img.size == 0:
            continue
        if depth is None or getattr(depth, "size", 0) == 0:
            continue

        display = img.copy()
        light_detected, color = detect_traffic_light(display, depth)

        # Update last light seen time
        if light_detected:
            last_light_seen_time = time.time()

        # Stop logic - stop for both red and yellow lights
        if light_detected and not stopping:
            stopping = True
            stop_timer = time.time()
            speed = 0.0
            print(f"Stopping for {color} light!")
        elif stopping:
            speed = 0.0
            # Check if stop duration has passed AND no light seen for CLEAR_DURATION
            if time.time() - stop_timer >= stop_duration:
                time_since_last_light = time.time() - last_light_seen_time
                if time_since_last_light >= CLEAR_DURATION:
                    stopping = False
                    speed = base_speed
                    print("Resuming movement - light clear for 1 second")
                else:
                    # Still waiting for clear period
                    print(f"Waiting for clear signal... {CLEAR_DURATION - time_since_last_light:.1f}s remaining")
        else:
            speed = base_speed

        # Display status
        status_text = "STOPPED" if stopping else "MOVING"
        status_color = (0, 0, 255) if stopping else (0, 255, 0)
        cv2.putText(display, status_text, (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, status_color, 3)

        # Drive
        mtr_cmd = np.array([speed, 0.0], dtype=np.float64)
        LEDs = np.array([0, 0, 0, 0, 0, 0, 1, 1], dtype=np.float64)
        myCar.read_write_std(mtr_cmd, LEDs)

        cv2.imshow(WINDOW, display)
        if (cv2.waitKey(1) & 0xFF) == 27:
            break

        time.sleep(0.05)

finally:
    for window in WINDOWS:
        try:
            cv2.destroyWindow(window)
        except Exception:
            pass
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass
    try:
        myCar.terminate()
    except Exception:
        pass
    if hasattr(frontCam, "terminate"):
        try:
            frontCam.terminate()
        except Exception:
            pass
