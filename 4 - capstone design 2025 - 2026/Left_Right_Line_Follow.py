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

# Create QCar object for robot control
myCar = QCar()
# Create right camera object
rightCam = Camera2D(camera_id="0", frame_width=640, frame_height=480, frame_rate=30.0)
# Create left camera object
leftCam = Camera2D(camera_id="2", frame_width=640, frame_height=480, frame_rate=30.0)

# Desired pixel offset from right edge for line following
target_offset = 50
# Forward speed of the robot (lower value for slower movement)
speed = 0.07

# Function to find the x-position of the detected line in the right crop
def get_line_offset(image, side='right'):
    h, w, _ = image.shape  # Get image dimensions
    lower_half = image[h//2:h, :]  # Only lower half
    if side == 'right':
        crop = lower_half[:, int(w*0.7):w]  # Crop right 30%
        offset_base = int(w*0.7)
    else:
        crop = lower_half[:, 0:int(w*0.3)]  # Crop left 30%
        offset_base = 0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay_info = None
    if contours:
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            contour_full = largest + np.array([offset_base, h//2])
            centroid_full = (offset_base+cx, h//2+cy)
            overlay_info = {
                'contour': contour_full,
                'centroid': centroid_full,
                'offset': cx
            }
            return overlay_info
    return None

try:
    while True:  # Main control loop
        rightCam.read()
        leftCam.read()
        img_right = rightCam.image_data
        img_left = leftCam.image_data
        # Check for valid image data
        if (img_right is None or img_right.shape[0] == 0 or img_right.shape[1] == 0 or
            img_left is None or img_left.shape[0] == 0 or img_left.shape[1] == 0):
            print("Warning: Camera returned invalid image data.")
            continue

        overlay_right = get_line_offset(img_right, side='right')
        overlay_left = get_line_offset(img_left, side='left')

        h, w, _ = img_right.shape
        display_img_right = img_right.copy()
        display_img_left = img_left.copy()

        steering = 0
        overlays_drawn = False
        if overlay_right is not None:
            error_right = target_offset - overlay_right['offset']
            steering += np.clip(error_right * 0.005, -1, 1)
            cv2.drawContours(display_img_right, [overlay_right['contour']], -1, (255,0,0), 2)
            cv2.circle(display_img_right, overlay_right['centroid'], 10, (0,0,255), -1)
            overlays_drawn = True
        if overlay_left is not None:
            error_left = overlay_left['offset'] - target_offset
            steering -= np.clip(error_left * 0.005, -1, 1)
            cv2.drawContours(display_img_left, [overlay_left['contour']], -1, (0,255,0), 2)
            cv2.circle(display_img_left, overlay_left['centroid'], 10, (0,255,255), -1)
            overlays_drawn = True
        if not overlays_drawn:
            steering = 0

        cv2.imshow('Right Camera View', display_img_right)
        cv2.imshow('Left Camera View', display_img_left)
        key = cv2.waitKey(1)
        if key == 27:
            print("Kill switch activated: ESC pressed.")
            break

        mtr_cmd = np.array([speed, steering])
        LEDs = np.array([0, 0, 0, 0, 0, 0, 1, 1])
        myCar.read_write_std(mtr_cmd, LEDs)
        time.sleep(0.05)
finally:
    cv2.destroyAllWindows()  # Close all OpenCV windows
    myCar.terminate()  # Terminate QCar connection
    rightCam.terminate()  # Terminate right camera connection
    leftCam.terminate()  # Terminate left camera connection
