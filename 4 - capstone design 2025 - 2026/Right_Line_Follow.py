from Quanser.product_QCar import QCar
from Quanser.q_essential import Camera2D
import cv2
import numpy as np
import time


# Initialize QCar and right-side Camera2D
myCar = QCar()
rightCam = Camera2D(camera_id="0", frame_width=640, frame_height=480, frame_rate=30.0)

# Parameters
target_offset = 50  # Desired pixel offset from right edge
speed = 0.2         # Forward speed

def get_right_line_offset(image):
    # Crop right side of image
    h, w, _ = image.shape
    right_crop = image[:, int(w*0.7):w]  # Crop right 30%
    # Convert to grayscale and threshold
    gray = cv2.cvtColor(right_crop, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Find largest contour (assume it's the line)
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            return cx
    return None

try:
    while True:
        rightCam.read()
        img = rightCam.image_data
        offset = get_right_line_offset(img)

        # Visualize the right crop and detected line
        h, w, _ = img.shape
        right_crop = img[:, int(w*0.7):w]
        display_img = right_crop.copy()
        if offset is not None:
            error = target_offset - offset
            steering = np.clip(error * 0.01, -1, 1)  # Simple proportional control
            cv2.circle(display_img, (offset, h//2), 10, (0,255,0), -1)
        else:
            steering = 0  # No line detected, go straight or stop

        cv2.imshow('Right Camera View', display_img)
        cv2.waitKey(1)

        # Use read_write_std for car control (speed, steering, LEDs)
        mtr_cmd = np.array([speed, steering])
        LEDs = np.array([0, 0, 0, 0, 0, 0, 1, 1])
        myCar.read_write_std(mtr_cmd, LEDs)
        time.sleep(0.05)
finally:
    cv2.destroyAllWindows()
    myCar.terminate()
    rightCam.terminate()
