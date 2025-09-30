from Quanser.product_QCar import QCar
from Quanser.q_essential import Camera3D
import cv2
import numpy as np
import time

# Initialize QCar and right-side Camera
car = QCar()
camera = Camera3D(0)  # Right-side camera

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
        img = camera.read()
        offset = get_right_line_offset(img)
        if offset is not None:
            error = target_offset - offset
            steering = np.clip(error * 0.01, -1, 1)  # Simple proportional control
        else:
            steering = 0  # No line detected, go straight or stop

        car.set_speed(speed)
        car.set_steering(steering)
        time.sleep(0.05)
finally:
    car.set_speed(0)
    car.set_steering(0)
    car.terminate()
    camera.terminate()