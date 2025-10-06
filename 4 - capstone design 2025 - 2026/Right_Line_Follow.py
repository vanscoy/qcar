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

# Desired pixel offset from right edge for line following
target_offset = 50
# Forward speed of the robot (lower value for slower movement)
speed = 0.07

# Function to find the x-position of the detected line in the right crop
def get_right_line_offset(image):
    h, w, _ = image.shape  # Get image dimensions
    # Crop lower half and right 30% of the image for line detection
    lower_half = image[h//2:h, :]  # Only lower half
    right_crop = lower_half[:, int(w*0.7):w]  # Crop right 30% of lower half
    gray = cv2.cvtColor(right_crop, cv2.COLOR_BGR2GRAY)  # Convert cropped image to grayscale
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)  # Threshold to highlight bright lines
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours
    if contours:  # If any contours are found
        largest = max(contours, key=cv2.contourArea)  # Select the largest contour (assumed to be the line)
        M = cv2.moments(largest)  # Calculate moments for the largest contour
        if M['m00'] > 0:  # Prevent division by zero
            cx = int(M['m10'] / M['m00'])  # Compute center x-position of the contour (in right_crop)
            cy = int(M['m01'] / M['m00'])  # Compute center y-position of the contour (in right_crop)
            # Draw the contour in blue on the full image (adjust coordinates)
            contour_full = largest + np.array([int(w*0.7), h//2])
            cv2.drawContours(image, [contour_full], -1, (255,0,0), 2)
            # Draw a red dot at the centroid (adjust coordinates to full image)
            cv2.circle(image, (int(w*0.7)+cx, h//2+cy), 10, (0,0,255), -1)
            return cx  # Return x-position of detected line (in right_crop)
    return None  # Return None if no line is found

try:
    while True:  # Main control loop
        rightCam.read()  # Capture image from right camera
        img = rightCam.image_data  # Get image data from camera object
        # Check for valid image data
        if img is None or img.shape[0] == 0 or img.shape[1] == 0:
            print("Warning: Camera returned invalid image data.")  # Warn if image is invalid
            continue  # Skip to next loop iteration

        offset = get_right_line_offset(img)  # Find line offset in the image

        h, w, _ = img.shape  # Get image dimensions
        right_crop = img[:, int(w*0.7):w]  # Crop right 30% for visualization
        display_img = right_crop.copy()  # Copy cropped image for display

        if offset is not None:  # If line is detected
            error = target_offset - offset  # Calculate error from desired offset
            steering = np.clip(error * 0.01, -1, 1)  # Proportional control for steering, clipped to [-1, 1]
            cv2.circle(display_img, (offset, h//2), 10, (0,255,0), -1)  # Draw green circle at detected line
        else:
            steering = 0  # No line detected, go straight

        cv2.imshow('Right Camera View', display_img)  # Show cropped camera view with detection
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




