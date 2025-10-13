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
speed = 0.075

# Frame counter and FPS calculation
frame_count = 0
fps = 0
last_time = time.time()

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
            steering = np.clip(error * 0.005, -1, 1)  # Reduced gain for smoother turns
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

        # Put frame count, FPS, and computation time on image
        cv2.putText(display_img, f'Frames: {frame_count}  FPS: {fps}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.putText(display_img, f'Calc Time: {calc_time_ms:.1f} ms', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

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
    

