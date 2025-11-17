"""
Untitled-1.py
Copied utility: open a camera stream and display it. This file was created because the editor "untitled" buffer was empty.

Usage:
    python c:/Users/Nova-/Downloads/Untitled-1.py --right

Keys while running:
  q or ESC - quit
  s        - save current frame (timestamped)

Requires: OpenCV (pip install opencv-python)
"""

import argparse
import time
import cv2
from datetime import datetime

# Try to use the same Camera2D helper used in Right_Turn.py when available. If
# the Quanser SDK is not installed on this machine, fall back to OpenCV's
# VideoCapture. This mirrors the camera activation used by the runtime scripts
# (Right_Turn.py / Left_Turn.py) while remaining portable for desktop testing.
USE_QUANSER_CAMERA = False
try:
    # Prefer the Quanser Camera2D wrapper when it exists in the environment
    from Quanser.q_essential import Camera2D
    USE_QUANSER_CAMERA = True
except Exception:
    USE_QUANSER_CAMERA = False


def try_open(indices, width=None, height=None, timeout=2.0):
    """Try opening each index in indices and return (cap, idx) of the first that works."""
    t0 = time.time()
    for idx in indices:
        # Use DirectShow backend on Windows for more reliable index handling
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        opened = cap.isOpened()
        if opened:
            if width is not None:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            if height is not None:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            return cap, idx
        else:
            cap.release()
        if time.time() - t0 > timeout:
            break
    return None, None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--id', type=int, help='Camera index to open (overrides --right)')
    p.add_argument('--right', action='store_true', help='Try common "right" camera indices (1,2,3)')
    p.add_argument('--width', type=int, default=640, help='Requested capture width')
    p.add_argument('--height', type=int, default=480, help='Requested capture height')
    args = p.parse_args()

    # If Quanser Camera2D is available, create it using the same parameters
    # used in Right_Turn.py. Otherwise, fall back to probing device indices.
    if USE_QUANSER_CAMERA:
        cam_id = str(args.id) if args.id is not None else "0"
        try:
            cam = Camera2D(camera_id=cam_id, frame_width=args.width, frame_height=args.height, frame_rate=30.0)
            print(f'Opened Quanser Camera2D id={cam_id}')
            use_quanser = True
        except Exception as e:
            print('Quanser Camera2D failed to open:', e)
            print('Falling back to OpenCV VideoCapture probing')
            use_quanser = False
    else:
        use_quanser = False

    if not use_quanser:
        if args.id is not None:
            indices = [args.id]
        elif args.right:
            indices = [1, 2, 3, 0]
        else:
            indices = [0, 1, 2]

        cap, idx = try_open(indices, width=args.width, height=args.height)
        if cap is None:
            print('No camera could be opened. Tried indices:', indices)
            return

        print(f'Opened camera index {idx}. Press q or ESC to quit, s to save a frame.')
        win = f'Camera {idx}'
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    last_t = time.time()
    frame_count = 0
    fps = 0.0
    while True:
        if USE_QUANSER_CAMERA and use_quanser:
            # read from the Camera2D wrapper similar to Right_Turn.py
            cam.read()
            frame = cam.image_data
            ret = frame is not None and getattr(frame, 'size', 1) != 0
            if not ret:
                print('Warning: Quanser Camera returned invalid image data.')
                time.sleep(0.05)
                continue
        else:
            ret, frame = cap.read()
            if not ret:
                print('Frame read failed, exiting.')
                break

        frame_count += 1
        if frame_count % 30 == 0:
            now = time.time()
            fps = 30.0 / max(1e-6, now - last_t)
            last_t = now

        text = f'CAM={idx} {args.width}x{args.height} FPS={fps:.1f}'
        cv2.putText(frame, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

        cv2.imshow(win, frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q') or k == 27:
            break
        elif k == ord('s'):
            fname = f'camera_{idx}_frame_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
            cv2.imwrite(fname, frame)
            print('Saved', fname)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
