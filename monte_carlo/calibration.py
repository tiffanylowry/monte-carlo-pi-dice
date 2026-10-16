'''Handles persepctive calibration by capturing mouse clicks on the webcam feed for a cleaner top-down perspective'''

import cv2
import numpy as np
from .config import CANVAS_SIZE, WINDOW_CALIBRATION

# Global storage for selcted calibration corners
ref_pts = []

def click_event(event: int, x: int, y: int, flags: int, param) -> None:
    '''Mouse callback handler that record when left mouse button is clicked on the calibration display window.
    
    Args:
        event: OpenCV mouse event type
        x: X-coordinate of the mouse cursor
        y: Y-corodinate of the mouse cursor
        flags: Relevant OpenCV flags associated with the event
        param: Optional user-defined param passed to callback

    Returns:
        None
    '''

    global ref_pts

    if event == cv2.EVENT_LBUTTONDOWN:
        ref_pts.append((x,y))
        print(f"Corner slected: ({x},{y})")

def get_perspective_matrix(cap: cv2.VideoCapture) -> np.ndarray | None:
    '''Walk users throught selecting the foru grid points on the wevcae and computes the persepctive transformation matrix
    
    Args:
        cap: Active OpenCV VideoCapture object that reads webcam frames

    Returns:
        3x3 perspective trasnformation matrix as a  NumPy ndarray OR none if calibration fails
    '''

    global ref_pts
    ref_pts.clear() # Clear points if still present ffrom previosu attempts

    print("Step 1: Position your webcam to view the grid (1x1m).")
    print("Click the 4 corners of your grid in this order: Top-Left, Top-Right, Bottom-Right, Bottom-Left.")
    print("Press 'c' when all 4 corners are selcted or 'q' to quit")

    cv2.namedWindow(WINDOW_CALIBRATION)
    cv2.setMouseCallback(WINDOW_CALIBRATION, click_event)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to grab frame.")
            break

        display_frame = frame.copy()

        # Draw red indicator circles on the fram for each clicked corner
        for pt in ref_pts:
            cv2.circle(display_frame, pt, 6, (0, 0, 255), -1)

        cv2.imshow(WINDOW_CALIBRATION, display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            # Once 4 points are chosen, we compute the perspective transform matrix!
            if len(ref_pts) == 4:
                pts_src = np.array(ref_pts, dtype=np.float32)
                pts_dst = np.array(
                    [
                        [0,0],
                        [float(CANVAS_SIZE), 0],
                        [float(CANVAS_SIZE), float(CANVAS_SIZE)],
                        [0, float(CANVAS_SIZE)],
                    ],
                    dtype=np.float32,
                )

                matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
                print("Calibration successful!")
                break
            else:
                print(f"Selected {len(ref_pts)}. Please choose exactly 4 corners.")

    cv2.destroyWindow(WINDOW_CALIBRATION) 
    
    return matrix