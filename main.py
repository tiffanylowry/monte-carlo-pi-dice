'''Main script coordinating webcam capture, perspective transformation, user controls, and real time Monte Carlo Loop'''

import cv2
import time
from monte_carlo.calibration import get_perspective_matrix
from monte_carlo.config import (
    CANVAS_SIZE,
    WINDOW_SIMULATION,
    WINDOW_WARPED,
)
from monte_carlo.detector import detect_dice
from monte_carlo.simulation import MonteCarloSimulation

def main() -> None:
    '''Initialize camera, run calibratoin, execute control loop
    
    Returns:
        None
    '''
    # Initialize connection to primary webcam device
    cap = cv2.VideoCapture(0) # 0 for tsting, 1 for external web cam

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Run perspective calibration
    matrix = get_perspective_matrix(cap)
    if matrix is None:
        print("Error: No matrix recieved")
        cap.release()
        cv2.destroyAllWindows()
        return

        
    # Initialize state manager

    sim = MonteCarloSimulation()

    print("\nSIMULATION STARTED")
    print("1. Throw dice on grid")
    print("2. Press SPACEBAR to capture the position of the dice")
    print("3. Pick up dice for next round")
    print("4. Press 'r' to reset, or 'q' to quit")

    # Main event loop
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame")
            break

        # Apply perspective warp to make camera fram into a flat grid
        warped = cv2.warpPerspective(frame, matrix, (CANVAS_SIZE, CANVAS_SIZE))

        # Render simulation state canvas
        canvas = sim.render_canvas()

        # Display feeds
        cv2.imshow(WINDOW_WARPED, warped)
        cv2.imshow(WINDOW_SIMULATION, canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            # Capture time: extract dice centers
            round_points = detect_dice(warped)
            sim.add_points(round_points)
            print(f"Captured dice! Added {len(round_points)}. Total Accumulation: {len(sim.accumulated_points)}")
        elif key == ord("r"):
            # Reset sim history
            sim.reset()
            print("Simulation reset :(. All points cleared.")

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
  main()