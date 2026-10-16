'''Detect dice positions and their center points with image processing and contour analysis'''

import cv2

def detect_dice(warped_frame: cv2.Mat) -> list[tuple[int,int]]:
    '''Processes top-down frame to find dice center with grayscale conversion, gaussian blue, thresholding, and contur analaysis
    
    Args:
        warped_frame: A top-down perspective-warped frame of the grid.

    Returns:
        A list of (x, y) tuples representing the pixel coordination of detected dice centers
    '''

    # Convert frame to grayscale
    gray = cv2.cvtColor(warped_frame, cv2.COLOR_BGR2GRAY)

    # Blur image to reduce noise
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Binary inversion thresholding to isolate dice objects from paper
    _, thresh = cv2.threshold(blurred, 140, 255, cv2.THRESH_BINARY_INV)

    # Find external object contors
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    round_points = []
    for ct in contours:
        area = cv2.contourArea(ct)

        # Filter contours by pixel area bounds to ignore noise and shadows
        if 40 < area < 4000:
            M = cv2.moments(ct)
            if M["m00"] > 0:
                # Calculate image moments to find center of the dice
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                round_points.append((cx, cy))

    return round_points
