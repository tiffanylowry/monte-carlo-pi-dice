'''Manages simulation state, coordinate persistence, and real-time rendering of points and statistical outputs'''

import cv2
import numpy as np
from .config import CANVAS_SIZE, GRID_RADIUS

class MonteCarloSimulation:
    '''Maintain state of oaccumulated throw points and handle rendering the simulation cavas with inscrobed circle and real-time Pi calculations'''

    def __init__(self):
        ''' Initalize Monte Carlo simulation state with an empty point array'''
        # Persisitent storage for all dice coordinates across multiple rounds
        self.accumulated_points = []

    def add_points(self, points: list[tuple[int, int]]) -> None:
        ''' Appends new point to the persistent collection
        Args:
            points: List of (x,y) coordinate tuples to add to simulation history.

        Returns:
            None
        '''

        self.accumulated_points.extend(points)

    def reset(self) -> None:
        '''Clear all accumulated points, resetting the simulation
        
        Returns:
            None
        '''

        self.accumulated_points.clear()

    def render_canvas(self) -> np.ndarray:
        '''Generate visual canvas window, draw athe boundary, target circle, accumulated point 
        colors (blue -> inside, red -> outside), and live PI Statisitics

        Returns:
            A NumPy ndarray representing the rendered canvas image
        '''
        # Create a fresh blank white canvas
        canvas = np.ones((CANVAS_SIZE, CANVAS_SIZE, 3), dtype=np.uint8) * 255

        # Draw outer bounding square and target inscribed circle
        cv2.rectangle(canvas, (0,0), (CANVAS_SIZE, CANVAS_SIZE), (200, 200, 200), 2)
        cv2.circle(
            canvas,
            (int(GRID_RADIUS), int(GRID_RADIUS)),
            int(GRID_RADIUS),
            (0, 0, 0),
            2
        )

        inside_count = 0
        total_count = len(self.accumulated_points)

        # Evaluate each point's position relative to the circle
        for px, py in self.accumulated_points:
            nx = px / CANVAS_SIZE
            ny = py / CANVAS_SIZE

            # Normalized square distance from center
            dist_sq = (nx - 0.5) ** 2 + (ny - 0.5) ** 2

            if dist_sq <= 0.5**2:
                inside_count += 1
                color = (255, 0, 0) # Blue for inside circle
            else:
                color = (0, 0, 255) # Red for outside circle

            cv2.circle(canvas, (int(px), int(py)), 3, color, -1)

        # THE POINT! Calculate Monte Carlo estimate of Pi
        # Area Circle = pi*r^2
        # Area Square = (2r)^2 = 4r^2
        # AC/AS = pi/4
        # pi = 4 * AC/AS
        pi_estimate = (4 * (inside_count / total_count) if total_count > 0 else 0.0 )

        # Render metrix text on canvas
        cv2.putText(
            canvas,
            f"Estimated PI = {pi_estimate:.5f}",
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            canvas,
            f"Total Points: {total_count}",
            (20,80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

        return canvas
