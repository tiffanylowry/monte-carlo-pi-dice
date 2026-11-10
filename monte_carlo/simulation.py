'''Manages simulation state, coordinate persistence, and real-time rendering of points and statistical outputs'''

import time
import math
import cv2
import numpy as np
from .config import CANVAS_SIZE, GRID_RADIUS

class MonteCarloSimulation:
    '''Maintain state of oaccumulated throw points and handle rendering the simulation cavas with inscrobed circle and real-time Pi calculations'''

    def __init__(self):
        ''' Initalize Monte Carlo simulation state with an empty point array'''
        # Persisitent storage for all dice coordinates across multiple rounds
        self.accumulated_points = []
        self.total_counts = 0
        self.inside_counts = 0

        # History trackers
        self.pi_history: list[float] = []
        self.error_history: list[float] = []
        self.round_timestamps: list[float] = []

    def add_points(self, points: list[tuple[int, int]]) -> None:
        ''' Appends new points from round to the persistent collection, evaluate boundary conditions, and trigger statistics/history metrics
        Args:
            points: List of (x,y) coordinate tuples to add to simulation history.

        Returns:
            None
        '''
        if not points:
            return

        for px, py in points:
            self.accumulated_points.append((px, py))
            self.total_counts += 1

            nx = px / CANVAS_SIZE
            ny = py / CANVAS_SIZE

            # Normalized square distance from center
            dist_sq = (nx - 0.5) ** 2 + (ny - 0.5) ** 2

            if dist_sq <= 0.5**2:
                self.inside_counts += 1

        # Trigger calculations and history logging
        self._update_history()

    def _update_history(self) -> None:
        '''Computes current round Pi estimate, absolute error, and appends to history logs.
        
        Returns:
            None
        '''

        # THE POINT! Calculate Monte Carlo estimate of Pi
        # Area Circle = pi*r^2
        # Area Square = (2r)^2 = 4r^2
        # AC/AS = pi/4
        # pi = 4 * AC/AS
        current_pi = (4.0 * (self.inside_counts / self.total_counts)) if self.total_counts > 0 else 0.0
        current_error = abs(current_pi - math.pi)

        self.pi_history.append(current_pi)
        self.error_history.append(current_error)
        self.round_timestamps.append(time.time())    

    def reset(self) -> None:
        '''Clear all accumulated points, resetting the simulation
        
        Returns:
            None
        '''

        self.accumulated_points.clear()
        self.total_counts = 0
        self.inside_counts = 0
        self.pi_history.clear()
        self.error_history.clear()
        self.round_timestamps.clear()

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

        # Evaluate each point's position relative to the circle
        for px, py in self.accumulated_points:
            nx = px / CANVAS_SIZE
            ny = py / CANVAS_SIZE

            # Normalized square distance from center
            dist_sq = (nx - 0.5) ** 2 + (ny - 0.5) ** 2

            if dist_sq <= 0.5**2:
                color = (255, 0, 0) # Blue for inside circle
            else:
                color = (0, 0, 255) # Red for outside circle

            cv2.circle(canvas, (int(px), int(py)), 3, color, -1)

        # Fetch latest metrics
        current_pi = self.pi_history[-1] if self.pi_history else 0.0
        current_error = self.error_history[-1] if self.error_history else 0.0
        round_count = len(self.pi_history)


        # Render metrix text on canvas
        cv2.putText(
            canvas,
            f"Estimated PI = {current_pi:.5f} | Err: {current_error:.4f}",
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            canvas,
            f"Rounds: {round_count} | Total Points: {self.total_counts}",
            (20,80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

        return canvas
