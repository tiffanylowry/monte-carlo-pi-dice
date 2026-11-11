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
        self.accumulated_points: list[tuple[float, float]] = []
        self.total_counts = 0
        self.inside_counts = 0

        # History trackers
        self.pi_history: list[float] = []
        self.error_history: list[float] = []
        self.se_history: list[float] = []
        self.round_timestamps: list[float] = []

    def add_points(self, points: list[tuple[int, int]]) -> None:
        ''' Appends new points from round, convert points to normalized coordinates and add to the persistent collection, evaluate boundary conditions, and trigger statistics/history metrics
        Args:
            points: List of (x,y) coordinate tuples to add to simulation history.

        Returns:
            None
        '''
        if not points:
            return

        for px, py in points:
            # Normalize pixel coordinates to [0.0, 1.0] domain
            nx = px / CANVAS_SIZE
            ny = py / CANVAS_SIZE

            # Append normalized pixel coordinates
            self.accumulated_points.append((nx, ny))
            self.total_counts += 1

            # Normalized square distance from center
            dist_sq = (nx - 0.5) ** 2 + (ny - 0.5) ** 2

            if dist_sq <= 0.5**2:
                self.inside_counts += 1

        # Trigger calculations and history logging
        self._update_history()

    def _update_history(self) -> None:
        '''Computes current round Pi estimate, absolute error, and running standard error and appends to history logs.
        
        Returns:
            None
        '''
        if (self.total_counts > 0) :
            p_hat = self.inside_counts / self.total_counts
            
            # THE POINT! Calculate Monte Carlo estimate of Pi
            # Area Circle = pi*r^2
            # Area Square = (2r)^2 = 4r^2
            # AC/AS = pi/4
            # pi = 4 * AC/AS
            current_pi = (4.0 * p_hat)

            # Standard Error for Monte Carlo Bernoulli Trial
            # SE(p_hat) = sqrt(p_hat * (1- p_hat) / N)
            # pi_hat = 4 * p_hat therefore, SE(pi_hat) = 4 * SE(p_hat)
            se_p = math.sqrt((p_hat * (1.0 - p_hat)) / self.total_counts)
            current_se = 4.0 * se_p
        else:
            current_pi = 0.0
            current_se = 0.0
        
        current_error = abs(current_pi - math.pi)

        self.pi_history.append(current_pi)
        self.error_history.append(current_error)
        self.se_history.append(current_se)
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
        self.se_history()
        self.round_timestamps.clear()

    def render_canvas(self) -> np.ndarray:
        '''Generate visual canvas window, draw the boundary, target circle, accumulated points (converted back to pi)
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

        # Evaluate each point's position  by converting normalzied floats back to pixels
        for nx, ny in self.accumulated_points:
            px = int(nx * CANVAS_SIZE)
            py = int(ny * CANVAS_SIZE)

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
        current_se = self.se_history[-1] if self.se_history else 0.0
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
            f"Std Err: {current_se:.4f} | Rounds: {round_count}",
            (20,80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            canvas,
            f"Total Points: {self.total_counts}",
            (20,120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

        return canvas
