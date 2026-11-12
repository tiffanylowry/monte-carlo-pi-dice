'''Generate spatial density heatmap for physical throw distribution '''

import cv2
import numpy as np
from .config import CANVAS_SIZE

def render_spatial_heatmap(accumulated_points: list[tuple[float, float]], base_canvas: np.ndarrat) -> np.ndarray:
    '''Generate smoothed 2D Gaussian densiity heatmap from normalized coordinated and blends it over the simulation canvas.

    Args:
        accumulated_points: List of normalzied coordinate tuples (nx, ny) ranging form 0.0 to 1.0
        base_canvas: The base BGR NumpPy ndarray representing the simulation canvas

    Returns
        A NumPy ndarray representing the canvas with blended spatial density heatmap
    '''
    if not accumulated_points:
        return base_canvas

    # Convert normalized coordinates back to pixel coordinates
    px_coords = [n[0] * CANVAS_SIZE for n in accumulated_points]
    py_coords = [n[1] * CANVAS_SIZE for n in accumulated_points]

    # Compute 2D heatmap densisty
    heatmap, _, _ = np.histogram2d(
        py_coords, px_coords,
        bins = 50, # potentially change if CANVAS_SIZE changes to keep smooth image
        range = [[0, CANVAS_SIZE], [0, CANVAS_SIZE]]
    )

    # Apply Gaussian smoothing for smooth density contours
    heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)
    if heatmap.max() > 0:
        heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)

    heatmap = np.uint8(heatmap)

    # Apply OpenCV colormap (JET)
    colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Force colored_heatmap dimensiosn to explicitly match base_canvas dimensiosn
    if colored_heatmap.shape != base_canvas.shape:
        colored_heatmap = cv2.resize(colored_heatmap, (base_canvas.shape[1], base_canvas.shape[0]))

    # Blend heatmap over base canvas
    blended_canvas = cv2.addWeighted(base_canvas, 0.4, colored_heatmap, 0.6, 0)
    return blended_canvas