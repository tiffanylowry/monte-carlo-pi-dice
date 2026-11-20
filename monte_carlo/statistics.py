"""Statistical and geometric helpers for the Monte Carlo Pi simulation including boundary checks, error/confidence, and spatial uniformity"""

import math
import numpy as np
from scipy.stats import chisquare

def is_inside_inscribed_circle(nx: float, ny: float) -> bool:
    '''Checks whether a normalized point falls inside or outside the inscribed circle
    
    Args:
        nx: Normalized x-coordinate in [0.0, 1.0].
        ny: Normalized y-coordinate in [0.0, 1.0].

    Returns:
        True if points is within radius 0.5 of center (0.5, 0.5), else false.   
    '''
    # Normalized squared distance from grid center (0.5, 0.5)
    return ((nx - 0.5) ** 2 + (ny - 0.5) ** 2) <= (0.5 ** 2)

def compute_pi_estimate(inside_count: int, total_count: int) -> float:
    '''Computers the running Monte Carlo Pi estimate: 4 * (inside/total)
    
    Args:
        inside_count: Number of points falling inside the inscribed circle.
        total_count: Total number of accumulated points.

    Returns:
        Estimated values of Pi, or 0.0 if total_count is zero or negative
    '''
    if total_count <= 0:
        return 0.0

    # THE POINT! Calculate Monte Carlo estimate of Pi
    # Area Circle = pi*r^2
    # Area Square = (2r)^2 = 4r^2
    # AC/AS = pi/4
    # pi = 4 * AC/AS
    return 4.0 * (inside_count/total_count)

def compute_absolute_error(pi_esimate: float) -> float:
    '''Computers absolute error |pi_hat - true_pi|
    
    Args:
        pi_estimate: Current estimated value of pi

    Returns:
        Absolute difference between estimated pi and math.pi
    '''
    return abs(pi_esimate - math.pi) # Note! math.pi is accurate to the 15th decimal point

def compute_standard_error(inside_count: int, total_count: int) -> float:
    '''Computes Bernoulli standard error of the pi estimate:
    SE(pi_hat) = 4 * sqrt(p_hat * (1 - p_hat) / N)
    
    Args:
        inside_count: Number of points falling inside the inscribed circle.
        total_count: Total number of accumulated points.
    
    Returns:
        Standard error of pi estimate or 0.0 if total_count is zero or negative
    '''
    if total_count <= 0:
        return 0.0

    # Sample proportion of points inside the target circle
    p_hat = inside_count / total_count
    return 4.0 * math.sqrt((p_hat * (1.0 - p_hat)) / total_count)

def compute_confidence_interval_95(pi_estimate: float, standard_error: float) -> tuple[float, float]:
    '''Computes the 95% confidence interal bounds: pi_hat +/- 1.96 * SE
    
    Arg:
        pi_estimate: Estimated values of pi
        standard_error: Standard error of the pi_estimate

    Returns:
        Tuple of (lower_bound, upper_bound) for the 95% confidence interval
    '''
    margin = 1.96 * standard_error
    return (pi_estimate - margin, pi_estimate + margin)

def evaluate_spatial_uniformity(points: list[tuple[float, float]], grid_bins: int = 5, min_expected_per_bin: float = 5.0) -> tuple[float, float] | None:
    '''Evaluates wether normalized points in [0.0, 1.0] are uniformly distributed across a grid_bins x grid_bins grid using a
    Chi-Square goodness-of-fit test.

    Args:
        points: List of normalized (nx, ny) coordinates tuples in [0.0, 1.0]
        grid_bins: Number of divisions along each axis of the square grid.
        min_expected_per_bin: Minimum expected count per bin required to run the test

    Returns:
        Tuple of (chi2_stat, p_value) as floats, or None if sample size is insufficent
    '''
    total_points = len(points)
    num_bins = grid_bins ** 2
    expected_per_bin = total_points / num_bins if total_points > 0 else 0.0

    # Guard from unreliable Chi-Square approximations when expected counts are small
    if total_points == 0 or expected_per_bin < min_expected_per_bin:
        return None

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    # Bin normalized coordinates into equal area spatial cells
    counts, _, _ = np.histogram2d(ys, xs, bins=grid_bins, range=[[0.0, 1.0], [0.0, 1.0]])
    observed = counts.ravel()
    expected = np.full(num_bins, expected_per_bin, dtype=float)

    chi2_stat, p_value = chisquare(f_obs=observed, f_exp=expected)
    return float(chi2_stat), float(p_value) 

def is_spatially_uniform(p_value: float, alpha: float = 0.05) -> bool:
    '''Returns True if the Chi-Square p-value indicates spatial uniformity (p >= alpha).

    Args:
        p_value: The p-value returned by evaluate_spatial_uniformity
        alpha: Significance threshold (default 0.05)

    Returns:
        True if p_value >= alpha, else False.
    '''
    return p_value >= alpha