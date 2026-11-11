'''Generates real-time convergence and error plots using Matplotlib for OpenCV display'''

import math
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg') # Non-interactive backend for headless renfering
import matplotlib.pyplot as plt

def generate_convergence_plot(pi_history: list[float], se_history: list[float]) -> np.ndarray:
    '''Generates a dual subplot image tracking both pi convergence and absolute error over rounds.
    
    Args:
        pi_history: List of historical pi estimates per round
        se_history: List of historical Standard Error values per round

    Returns: 
        A NumPy nd array reperesnting rendered plot image in BGR format for OpenCV.
    '''
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5), dpi=100)
    rounds = list(range(1, len(pi_history) + 1))
    errors = [abs(val - math.pi) for val in pi_history] # Note! math.pi is accurate to the 15th decimal point

    # Compute 95% Confidence Intervals: Est Pi ± (1.96 * SE)
    pi_arr = np.array(pi_history)
    se_arr = np.array(se_history)
    upper_ci = pi_arr + (1.96 * se_arr)
    lower_ci = pi_arr - (1.96 * se_arr)

    # Plot 1: Pi estimate convergence on true pi
    ax1.plot(rounds, pi_history, marker='o', color='b', linestyle='-', linewidth=1.5, markersize=4, label=r'Est. $\pi$')
    ax1.axhline(y=math.pi, color='r', linestyle="--", label=r'True $\pi$ (3.14159...)')
    ax1.set_title(r'$\pi$ Convergence Over Time')

    # Shaded confidence band
    if len(rounds) > 0:
        ax1.fill_between(rounds, lower_ci, upper_ci, color='b', alpha=0.2, label="95% Confidence Interval")
    ax1.set_ylabel(r"Estimated $\pi$")
    ax1.set_ylim(2.0, 4.0)
    ax1.grid(True, linestyle=":", alpha = 0.6)
    ax1.legend(loc='upper right', fontsize=8)

    # Plot 2: Absolute error magnitude
    ax2.plot(rounds, errors, marker='x', color='r', linestyle='-', linewidth=1.5, markersize=4)
    ax2.set_title(r"Absolute Error ($|\hat{\pi} - \pi|$)")
    ax2.set_xlabel("Capture Rounds")
    ax2.set_ylabel("Error Magnitude")
    ax2.grid(True, linestyle=':', alpha=0.6)

    fig.tight_layout()

    # Convert Matplotlib canvas to Numpoy arrary for OpenCV
    fig.canvas.draw()
    plot_img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGBA2BGR)

    plt.close(fig)
    return plot_img