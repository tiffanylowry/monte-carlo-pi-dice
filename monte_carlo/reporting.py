'''Compile experiemental session metrix and generate Markdown performance reports'''

import math
from .statistics import (
    compute_confidence_interval_95,
    evaluate_spatial_uniformity,
    is_spatially_uniform,
)

def generate_session_markdown(
        pi_history: list[float],
        error_history: list[float],
        se_history: list[float],
        round_timestamps: list[float],
        total_counts: int,
        inside_counts: int,
        accumulated_points: list[tuple[float, float]] | None = None,
        uniformity: tuple[float, float] | None = None,
) -> str:
    '''Compiles comprehensive Markdown session report
    
    Args: 
        pi_history: List of historical Pi estimates per round
        error_history: List of historical absolute error values per round
        se_history: List of historical Standard Error values per round
        round_timestamps: List of UNIX timestamps for each capture event
        total_counts: Cumulative total number of points thrown
        inside_counts: Cumulative number of points landing inside the circle
        accumulated_points: (Optional) list of normalized (nx, ny) point tuples
        uniformity: tuple[float, float] | None = None,
        
    Returns:
        A string containing the formatted Markdown performance report.

    '''
    duration = (round_timestamps[-1] - round_timestamps[0]) if len(round_timestamps) > 1 else 0.0
    final_pi = pi_history[-1] if pi_history else 0.0
    final_error = pi_history[-1] if error_history else 0.0
    final_se = se_history[-1] if se_history else 0.0
    outside_circle = total_counts - inside_counts

    ci_low, ci_high = compute_confidence_interval_95(final_pi, final_se)
    ci_margin = 1.96 * final_se

    resolved_uniformity = (
            uniformity
            if uniformity is not None
            else evaluate_spatial_uniformity(accumulated_points if accumulated_points is not None else [])
    )
    if resolved_uniformity is None:
        uniformity_section = (
            "## Spatial Uniformity (Chi-Square Test)\n"
            "- **Status:** Insufficient sample size for spatial grid uniformity test.\n"
        )
    else:
        chi2_stat, p_value = resolved_uniformity
        # [UPDATED: use is_spatially_uniform from statistics.py for threshold assessment]
        if is_spatially_uniform(p_value):
            assessment = (
                "Consistent with spatial uniformity across grid bins (p >= 0.05); "
                "no significant clustering bias detected."
            )
        else:
            assessment = (
                "Significant spatial non-uniformity detected (p < 0.05); "
                "check for throw clustering, calibration drift, or lighting bias."
            )

        uniformity_section = (
            "## Spatial Uniformity (Chi-Square Test)\n"
            f"- **Chi-Square Statistic ($\\chi^2$):** `{chi2_stat:.4f}`\n"
            f"- **$p$-value:** `{p_value:.4f}`\n"
            f"- **Assessment:** {assessment}\n"
        )
        
    report = f"""# Monte Carlo Simulation - Session Report

    ## Summary Statistics
    - **Final Estimated $\\pi$:** `{final_pi:.6f}`
    - **True $\\pi$:** `{math.pi:.6f}`
    - **Absolute Error:** `{final_error:.6f}`
    - **95% Confidence Interval SE:** `± {ci_margin:.6f}` (`[{ci_low:.6f}, {ci_high:.6f}]`)
    - **Total Rounds Completed:** `{len(pi_history)}`
    - **Total Sample Points ($N$):** `{total_counts}`
    - Points Inside Circle: `{inside_counts}`
    - Points Outside Circle: `{outside_circle}`
    - **Session Duration:** `{duration:.2f} seconds`

    {uniformity_section}
    """

    return report