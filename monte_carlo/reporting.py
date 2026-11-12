'''Compile experiemental session metrix and generate Markdown performance reports'''

import math

def generate_session_markdown(
        pi_history: list[float],
        error_history: list[float],
        se_history: list[float],
        round_timestamps: list[float],
        total_counts: int,
        inside_counts: int
) -> str:
    '''Compiles comprehensive Markdown session report
    
    Args: 
        pi_history: List of historical Pi estimates per round
        error_history: List of historical absolute error values per round
        se_history: List of historical Standard Error values per round
        round_timestamps: List of UNIX timestamps for each capture event
        total_counts: Cumulative total number of points thrown
        inside_counts: Cumulative number of points landing inside the circle
    
    Returns:
        A string containing the formatted Markdown performance report.

    '''
    duration = (round_timestamps[-1] - round_timestamps[0]) if len(round_timestamps) > 1 else 0.0
    final_pi = pi_history[-1] if pi_history else 0.0
    final_error = pi_history[-1] if error_history else 0.0
    final_se = se_history[-1] if se_history else 0.0
    outside_circle = total_counts - inside_counts

    report = f"""# Monte Carlo Simulation - Session Report

    ## Summary Statistics
    - **Final Estimated $\\pi$:** `{final_pi:.6f}`
    - **True $\\pi$:** `{math.pi:.6f}`
    - **Absolute Error:** `{final_error:.6f}`
    - **95% Confidence Interval SE:** `± {1.96 * final_se:.6f}`
    - **Total Rounds Completed:** `{len(pi_history)}`
    - **Total Sample Points ($N$):** `{total_counts}`
    - Points Inside Circle: `{inside_counts}`
    - Points Outside Circle: `{outside_circle}`
    - **Session Duration:** `{duration:.2f} seconds`
    """

    return report