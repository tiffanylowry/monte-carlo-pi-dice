# Monte Carlo Pi Estimation with Dice and Computer Vision

Estimate the value of π by throwing dice onto a marked 1 m × 1 m grid, tracking where they land with a webcam, and running a Monte Carlo simulation on the results in real time.

> **Part of the Texas A&M DEEP program.**
> This project was created as part of the [Texas A&M University DEEP program](https://artsci.tamu.edu/physics-astronomy/outreach/deep/index.html)

---

## Table of Contents

- [How It Works](#how-it-works)
- [The Math](#the-math)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Controls](#controls)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Tuning Detection](#tuning-detection)
- [Troubleshooting](#troubleshooting)
- [Known Limitations](#known-limitations)
- [Roadmap](#roadmap)
- [Changelog](#changelog)
- [Acknowledgments](#acknowledgments)

---

## How It Works

1. **Calibrate.** Point a webcam at your grid and click its four corners. The program computes a perspective transform that turns the angled camera view into a flat, top-down square.
2. **Throw.** Toss dice onto the grid.
3. **Capture.** Press the spacebar. The program finds each die's center in the warped view and adds those points to the simulation.
4. **Repeat.** Pick up the dice and throw again. Points accumulate across rounds, so the estimate improves as the sample size grows.


## The Math

Draw a circle of radius `r` inside a square of side `2r`:

- Area of circle = πr²
- Area of square = (2r)² = 4r²
- Ratio = π / 4

If dice land uniformly at random, the fraction landing inside the circle approaches that ratio, so:

```
π ≈ 4 × (points inside circle / total points)
```

A point `(x, y)` is inside the circle if its normalized distance from the center is at most 0.5:

```
(x/W - 0.5)² + (y/H - 0.5)² ≤ 0.5²
```

## Requirements

- Python 3.10 or newer (the code uses `X | None` type hints)
- A webcam
- A flat, light-colored surface with a marked 1 m × 1 m square, and dice that contrast with it
- Python packages (see `requirements.txt`):
  - `opencv-python >= 4.8.0`
  - `numpy >= 1.24.0`

## Installation

From the project folder, in PowerShell:

```powershell
# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

## Usage

Run from the project root (the folder that contains `main.py` and the `monte_carlo/` package):

```bash
python main.py
```

**Calibration step**

1. Position the webcam so the whole grid is visible.
2. Click the four corners in this order: **top-left, top-right, bottom-right, bottom-left**.
3. Press `c` to confirm, or `q` to quit.

**Simulation step**

Three windows are shown: the raw calibration feed (closed after calibration), the warped top-down grid, and the simulation canvas with the live π estimate.

## Controls

| Key | Context | Action |
|-----|---------|--------|
| Left click | Calibration | Select a grid corner |
| `c` | Calibration | Confirm the four corners |
| `q` | Calibration / Simulation | Quit |
| `Space` | Simulation | Capture the current dice positions |
| `r` | Simulation | Reset all accumulated points |

## Project Structure

```
.
├── main.py                  # Entry point: camera, calibration, main loop
├── requirements.txt         # Python dependencies
├── README.md
└── monte_carlo/
    ├── __init__.py          # Package marker
    ├── config.py            # Constants: canvas size, window titles
    ├── calibration.py       # Click-to-calibrate perspective transform
    ├── detector.py          # Dice detection (threshold + contours)
    └── simulation.py        # Point storage, rendering, π estimate
```

| Module | Responsibility |
|--------|----------------|
| `main.py` | Opens the webcam, runs calibration, warps each frame, handles key presses |
| `calibration.py` | Collects four mouse clicks and builds a 3×3 perspective matrix |
| `detector.py` | Converts a warped frame to grayscale, blurs, thresholds, and returns contour centers |
| `simulation.py` | Holds `MonteCarloSimulation`, which stores points and draws the canvas and statistics |
| `config.py` | Single place for shared constants |

## Configuration

Shared constants live in `monte_carlo/config.py`:

| Constant | Default | Meaning |
|----------|---------|---------|
| `CANVAS_SIZE` | `800` | Side length in pixels of the warped view and simulation canvas |
| `GRID_RADIUS` | `CANVAS_SIZE / 2` | Radius of the inscribed circle in pixels |
| `WINDOW_CALIBRATION` | `"Webcam Feed - Calibration"` | Calibration window title |
| `WINDOW_WARPED` | `"Warped Grid View"` | Warped view window title |
| `WINDOW_SIMULATION` | `"Monte Carlo Dice Pi Simulation"` | Simulation window title |

The webcam index is set in `main.py`: `cv2.VideoCapture(0)`. Use `0` for a built-in camera and `1` (or higher) for an external one.

## Tuning Detection

Detection settings are currently hardcoded in `monte_carlo/detector.py`. Adjust them if dice are missed or noise is picked up:

| Setting | Current value | Effect |
|---------|---------------|--------|
| Blur kernel | `(7, 7)` | Larger values smooth more noise but blur small features |
| Threshold | `140` | Pixels darker than this become foreground. Raise it for lighter dice, lower it for stricter detection |
| Area filter | `40 < area < 4000` | Contour size range in pixels (measured in the warped 800×800 view) |

Tip: check the **Warped Grid View** window first. If the dice don't look clean and distinct there, detection will struggle.

## Troubleshooting

| Problem | Likely cause and fix |
|---------|----------------------|
| `Error: Could not open webcam.` | Wrong camera index, or another app is using the camera. Try `VideoCapture(1)` or close other apps |
| `Error: No matrix recieved` | Calibration was cancelled with `q` or failed before four corners were confirmed |
| Windows flash and close immediately | Camera cleanup code is inside the main loop. It should run once after the loop ends |
| `ModuleNotFoundError: monte_carlo` | Run `python main.py` from the project root, and make sure the package folder is named `monte_carlo` |
| No dice detected | Adjust the threshold or area filter, improve lighting, or use dice that contrast with the surface |
| Too many false detections | Reduce shadows, tighten the area filter, or raise the minimum area |

## Known Limitations

- Dice that touch or overlap are detected as a single object.
- Shadows and uneven lighting can create false detections.
- Detection assumes dark objects on a light background.
- Dice are thrown by hand, so landing positions may not be perfectly uniform. This can bias the estimate.
- Calibration must be redone every time the program starts.

## Changelog

### v0.1.0
- Initial version: click-based perspective calibration, contour-based dice detection, live Monte Carlo π estimate

## Acknowledgments

- Built as part of the [Texas A&M University DEEP program](https://artsci.tamu.edu/physics-astronomy/outreach/deep/index.html), Department of Physics & Astronomy outreach.
- Built with [OpenCV](https://opencv.org/) and [NumPy](https://numpy.org/).

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.