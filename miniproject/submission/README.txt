============================================================================
  BIOENG-456 — Controlling Behavior in Animals and Robots — Miniproject
  Hierarchical Control of Navigation, Obstacle Avoidance,
  and Postural Stability in a Simulated Drosophila
----------------------------------------------------------------------------
  Authors : Tom Herrmann, Alexandros Dellios, Flavio Caroli
  Date    : 22 May 2026
============================================================================


1. REQUIREMENTS
----------------------------------------------------------------------------
Python 3.10 or higher.

Required packages:
    numpy
    scipy
    matplotlib
    opencv-python
    pygame
    tqdm
    flygym
    miniproject

Install with:
    pip install -r requirements.txt



2. FILE STRUCTURE
----------------------------------------------------------------------------
    submission/
        controller.py            -- Main hierarchical controller
    run_simulation.py            -- Headless evaluation script (official)
    run_interactive.py           -- Interactive run with rendering window
    preview_controller.py

3. RUNNING THE SIMULATION
----------------------------------------------------------------------------

    uv run preview_controller.py --level <level> --seed <seed>

    