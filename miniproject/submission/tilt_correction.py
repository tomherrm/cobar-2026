import numpy as np
from miniproject.simulation import MiniprojectSimulation
from scipy.spatial.transform import Rotation

def tilt_to_control_signal(quat, k_pitch=0.05, k_roll=0.02):
    """
    Converts the fly's body tilt into a CPG control signal to maintain stability on hills.
    
    Parameters:
    - pitch (float): Nose up/down tilt in degrees (positive = climbing).
    - roll (float): Left/right tilt in degrees (positive = tilting right).
    - base_drive (np.ndarray): The default descending drive on flat ground.
    - k_pitch (float): Proportional gain for climbing compensation.
    - k_roll (float): Proportional gain for lateral balance compensation.
    
    Returns:
    - np.ndarray: Control signal [left_gain, right_gain] for the CPGs.
    """

    # 1. Pitch Compensation (Uphill/Downhill)
    # We only add power if climbing (pitch > 0)
    rot = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
    pitch, roll, yaw = rot.as_euler('xyz', degrees=True)
    pitch_boost = max(0, pitch) * k_pitch
    control_signal += pitch_boost
    
    # 2. Roll Compensation (Uneven lateral terrain)
    # If tilting right (positive roll), the right legs need to push harder to level out
    # If tilting left (negative roll), the left legs need to push harder
    roll_compensation = np.array([-roll * k_roll, roll * k_roll])

    
    # Ensure drives don't drop below 0 (which would stop the CPG)
    return np.clip(control_signal, a_min=0.1, a_max=None)