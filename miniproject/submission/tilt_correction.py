import numpy as np
from miniproject.simulation import MiniprojectSimulation
from scipy.spatial.transform import Rotation

def tilt_to_control_signal(quat, k_pitch=0.02, k_roll=0.01, max_pitch_boost=0.2, max_roll_boost=0.1):
    """
    Converts the fly's body tilt into a CPG control signal gain to maintain stability on hills.
    
    Parameters:
    - quat (np.ndarray): Quaternion representing the fly's body orientation [w, x, y, z].
    - k_pitch (float): Proportional gain for climbing compensation.
    - k_roll (float): Proportional gain for lateral balance compensation.
    - max_pitch_boost (float): Maximum boost applied for pitch compensation to prevent excessive values on steep slopes.
    - max_roll_boost (float): Maximum boost applied for roll compensation to prevent overcomp
    
    Returns:
    - np.ndarray: Control signal [left_gain, right_gain] for the CPGs.
    """
    # Convert quaternion to Euler angles
    rot = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
    pitch, roll, yaw = rot.as_euler('xyz', degrees=True)

    # Pitch Compensation (Uphill/Downhill)
    pitch_boost_raw = max(0, pitch) * k_pitch #max(0, pitch) ensures we only boost when climbing
    pitch_boost_scalar = max_pitch_boost * np.tanh(pitch_boost_raw) #smoothly saturate the boost to avoid excessive values on steep slopes
    pitch_boost = np.array([pitch_boost_scalar, pitch_boost_scalar]) # Apply the same boost to both legs for climbing

    # Roll Compensation (Uneven lateral terrain)
    roll_boost_raw = roll * k_roll 
    roll_boost_scalar = max_roll_boost * np.tanh(roll_boost_raw) 
    if roll_boost_raw > 0:
        roll_boost = np.array([0, roll_boost_scalar]) # Left leg boost when roll is positive (tilting right)
    else:
        roll_boost = np.array([roll_boost_scalar, 0]) # right leg boost when roll is negative (tilting left)

    control_signal = pitch_boost + roll_boost

    # Ensure drives don't drop below 0
    return np.clip(control_signal, a_min=0.1, a_max=None)