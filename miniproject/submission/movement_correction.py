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
    pitch_boost_raw = max(0, pitch) * k_pitch #max(0, pitch) ensures we only boost when climbing #smoothly saturate the boost to avoid excessive values on steep slopes
    pitch_boost = np.array([pitch_boost_raw, pitch_boost_raw]) # Apply the same boost to both legs for climbing

    # Roll Compensation (Uneven lateral terrain)
    roll_boost_raw = roll * k_roll 
    if roll_boost_raw > 0:
        roll_boost = np.array([-roll_boost_raw, roll_boost_raw]) # Left leg boost when roll is positive (tilting right)
    else:
        roll_boost = np.array([roll_boost_raw, -roll_boost_raw]) # right leg boost when roll is negative (tilting left)

    roll_boost = np.tanh(roll_boost)
    pitch_boost = np.tanh(pitch_boost)

    # Ensure drives don't drop below 0
    return roll_boost, pitch_boost, pitch, roll
import numpy as np

import numpy as np
import matplotlib.pyplot as plt
from flygym.vision.retina import Retina

# Instantiate this once at the start of your script
retina = Retina()

def process_vision_and_steer(omm, retina_tool, avoidance_threshold=50, avoidance_gain=5.5):
    """
    Converts raw hex vision, crops it, plots it, and calculates steering.
    """
    # 1. CONVERT TO 2D IMAGES
    # Convert both eyes to 2D human-readable arrays (using .max(-1) if you have spectral channels)
    left_img_2d = retina_tool.hex_pxls_to_human_readable(omm[0].max(-1), color_8bit=True)
    right_img_2d = retina_tool.hex_pxls_to_human_readable(omm[1].max(-1), color_8bit=True)
    
    # 2. CROP THE TOP HALF
    crop_height = left_img_2d.shape[0] // 3
    left_cropped = left_img_2d[:crop_height, :]
    right_cropped = right_img_2d[:crop_height, :]
    
    # 3. CALCULATE INTENSITIES (Only on the cropped top half!)
    left_intensity = left_cropped.mean()
    right_intensity = right_cropped.mean()
    
    # 4. STEERING LOGIC
    avoidance_compensation = np.zeros(2)
    if left_intensity > avoidance_threshold or right_intensity > avoidance_threshold:
        raw_steer = (left_intensity - right_intensity) * avoidance_gain
        if raw_steer > 0:
            avoidance_compensation = np.array([raw_steer, -raw_steer])
        else:
            avoidance_compensation = np.array([raw_steer, abs(raw_steer)])
            
    control_signal = np.clip(np.tanh(avoidance_compensation), a_min=-1.0, a_max=1.0)
    
    # Return the images too so we can plot them!
    return control_signal, left_cropped, right_cropped

