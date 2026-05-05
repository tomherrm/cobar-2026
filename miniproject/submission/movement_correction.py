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

def process_vision_and_steer(omm, retina_tool, darkness_threshold=50, coverage_threshold=0.15, avoidance_gain=5.5):
    """
    Converts raw hex vision, crops it, plots it, and calculates steering based on dark pixel percentage.
    """
    # 1. CONVERT TO 2D IMAGES
    left_img_2d = retina_tool.hex_pxls_to_human_readable(omm[0].max(-1), color_8bit=True)
    right_img_2d = retina_tool.hex_pxls_to_human_readable(omm[1].max(-1), color_8bit=True)
    
    # 2. CROP A SPECIFIC HORIZONTAL BAND
    total_height = left_img_2d.shape[0]
    start_row = total_height // 4  
    end_row = total_height // 3    
    
    left_cropped = left_img_2d[start_row:end_row, 120:]
    right_cropped = right_img_2d[start_row:end_row, :-120]
    
    # 3. CALCULATE BLACK PIXEL PERCENTAGE
    # Create a mask of pixels darker than the threshold. 
    # Taking the .mean() of a boolean array gives the fraction of True values (0.0 to 1.0)!
    left_black_fraction = (left_cropped < darkness_threshold).mean()
    right_black_fraction = (right_cropped < darkness_threshold).mean()
    
    # 4. STEERING LOGIC
    

    control_signal = np.ones(2)
    
    # Trigger if either eye has MORE black pixels than the allowed coverage threshold
    if left_black_fraction > coverage_threshold or right_black_fraction > coverage_threshold:
        raw_steer = avoidance_gain

        
        # Steer away from the side that has the HIGHEST percentage of black pixels
        if left_black_fraction > right_black_fraction:
            control_signal = np.array([3.0, 0.5]) # Object on left, steer right
        else:
            control_signal = np.array([0.5, 3.0]) # Object on right, steer left
            
    
    
    # Return the fractions instead of raw intensities so you can print/debug them!
    return control_signal, left_cropped, right_cropped, left_black_fraction, right_black_fraction
