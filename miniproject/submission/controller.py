import numpy as np
from miniproject.simulation import MiniprojectSimulation

<<<<<<< Updated upstream
def odor_intensity_to_control_signal(
    odor_intensities,
    attractive_gain=-500
):
    """Convert odor sensor readings to a turning control signal.

    Parameters
    ----------
    odor_intensities : np.ndarray
        Odor intensities from the four sensors, shape ``(4, n_odor_dims)``.
    attractive_gain : float
        Gain applied to the attractive odor dimension.
    

    Returns
    -------
    np.ndarray
        Control signal of shape ``(2,)`` for left and right descending drive.
    """
    attractive_intensities = np.average(
        odor_intensities[:, 0].reshape(2, 2), axis=0, weights=[9, 1]
    )
    
    attractive_bias = (
        attractive_gain
        * (attractive_intensities[0] - attractive_intensities[1])
        / attractive_intensities.mean()
        if attractive_intensities.mean() != 0
        else 0
    )
    aversive_bias = 0
    
    effective_bias = aversive_bias + attractive_bias
    effective_bias_norm = np.tanh(effective_bias**2) * np.sign(effective_bias)
    assert np.sign(effective_bias_norm) == np.sign(effective_bias)

    control_signal = np.ones(2)
    side_to_modulate = int(effective_bias_norm > 0)
    modulation_amount = np.abs(effective_bias_norm) * 0.8
    control_signal[side_to_modulate] -= modulation_amount

    return control_signal

def pitch_correction( rot):

    sin_pitch = np.clip(rot[2, 0], -1.0, 1.0)

    if sin_pitch > 0.10:        
        boost = np.clip(sin_pitch * 1.8, 0.0, 0.5)
        return np.array([boost, boost])
    elif sin_pitch < -0.10:      
        brake = np.clip(-sin_pitch * 0.6, 0.0, 0.25)
        return np.array([-brake, -brake])
    return np.zeros(2)

def roll_correction( rot):
    
    sin_roll = np.clip(rot[2, 1], -1.0, 1.0)
    correction = np.zeros(2)

    if abs(sin_roll) > 0.08:
        side = 0 if sin_roll > 0 else 1  # côté bas
        correction[side] = -np.clip(abs(sin_roll) * 0.4, 0.0, 0.35)

    return correction
=======
def red_ratio(img):
        r = img[:, :, 0].astype(float)
        g = img[:, :, 1].astype(float)
        b = img[:, :, 2].astype(float)
        red_mask = (r > 150) & (r > 2* g) & (r > 2 * b)
        return red_mask.mean() 
>>>>>>> Stashed changes

def detect_dragonfly(sim):
    """
    red detection with 'get_raw_vision' RGB
    """
    frames = sim.get_raw_vision(sim.fly.name) 
    
    left_red  = red_ratio(frames[0])
    right_red = red_ratio(frames[1])

    """ print(left_red)
    print(right_red) """

    THRESHOLD = 1e-3

    if left_red<THRESHOLD and  right_red<THRESHOLD:
        return False, 0 #no danger
    if left_red>THRESHOLD and right_red>THRESHOLD:
        print("danger front")
        return True, 0  #coming from front
    elif left_red>THRESHOLD:
        print("danger left")
        return True, -1  #from left
    else:
        print("danger right")
        return True, +1  #from right

def dragonfly_escape(side):
    if side == -1:    
        return np.array([3.0, 0.5])
    elif side == +1: 
        return np.array([0.5, 3.0])
    else:             
        return np.array([3.0, 3.0])
        

class Controller:
    def __init__(self, sim: MiniprojectSimulation):
        # you may also implement your own turning controller
        from flygym.examples.locomotion import TurningController

        self.turning_controller = TurningController(sim.timestep)
<<<<<<< Updated upstream
=======
        
        self.speed_gain = 1.5
        self.attractive_gain = 500
        self.K_PITCH = 0.1
        self.K_ROLL = 0.05
        self.max_pitch_boost = 0.5
        self.max_roll_boost = 0.3
        self.avoidance_gain = 500
        self.max_avoidance = 0.3
        self.avoidance_penalty = 0.5
    

    def drive_logic(self, olfaction, quat, omm):
        olfaction_drives = odor_attraction.odor_intensity_to_control_signal(olfaction, -self.attractive_gain)
        """ tilt_compensation = movement_correction.tilt_to_control_signal(quat, 
                                                                   self.K_PITCH, 
                                                                   self.K_ROLL, 
                                                                   self.max_pitch_boost, 
                                                                   self.max_roll_boost
                                                                   ) """
        """ avoidance_compensation = movement_correction.obstacle_avoidance_control_signal(omm, self.avoidance_gain, self.max_avoidance) """

        """ if np.any(avoidance_compensation):
            control_signal =  avoidance_compensation
        else: """
        
        control_signal = self.speed_gain * (olfaction_drives)

        return control_signal


>>>>>>> Stashed changes
    
    def step(self, sim: MiniprojectSimulation):
        # implement your control algorithm here
        olfaction = sim.get_olfaction(sim.fly.name)
<<<<<<< Updated upstream
        rot=sim.get_body_rotations(sim.fly.name)
        omm = sim.get_ommatidia_readouts(sim.fly.name)
        
        left_intensity  = omm[0].mean()
        right_intensity = omm[1].mean()

        
        diff = left_intensity - right_intensity  
        


        

        # get other observations as needed
        drives = odor_intensity_to_control_signal(olfaction) # replace with your control logic
        
        drives += pitch_correction(rot)
        drives += roll_correction(rot)

        drives[0] -= np.clip( diff * 1.5, 0.0, 0.5)  
        drives[1] -= np.clip(-diff * 1.5, 0.0, 0.5)  

        
        drives = np.clip(drives, 0.0, 1.0)
        
        joint_angles, adhesion = self.turning_controller.step(drives)
        return joint_angles, adhesion
    
 
=======
        #quat = sim.get_body_rotations(sim.fly.name)[0]
        #omm = sim.get_ommatidia_readouts(sim.fly.name)

        # Output control logic to drives
        danger,side=detect_dragonfly(sim)
        if danger :
            drives=dragonfly_escape(side)
        else :
            drives = odor_attraction.odor_intensity_to_control_signal(olfaction, -self.attractive_gain)
 
        joint_angles, adhesion = self.turning_controller.step(drives)
        return joint_angles, adhesion 




""" WORKED WITHOUT IT BUT WE MIGHT ADD IT AFTER

def get_wind_direction(sim):
    #Find wind direction thanks to the antennas

    antenna = sim.get_antenna_data(sim.fly.name)
    left_force  = antenna["l"]["qfrc_passive"]
    right_force = antenna["r"]["qfrc_passive"]
    
    # Difference between left and right
    wind_lateral = np.linalg.norm(left_force) - np.linalg.norm(right_force)
    return wind_lateral  


def drive_logic(olfaction, quat, omm, sim,wind_gain=500,attractive_gain=500):
    olfaction_drives = odor_attraction.odor_intensity_to_control_signal(
        olfaction, -attractive_gain
    )

    wind_lateral = get_wind_direction(sim)
    wind_correction = np.array([wind_lateral, -wind_lateral]) * wind_gain

    
    control_signal = olfaction_drives + wind_correction 

    return np.clip(control_signal, 0.0, 1.0)  """


    
>>>>>>> Stashed changes
