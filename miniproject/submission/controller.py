import numpy as np
from miniproject.simulation import MiniprojectSimulation

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


class Controller:
    def __init__(self, sim: MiniprojectSimulation):
        # you may also implement your own turning controller
        from flygym.examples.locomotion import TurningController

        self.turning_controller = TurningController(sim.timestep)
    
    def step(self, sim: MiniprojectSimulation):
        # implement your control algorithm here
        olfaction = sim.get_olfaction(sim.fly.name)
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
    
 
