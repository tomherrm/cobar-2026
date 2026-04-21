import numpy as np
from miniproject.simulation import MiniprojectSimulation
from . import odor_attraction
from . import movement_correction



class Controller:
    def __init__(self, sim: MiniprojectSimulation):
        # you may also implement your own turning controller
        from flygym.examples.locomotion import TurningController
        self.turning_controller = TurningController(sim.timestep)
        
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
        tilt_compensation = movement_correction.tilt_to_control_signal(quat, 
                                                                   self.K_PITCH, 
                                                                   self.K_ROLL, 
                                                                   self.max_pitch_boost, 
                                                                   self.max_roll_boost
                                                                   )
        avoidance_compensation = movement_correction.obstacle_avoidance_control_signal(omm, self.avoidance_gain, self.max_avoidance)

        if np.any(avoidance_compensation):
            control_signal =  avoidance_compensation
        else:
            control_signal = self.speed_gain * (olfaction_drives + olfaction_drives * tilt_compensation)

        return control_signal


    
    def step(self, sim: MiniprojectSimulation):
        # Observations 
        olfaction = sim.get_olfaction(sim.fly.name)
        quat = sim.get_body_rotations(sim.fly.name)[0]
        omm = sim.get_ommatidia_readouts(sim.fly.name)

        # Output control logic to drives
        drives = self.drive_logic(olfaction, quat, omm)
        joint_angles, adhesion = self.turning_controller.step(drives)
        return joint_angles, adhesion
