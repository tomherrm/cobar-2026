import numpy as np
from miniproject.simulation import MiniprojectSimulation
from . import odor_attraction
from . import movement_correction
from flygym.vision.retina import Retina  
from scipy.spatial.transform import Rotation



class Controller:
    def __init__(self, sim: MiniprojectSimulation):
        # you may also implement your own turning controller
        from flygym.examples.locomotion import TurningController
        self.turning_controller = TurningController(sim.timestep)
        self.retina = Retina()

        self.speed_gain = 2.2
        self.attractive_gain = 500
        self.K_PITCH = 10
        self.K_ROLL = 50
        self.max_pitch_boost = 0.5
        self.max_roll_boost = 0.3
        self.max_pitch = 10
        self.max_roll = 40
        self.avoidance_gain = 500
        self.max_avoidance = 0.3
        self.avoidance_penalty = 0.5
        self.avoidance_threshold = 100
        self.tilt_gain = 0.3
    

    def drive_logic(self, olfaction, quat, omm):
        #Sensory drives
        odor_steer = odor_attraction.odor_intensity_to_control_signal(olfaction
                                                                      , -self.attractive_gain
                                                                )
        avoid_steer, _, _, leftomm, rightomm = movement_correction.process_vision_and_steer(omm, self.retina)
        roll_compensation, pitch_compensation, pitch, roll = movement_correction.tilt_to_control_signal(quat, 
                                                                self.K_PITCH, 
                                                                self.K_ROLL, 
                                                                self.max_pitch_boost, 
                                                                self.max_roll_boost
                                                                )

        # ---------------------------------------------------------
        # HIERARCHICAL DECISION TREE 
        # ---------------------------------------------------------
        
        intended_movement = odor_steer

        if leftomm < self.avoidance_threshold or rightomm < self.avoidance_threshold:
            intended_movement = avoid_steer

        if roll > self.max_roll :
            intended_movement += roll_compensation 
        if pitch > self.max_pitch :
            intended_movement += pitch_compensation * self.tilt_gain
        
        final_drive = intended_movement * self.speed_gain  


        return np.clip(final_drive, a_min=-self.speed_gain, a_max=self.speed_gain)


    
    def step(self, sim: MiniprojectSimulation):
        # Observations 
        olfaction = sim.get_olfaction(sim.fly.name)
        quat = sim.get_body_rotations(sim.fly.name)[0]
        omm = sim.get_ommatidia_readouts(sim.fly.name)

        # Output control logic to drives
        drives = self.drive_logic(olfaction, quat, omm)
        avoid_steer, _, _, leftomm, rightomm = movement_correction.process_vision_and_steer(omm, self.retina)
        print(f"Drives: {drives}, Olfaction: {olfaction}, OMMleft: {leftomm:.2f}, OMMright: {rightomm:.2f}")
        joint_angles, adhesion = self.turning_controller.step(drives)

        return joint_angles, adhesion
