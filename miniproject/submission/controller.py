import numpy as np
from miniproject.simulation import MiniprojectSimulation
import odor_attraction
import tilt_correction



class Controller:
    def __init__(self, sim: MiniprojectSimulation):
        # you may also implement your own turning controller
        from flygym.examples.locomotion import TurningController

        self.turning_controller = TurningController(sim.timestep)

    def drive_logic(self, olfaction, quat):
        olfaction_drives = odor_attraction.odor_intensity_to_control_signal(olfaction)

        tilt_compensation = tilt_correction.tilt_to_control_signal(quat, olfaction, 0.02, 0.02)

        control_signal = olfaction_drives + tilt_compensation

        return control_signal


    
    def step(self, sim: MiniprojectSimulation):
        # Observations 
        olfaction = sim.get_olfaction(sim.fly.name)
        quat = sim.get_body_rotations

        # Output control logic to drives
        drives = self.drive_logic(olfaction, quat)
        joint_angles, adhesion = self.turning_controller.step(drives)
        return joint_angles, adhesion
