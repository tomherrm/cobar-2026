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

        self.speed_gain = 1.2
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
        self.avoidance_threshold = 0.08
        self.tilt_gain = 0.3
        self._obs_gain=1.0
        
    

    def drive_logic(self, olfaction, quat, omm):
        #Sensory drives
        odor_steer = odor_attraction.odor_intensity_to_control_signal(olfaction
                                                                      , -self.attractive_gain
                                                                )
        #avoid_steer, _, _, leftomm, rightomm = movement_correction.process_vision_and_steer(omm, self.retina)
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

        #print(f"leftOMM : {leftomm}")
        #print(f"rightOMM : {rightomm}")

        """ if leftomm > self.avoidance_threshold or rightomm > self.avoidance_threshold:
            print(f"⚠️ OBSTACLE DÉTECTÉ — left: {leftomm:.3f}, right: {rightomm:.3f}")
            intended_movement = np.array([15.0,0.0])
 """
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

        obstacle_l,obstacle_r,_=self._get_raw_vision_obstacles(sim)
        turn, avoid = self._avoid_obstacles(obstacle_l,obstacle_r)

        if avoid :
            if obstacle_l > obstacle_r:
                drives = np.array([3.0, -3.0])  # virer droite
            else:
                drives = np.array([-3.0, 3.0])  # virer gauche
            #print(drives)
            joint_angles, adhesion = self.turning_controller.step(drives)

            return joint_angles, adhesion

        # Output control logic to drives
        drives = self.drive_logic(olfaction, quat, omm)
        """ avoid_steer, _, _, leftomm, rightomm = movement_correction.process_vision_and_steer(omm, self.retina)
        print(f"Drives: {drives}, Olfaction: {olfaction}, OMMleft: {leftomm:.2f}, OMMright: {rightomm:.2f}") """
        joint_angles, adhesion = self.turning_controller.step(drives)

        return joint_angles, adhesion


    def _get_raw_vision_obstacles(self, sim):
            images = sim.get_raw_vision(sim.fly.name)
            H = images[0].shape[0]

            def green_ratio(img):
                top_strip = img[:H // 4]   # top 25% — only close stalks reach this high
                r = top_strip[:, :, 0].astype(float)
                g = top_strip[:, :, 1].astype(float)
                b = top_strip[:, :, 2].astype(float)
                total = r + g + b + 1e-6
                return float(((g / total > 0.4) & (total > 30)).mean())

            obstacle_left  = green_ratio(images[0])
            obstacle_right = green_ratio(images[1])
            #print(f"Raw vision green ratio: L={obstacle_left:.3f} R={obstacle_right:.3f}")
            total_obstacle = (obstacle_left + obstacle_right) / 2.0
            return obstacle_left, obstacle_right, total_obstacle
    
    def _avoid_obstacles(self, obstacle_left: float, obstacle_right: float):

        diff = obstacle_left - obstacle_right

        if obstacle_left > self.avoidance_threshold or obstacle_right > self.avoidance_threshold:
            turn = -np.sign(diff if abs(diff) > 0.01 else 1.0) * 2.5
            #print(f"REFLEX! L={obstacle_left:.3f} R={obstacle_right:.3f} diff={diff:+.3f} turn={turn:+.2f}")
            return turn, True

        turn = -self._obs_gain * np.tanh(5.0 * diff)
        return turn, False