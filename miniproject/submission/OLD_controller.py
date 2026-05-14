import numpy as np
from miniproject.simulation import MiniprojectSimulation
from . import odor_attraction
from . import OLD_movement_correction
from flygym.vision.retina import Retina  
from scipy.spatial.transform import Rotation



class Controller:
    def __init__(self, sim: MiniprojectSimulation):
        # you may also implement your own turning controller
        from flygym.examples.locomotion import TurningController
        self.turning_controller = TurningController(sim.timestep)
        self.retina = Retina()

        self.speed_gain = 2.8 #1.8
        self.attractive_gain = 1000
        self.K_PITCH = 10
        self.K_ROLL = 50
        self.max_pitch_boost = 0.5
        self.max_roll_boost = 0.3
        self.max_pitch = 20#10
        self.max_roll = 40
        self.avoidance_gain = 500
        self.max_avoidance = 0.3
        self.avoidance_penalty = 0.5
        self.avoidance_threshold = 0.35#0.085#0.04#0.01
        self.tilt_gain = 0.3
        self._obs_gain=1.0
        self._avoid_counter = 0
        self._last_drives = np.ones(2)
        self.counter=0
        self.obstacle_r=0
        self.obstacle_l=0
        self._aligned = False # we want the fly to align itself at first
        self._current_crop_row=0
        
        
    

    def drive_logic(self, olfaction, quat):
        #Sensory drives
        odor_steer, bias = odor_attraction.odor_intensity_to_control_signal(olfaction
                                                                      , -self.attractive_gain
                                                                )
        
        if not self._aligned: #initial alignement
            if abs(bias) < 0.1:  
                self._aligned = True
                print("The fly is aligned with goal")
            else:
                print(f"ALIGNEMENT")
                # Tourner sur place sans avancer
                if bias > 0:
                    return np.array([2.0, 0.0])  # tourner à droite
                else:
                    return np.array([0.0, 2.0])  # tourner à gauche
                
        

       
        roll_compensation, pitch_compensation, pitch, roll = OLD_movement_correction.tilt_to_control_signal(quat, 
                                                                self.K_PITCH, 
                                                                self.K_ROLL, 
                                                                self.max_pitch_boost, 
                                                                self.max_roll_boost
                                                                )

        # ---------------------------------------------------------
        # HIERARCHICAL DECISION TREE 
        # ---------------------------------------------------------
        
        intended_movement = odor_steer
        #print(f"ODOR STEER : {odor_steer}")


        #print(f"PITCH : {pitch}")

        if abs(roll) > self.max_roll :
            #print(f"ROLL CORRECTION : {roll_compensation}")
            intended_movement += roll_compensation 
            
        if abs(pitch) > self.max_pitch :
            #print(f"PITCH CORRECTION : {pitch_compensation}")
            intended_movement += pitch_compensation * self.tilt_gain
    
        
        final_drive = intended_movement * self.speed_gain


        return np.clip(final_drive, a_min=-self.speed_gain, a_max=self.speed_gain)


    
    def step(self, sim: MiniprojectSimulation):
        # Observations 
        olfaction = sim.get_olfaction(sim.fly.name)
        quat = sim.get_body_rotations(sim.fly.name)[0]
        #omm = sim.get_ommatidia_readouts(sim.fly.name)
        
        self.counter=self.counter+1

        faster_sim=True

        if faster_sim :
            if self.counter%80==0 :
                rot = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
                pitch, _, _ = rot.as_euler('xyz', degrees=True)
                self.obstacle_l,self.obstacle_r,_, _, _=self._get_raw_vision_obstacles_2(sim,pitch=pitch)
                print(f"Obstacle R : {self.obstacle_r}")
                #print(f"obstacer{self.obstacle_r}")

        else :
            self.obstacle_l,self.obstacle_r=self._get_raw_vision_obstacles(sim)
            print(f"obstacer{self.obstacle_r}")

        """ if self.counter%5000==0 : #alignement check every once in a while
            self._aligned=False  """
        
        """ print(f"Obstacle_left : {self.obstacle_l}")
        print(f"Obstacle_right : {self.obstacle_r}") """
        turn, avoid = self._avoid_obstacles(self.obstacle_l,self.obstacle_r)
        roll_compensation, pitch_compensation, pitch, roll = OLD_movement_correction.tilt_to_control_signal(quat, 
                                                                self.K_PITCH, 
                                                                self.K_ROLL, 
                                                                self.max_pitch_boost, 
                                                                self.max_roll_boost
                                                                )


        if avoid :
            left_drive  = np.clip(1.0 - turn, 0.0, 3.0)
            right_drive = np.clip(1.0 + turn, 0.0, 3.0)
            if abs(roll) > self.max_roll :
                #print(f"ROLL CORRECTION : {roll_compensation}")
                left_drive += roll_compensation[0]
                right_drive += roll_compensation[1]
                
            if abs(pitch) > self.max_pitch :
                #print(f"PITCH CORRECTION : {pitch_compensation}")
                left_drive += pitch_compensation[0] * self.tilt_gain
                right_drive += pitch_compensation[1] * self.tilt_gain
            print(f"TURN -> left : {left_drive}, right : {right_drive}")
            joint_angles, adhesion = self.turning_controller.step(np.array([left_drive,right_drive]))
            return joint_angles, adhesion 


        # Output control logic to drives
        drives = self.drive_logic(olfaction, quat)
        """ avoid_steer, _, _, leftomm, rightomm = movement_correction.process_vision_and_steer(omm, self.retina)
        print(f"Drives: {drives}, Olfaction: {olfaction}, OMMleft: {leftomm:.2f}, OMMright: {rightomm:.2f}") """
        joint_angles, adhesion = self.turning_controller.step(drives)

        return joint_angles, adhesion


    def _get_raw_vision_obstacles(self, sim, pitch=0.0):

            images = sim.get_raw_vision(sim.fly.name)
            H = images[0].shape[0]

            # Plus le pitch est positif (montée), plus on regarde bas dans l'image
            pitch_offset = int(np.clip(pitch * 1.5, -H//3, H//3))
            crop_start = max(0, H//3 + pitch_offset)  # ← s'adapte au pitch
            crop_end   = max(crop_start + 1, H//3 + pitch_offset)

            self._current_crop_row = crop_start

            def green_ratio(img):
                top_strip = img[:H // 3] #changed from 4 to 3  # top 25% — only close stalks reach this high
                #top_strip = img[crop_start:crop_end]
                r = top_strip[:, :, 0].astype(float)
                g = top_strip[:, :, 1].astype(float)
                b = top_strip[:, :, 2].astype(float)
                total = r + g + b + 1e-6
                return float(((g / total > 0.4) & (total > 30)).mean())

            obstacle_left  = green_ratio(images[0])
            obstacle_right = green_ratio(images[1])
            #print(f"Raw vision green ratio: L={obstacle_left:.3f} R={obstacle_right:.3f}")

            return obstacle_left, obstacle_right
    
    def _avoid_obstacles(self, obstacle_left: float, obstacle_right: float):

        diff = obstacle_left - obstacle_right
       
        turn = 0

        if obstacle_left > self.avoidance_threshold or obstacle_right > self.avoidance_threshold:
            print(f"DIFFERENCE : {diff}")
            if abs(diff) < 0.1 : #0.06
                print(f"GOING TROUGH")
                return 0, False  # the obstacle is approximately same on both side -> we can go through
            
            
            #turn = -np.sign(diff) * np.tanh(abs(diff) * 10) * 5.0
            #turn = -np.sign(diff if abs(diff) > 0.01 else 1.0) * 4
            turn = -np.sign(diff if abs(diff) > 0.001 else 1.0) * 3.0
            #print(f"REFLEX! L={obstacle_left:.3f} R={obstacle_right:.3f} diff={diff:+.3f} turn={turn:+.2f}")
            return turn, True

        
        return turn, False
    
    def _get_raw_vision_obstacles_2(self, sim, pitch: float):

        images = sim.get_raw_vision(sim.fly.name)
        H, W = images[0].shape[:2]

        pitch_offset = int(np.clip(pitch * 1.5, -H//4, H//4))
        horizon = np.clip(H // 2 + pitch_offset, H//4, 3*H//4)  # ← nouveau
        self._current_crop_row = horizon  # pour visualisation

        def analyze_eye(img, forward_col_start, forward_col_end):
            r = img[:, :, 0].astype(float)
            g = img[:, :, 1].astype(float)
            b = img[:, :, 2].astype(float)
            total = r + g + b + 1e-6
            g_ratio = g / total

            bottom_strip = g_ratio[int(H * 0.8):]

            # Garder seulement les pixels valides (pas noir, pas ciel)
            valid = bottom_strip[total[int(H * 0.8):] > 40]
            if len(valid) == 0:
                return 0.0, 0.0

            ground_green_mean = float(np.median(valid))  # médiane plus robuste que mean
            ground_green_std  = float(valid.std()) + 1e-6

            # Seuil clipé à max 0.95 pour rester dans [0,1]
            threshold = min(ground_green_mean + 1.0 * ground_green_std, 0.95)

            is_grass = (
                (g_ratio > threshold) &
                (g > b * 1.3) &
                (g > r * 1.1) &
                (total > 40)
            )
            print(f"is_grass pixels: {is_grass.sum()} / {H*W}")

            upper_mask = np.zeros((H, W), dtype=bool)
            upper_mask[:horizon, forward_col_start:forward_col_end] = True
            grass_upper = is_grass & upper_mask
            print(f"grass_upper pixels: {grass_upper.sum()}")

            if grass_upper.sum() == 0:
                return 0.0, 0.0

            # Proximity score: fraction of upper forward pixels that are grass
            forward_pixels = upper_mask.sum()
            obstacle_score = float(grass_upper.sum()) / (forward_pixels + 1e-6)

            # Height score: how high up does the grass reach?
            # Find the highest (smallest row index) grass pixel in forward region
            rows_with_grass = np.where(grass_upper.any(axis=1))[0]
            if len(rows_with_grass) == 0:
                height_score = 0.0
            else:
                highest_row = rows_with_grass.min()   # 0 = very top = very close
                # Normalize: 0 = at horizon (H//2), 1 = at very top of image
                height_score = float(1.0 - highest_row / (horizon + 1e-6))
                height_score = np.clip(height_score, 0.0, 1.0)

            # Combine: obstacle needs both presence AND height to trigger
            # height_score alone avoids false positives from distant grass
            combined = 0.5 * obstacle_score + 0.5 * height_score
            return float(combined), float(height_score)

        # Left eye: forward-facing = right half of image
        # Right eye: forward-facing = left half of image
        left_score,  left_height  = analyze_eye(images[0], W // 2, W)
        right_score, right_height = analyze_eye(images[1], 0,      W // 2)

        total_obstacle = (left_score + right_score) / 2.0

        # Red detection for dragonfly — unchanged
        def red_ratio(img):
            r = img[:, :, 0].astype(float)
            g = img[:, :, 1].astype(float)
            b = img[:, :, 2].astype(float)
            return float(((r > 150) & (r > 2 * g) & (r > 2 * b)).mean())

        red_left  = red_ratio(images[0])
        red_right = red_ratio(images[1])

        return left_score, right_score, total_obstacle, red_left, red_right

    def _detect_dragonfly(self, red_left: float, red_right: float):
        thr = 1e-3
        if red_left < thr and red_right < thr:
            return False, 0
        if red_left > thr and red_right > thr:
            return True, 0    # coming from front
        elif red_left > thr:
            return True, -1   # coming from left
        else:
            return True, +1   # coming from right

    def _dragonfly_escape(self, side: int) -> np.ndarray:
        if side == -1:
            return np.array([2.0, 3.0])   # dragonfly left → turn right
        elif side == +1:
            return np.array([3.0, 2.0])   # dragonfly right → turn left
        else:
            return np.array([4.0, 4.0])   # dragonfly front → run straight