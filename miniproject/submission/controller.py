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

        self.speed_gain = 1.8
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
        self.avoidance_threshold = 0.05#0.03#0.085#0.04#0.01
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
        #print(f"ODOR STEER : {odor_steer}")


        print(f"PITCH : {pitch}")

        if abs(roll) > self.max_roll :
            print(f"ROLL CORRECTION : {roll_compensation}")
            intended_movement += roll_compensation 
            
        if abs(pitch) > self.max_pitch :
            print(f"PITCH CORRECTION : {pitch_compensation}")
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
                self.obstacle_l,self.obstacle_r=self._get_raw_vision_obstacles(sim,pitch=0)
                #print(f"obstacer{self.obstacle_r}")

        else :
            self.obstacle_l,self.obstacle_r=self._get_raw_vision_obstacles(sim)
            print(f"obstacer{self.obstacle_r}")

        """ if self.counter%5000==0 : #alignement check every once in a while
            self._aligned=False  """
           
        turn, avoid = self._avoid_obstacles(self.obstacle_l,self.obstacle_r)

        """ if obstacle_l > obstacle_r:
                drives = np.array([3.0,-3.0])  # virer droite
            else:
                drives = np.array([-3.0,3.0])  # virer gauche
            #print(drives)
            joint_angles, adhesion = self.turning_controller.step(drives)
            """
        """ left_drive  = np.clip(1.0 - turn, 0.0, 3.0) 
            right_drive = np.clip(1.0 + turn, 0.0, 3.0)
            joint_angles, adhesion = self.turning_controller.step(np.array([left_drive, right_drive]))  """

        """ if avoid and self._avoid_counter==0: ###HOLD THE TURN FOR A CERTAIN AMMOUNT OF TIME
            print("start turning")
            
            left_drive  = np.clip(1.0 - turn, 0.0, 3.0)
            right_drive = np.clip(1.0 + turn, 0.0, 3.0)
            self._last_drives = np.array([left_drive, right_drive])
            self._avoid_counter = 30  

        if self._avoid_counter > 0:
            print(f"TURN ->{self._last_drives}")
            self._avoid_counter -= 1
            joint_angles, adhesion = self.turning_controller.step(self._last_drives)
            return joint_angles, adhesion """
        

        if avoid :
            
            left_drive  = np.clip(1.0 - turn, 0.0, 3.0)
            right_drive = np.clip(1.0 + turn, 0.0, 3.0)
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
            print(diff)
            if abs(diff) < 0.05 : #0.06
                print(f"GOING TROUGH")
                return 0, False  # the obstacle is approximately same on both side -> we can go through
            
            turn = -np.sign(diff) * np.tanh(abs(diff) * 5) * 3.0 #20 -> 5
            #turn = -np.sign(diff if abs(diff) > 0.01 else 1.0) * 4
            #print(f"REFLEX! L={obstacle_left:.3f} R={obstacle_right:.3f} diff={diff:+.3f} turn={turn:+.2f}")
            return turn, True

        
        return turn, False