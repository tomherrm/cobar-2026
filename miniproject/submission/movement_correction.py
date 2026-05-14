import numpy as np
from scipy.spatial.transform import Rotation

# ═══════════════════════════════════════════════════════════════════════════════
#  TUNING GUIDE
#    k_pitch        : 0.03–0.10  (0.05 recommended)
#    k_roll         : 0.03–0.10  (0.06 recommended)
#    max_pitch_boost: hard ceiling on pitch correction (0.3–0.6)
#    max_roll_boost : hard ceiling on roll  correction (0.2–0.4)
#
#  In controller.py:
#    self.K_PITCH         = 0.05
#    self.K_ROLL          = 0.06
#    self.max_pitch_boost = 0.5
#    self.max_roll_boost  = 0.35
#    self.max_pitch_deg   = 10
#    self.max_roll_deg    = 8
# ═══════════════════════════════════════════════════════════════════════════════

def tilt_to_control_signal(quat, k_pitch=0.05, k_roll=0.06,
                           max_pitch_boost=0.5, max_roll_boost=0.35):
    rot = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
    pitch_deg, roll_deg, _ = rot.as_euler('xyz', degrees=True)

    # Pitch : boost symétrique uniquement en montée
    pitch_scalar = max_pitch_boost * float(np.tanh(max(0.0, pitch_deg) * k_pitch))
    pitch_boost  = np.array([pitch_scalar, pitch_scalar])

    # Roll : boost asymétrique selon le côté bas
    roll_scalar = max_roll_boost * float(np.tanh(roll_deg * k_roll))
    roll_boost  = np.array([-roll_scalar, roll_scalar])

    return roll_boost, pitch_boost, pitch_deg, roll_deg