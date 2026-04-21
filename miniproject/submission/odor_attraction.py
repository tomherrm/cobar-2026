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