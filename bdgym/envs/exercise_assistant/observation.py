"""Functions related to Exercise Assistant Observations """
import numpy as np


def assistant_obs_str(obs: np.ndarray) -> str:
    """Get string representation of assistant observation

    Parameters
    ----------
    obs : np.ndarray
        the assistant observation

    Returns
    -------
    str
        a human readable string representation of the observation
    """
    assert obs.shape[0] in (3, 4), \
        f"Assistant observation should be of shape (3,) or (4,). {obs} invalid"

    obs_str = [
        f"Athlete_Energy={obs[0]:.3f}",
        f"Sets_Completed={obs[1]:.3f}",
        f"Athlete_Action={obs[2]:.3f}"
    ]
    if obs.shape[0] == 4:
        obs_str.append(f"Signal_Offset={obs[3]:.3f}")
    return " ".join(obs_str)


def athlete_obs_str(obs: np.ndarray) -> str:
    """Get string representation of athlete observation

    Parameters
    ----------
    obs : np.ndarray
        the athlete observation

    Returns
    -------
    str
        a human readable string representation of the observation
    """
    assert obs.shape[0] == 4, \
        f"Athlete observation should be of shape (4,). {obs} invalid"

    obs_str = [
        f"Percieved_Energy={obs[0]:.3f}",
        f"Sets_Completed={obs[1]:.3f}",
        f"Assistant_Signal={obs[2]:.3f}",
        f"Assistant_Rcmd={obs[3]:.3f}"
    ]
    return " ".join(obs_str)
