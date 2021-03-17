from gym.envs.registration import register

from .env import DriverAssistantEnv
from .discrete_env import DiscreteDriverAssistantEnv
from .fixed_driver_env import (
    FixedDriverDriverAssistantEnv,
    DiscreteFixedDriverDriverAssistantEnv
)
from .driver_types import get_driver_config


DRIVER_ASSISTANT_GYM_ENVS = [
    'DriverAssistant-v0',
    'DriverAssistantContinuous-v0',
]
"""List of names of registered Driver-Assistant Multi-agent Gym Envs """

FIXED_DRIVER_DRIVER_ASSISTANT_GYM_ENVS = [
    'DriverAssistantHD-v0',
    'DriverAssistantContinuousHD-v0',
    'DriverAssistantOD-v0',
    'DriverAssistantContinuousOD-v0',
    'DriverAssistantID-v0',
    'DriverAssistantContinuousID-v0',
    'DriverAssistantAggressiveID-v0',
    'DriverAssistantContinuousAggressiveID-v0'
]
"""List of names of registered Driver-Assistant Gym Envs with fixed driver
   policy.
"""

ALL_DRIVER_ASSISTANT_GYM_ENVS = \
    DRIVER_ASSISTANT_GYM_ENVS + FIXED_DRIVER_DRIVER_ASSISTANT_GYM_ENVS
"""List of names of all registered Driver-Assistant Gym Envs """


# Driver Assistant Env
# --------------------
register(
    id='DriverAssistant-v0',
    entry_point='bdgym.envs:DiscreteDriverAssistantEnv',
)

register(
    id='DriverAssistantContinuous-v0',
    entry_point='bdgym.envs:DriverAssistantEnv',
)


# Driver Assistant Env with fixed Driver Policy
# ---------------------------------------------

# Changing 'Human' driver policy
register(
    id='DriverAssistantHD-v0',
    entry_point='bdgym.envs:DiscreteFixedDriverDriverAssistantEnv',
    kwargs={
        'driver_policy_config': {
            "type": "ChangingGuidedIDMDriverPolicy"
        }
    }
)

register(
    id='DriverAssistantContinuousHD-v0',
    entry_point='bdgym.envs:FixedDriverDriverAssistantEnv',
    kwargs={
        'driver_policy_config': {
            "type": "ChangingGuidedIDMDriverPolicy"
        }
    }
)

# Changing Obedient driver policy
register(
    id='DriverAssistantOD-v0',
    entry_point='bdgym.envs:DiscreteFixedDriverDriverAssistantEnv',
    kwargs={
        'driver_policy_config': {
            "type": "GuidedIDMDriverPolicy",
            'independence': 0.0,
            **get_driver_config('standard')
        }
    }
)

register(
    id='DriverAssistantContinuousOD-v0',
    entry_point='bdgym.envs:FixedDriverDriverAssistantEnv',
    kwargs={
        'driver_policy_config': {
            "type": "GuidedIDMDriverPolicy",
            'independence': 0.0,
            **get_driver_config('standard')
        }
    }
)

# Independent 'Human' Driver Policy - 'Standard'
register(
    id='DriverAssistantID-v0',
    entry_point='bdgym.envs:DiscreteFixedDriverDriverAssistantEnv',
    kwargs={
        'driver_policy_config': {
            "type": "GuidedIDMDriverPolicy",
            'independence': 1.0,
            **get_driver_config('standard')
        }
    }
)

register(
    id='DriverAssistantContinuousID-v0',
    entry_point='bdgym.envs:FixedDriverDriverAssistantEnv',
    kwargs={
        'driver_policy_config': {
            "type": "GuidedIDMDriverPolicy",
            'independence': 1.0,
            **get_driver_config('standard')
        }
    }
)

# Independent 'Human' Driver Policy - 'Aggressive'
register(
    id='DriverAssistantAggressiveID-v0',
    entry_point='bdgym.envs:DiscreteFixedDriverDriverAssistantEnv',
    kwargs={
        'driver_policy_config': {
            "type": "GuidedIDMDriverPolicy",
            'independence': 1.0,
            **get_driver_config('aggressive')
        }
    }
)

register(
    id='DriverAssistantContinuousAggressiveID-v0',
    entry_point='bdgym.envs:FixedDriverDriverAssistantEnv',
    kwargs={
        'driver_policy_config': {
            "type": "GuidedIDMDriverPolicy",
            'independence': 1.0,
            **get_driver_config('aggressive')
        }
    }
)
