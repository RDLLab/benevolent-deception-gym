from gym.envs.registration import register

from .env import ExerciseAssistantEnv
from .discrete_env import DiscreteExerciseAssistantEnv
from .fixed_athlete_env import (
    FixedAthleteExerciseAssistantEnv,
    DiscreteFixedAthleteExerciseAssistantEnv
)
from .policy import (
    RandomWeightedAthletePolicy,
    ObedientAthletePolicy,
    IndependentAthletePolicy
)

EXERCISE_ASSISTANT_GYM_ENVS = [
    'ExerciseAssistant-v0',
    'ExerciseAssistantContinuous-v0',
]
"""List of names of registered Exercise-Assistant Multi-agent Gym Envs """

FIXED_ATHLETE_EXERCISE_ASSISTANT_GYM_ENVS = [
    'ExerciseAssistantHA-v0',
    'ExerciseAssistantContinuousHA-v0',
    'ExerciseAssistantOA-v0',
    'ExerciseAssistantContinuousOA-v0',
    'ExerciseAssistantIA-v0',
    'ExerciseAssistantContinuousIA-v0',
]
"""List of names of registered Exercise-Assistant Gym Envs with fixed athlete
   policy.
"""

ALL_EXERCISE_ASSISTANT_GYM_ENVS = \
    EXERCISE_ASSISTANT_GYM_ENVS + FIXED_ATHLETE_EXERCISE_ASSISTANT_GYM_ENVS
"""List of names of all registered Exercise-Assistant Gym Envs """


# Exercise Assistant Env
# ----------------------
register(
    id='ExerciseAssistant-v0',
    entry_point='bdgym.envs:DiscreteExerciseAssistantEnv',
)

register(
    id='ExerciseAssistantContinuous-v0',
    entry_point='bdgym.envs:ExerciseAssistantEnv',
)


# Exercise Assistant Env with fixed Athlete Policy
# ------------------------------------------------
register(
    id='ExerciseAssistantHA-v0',
    entry_point='bdgym.envs:DiscreteFixedAthleteExerciseAssistantEnv',
    kwargs={'athlete_policy': RandomWeightedAthletePolicy()}
)

register(
    id='ExerciseAssistantContinuousHA-v0',
    entry_point='bdgym.envs:FixedAthleteExerciseAssistantEnv',
    kwargs={'athlete_policy': RandomWeightedAthletePolicy()}
)

register(
    id='ExerciseAssistantOA-v0',
    entry_point='bdgym.envs:DiscreteFixedAthleteExerciseAssistantEnv',
    kwargs={'athlete_policy': ObedientAthletePolicy()}
)

register(
    id='ExerciseAssistantContinuousOA-v0',
    entry_point='bdgym.envs:FixedAthleteExerciseAssistantEnv',
    kwargs={'athlete_policy': ObedientAthletePolicy()}
)

register(
    id='ExerciseAssistantIA-v0',
    entry_point='bdgym.envs:DiscreteFixedAthleteExerciseAssistantEnv',
    kwargs={'athlete_policy': IndependentAthletePolicy()}
)

register(
    id='ExerciseAssistantContinuousIA-v0',
    entry_point='bdgym.envs:FixedAthleteExerciseAssistantEnv',
    kwargs={'athlete_policy': IndependentAthletePolicy()}
)
