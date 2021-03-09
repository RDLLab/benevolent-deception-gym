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
