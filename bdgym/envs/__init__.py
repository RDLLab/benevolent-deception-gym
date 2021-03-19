from .driver_assistant import (
    DriverAssistantEnv,
    DiscreteDriverAssistantEnv,
    FixedDriverDriverAssistantEnv,
    DiscreteFixedDriverDriverAssistantEnv,
    ALL_DRIVER_ASSISTANT_GYM_ENVS
)
from .exercise_assistant import (
    ExerciseAssistantEnv,
    DiscreteExerciseAssistantEnv,
    FixedAthleteExerciseAssistantEnv,
    DiscreteFixedAthleteExerciseAssistantEnv,
    ALL_EXERCISE_ASSISTANT_GYM_ENVS
)

BDGYM_ENVS = ALL_DRIVER_ASSISTANT_GYM_ENVS + ALL_EXERCISE_ASSISTANT_GYM_ENVS
"""List of names of all bdgym envs registered in OpenAI gym """
