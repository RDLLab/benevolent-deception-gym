"""Driver Assistant Environment with discrete actions. """

from bdgym.envs.driver_assistant.env import DriverAssistantEnv


class DiscreteDriverAssistantEnv(DriverAssistantEnv):
    """Driver Assistant Environment with discrete actions """

    def default_config(self) -> dict:
        config = super().default_config()
        config['action']['assistant']['type'] = 'AssistantDiscreteAction'
        config['action']['driver']['type'] = 'DriverDiscreteAction'
        config['observation']['type'] = 'DiscreteDriverAssistantObservation'
        return config
