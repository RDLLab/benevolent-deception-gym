"""Base abstract policy class """
from abc import ABC, abstractmethod


class BasePolicy(ABC):
    """Base policy interface """

    @abstractmethod
    def get_action(self, obs):
        """Get action from policy for given obs. """
