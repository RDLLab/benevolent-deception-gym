"""Fixed Athlete Policies """
from typing import Union
from abc import ABC, abstractmethod

import numpy as np

import bdgym.envs.utils as utils
from bdgym.envs.exercise_assistant.action import \
    AthleteAction, DiscreteAssistantAction
from bdgym.envs.exercise_assistant.observation import \
    assistant_obs_str, athlete_obs_str


class AthletePolicy(ABC):
    """Fixed Athlete Policy for Exercise Assistant Environment """

    @abstractmethod
    def get_action(self, obs: np.ndarray) -> AthleteAction:
        """Get athlete action for given observation

        Parameters
        ----------
        obs : np.ndarray
            the athlete observation

        Returns
        -------
        AthleteAction
           the athletes action
        """

    def reset(self):
        """Reset the policy following the end of an episode.

        Does nothing by default
        """


class AssistantPolicy(ABC):
    """Fixed Assistant Policy for Exercise Assistant Environment """

    @abstractmethod
    def get_action(self,
                   obs: np.ndarray
                   ) -> Union[DiscreteAssistantAction, np.ndarray]:
        """Get assistant action for given observation

        Parameters
        ----------
        obs : np.ndarray
            the athlete observation

        Returns
        -------
        Union[DiscreteAssistantAction, np.ndarray]
           the assistants action
        """

    def reset(self):
        """Reset the policy following the end of an episode.

        Does nothing by default
        """


class RandomAthletePolicy(AthletePolicy):
    """Athlete Policy that selects action uniformly at random """

    def get_action(self, obs: np.ndarray) -> AthleteAction:
        return np.random.choice(AthleteAction)


class WeightedAthletePolicy(AthletePolicy):
    """Athlete Policy that considers assistant feedback in it's decision.

    Specifically,
    1. the athlete's observed energy is a weighted sum of the athletes
       percieved energy and the assistants communicated energy, weighted by the
       independence parameter
    2. the athletes action choice is then to perform a repetition if the
       observed energy >= the threshold parameter
    3. Finally, the action performed is based the weighted sum of the athletes
       action choice and the assistant recommendation, again weighted by the
       independence parameter. If the this weighted sum >= 0.5 then athlete
       will END_SET, otherwise athlete will PERFORM_REP.

    Note if independence == 1.0, then the policy == GreedyAthletePolicy, and
    if independence == 0.0, then the policy == ObedientAthletePolicy
    """

    def __init__(self,
                 threshold: float = 0.1,
                 independence: float = 0.5):
        self.threshold = threshold
        self.independence = independence

    def get_action(self, obs: np.ndarray) -> AthleteAction:
        energy_obs = self.independence*obs[0] + (1-self.independence)*obs[2]
        if energy_obs >= self.threshold:
            athlete_choice = AthleteAction.PERFORM_REP
        else:
            athlete_choice = AthleteAction.END_SET
        assistant_choice = obs[3]
        action_prob = (
            self.independence*int(athlete_choice)
            + (1-self.independence)*assistant_choice
        )
        if action_prob >= 0.5:
            return AthleteAction.END_SET
        return AthleteAction.PERFORM_REP


class RandomWeightedAthletePolicy(WeightedAthletePolicy):
    """A WeightedAthletePolicy but where independence changes over time.

    Specifically, each time the RandomWeightedAthletePolicy.reset() function is
    called a new independence value is sampled at random from a normal
    distribution.
    """

    def __init__(self,
                 threshold: float = 0.1,
                 independence_mean: float = 0.5,
                 independence_std: float = 0.25):
        self.independence_dist = utils.get_truncated_normal(
            independence_mean, independence_std, 0.0, 1.0
        )
        super().__init__(threshold, self.independence_dist.rvs())

    def reset(self):
        self.independence = self.independence_dist.rvs()


class ObedientAthletePolicy(WeightedAthletePolicy):
    """Athlete Policy that selects follows the assistant recommendation.

    Specifically, if the assistant recommendation is >= 0.5 then athlete will
    END_SET, otherwise athlete will PERFORM_REP.
    """

    def __init__(self,
                 threshold: float = 0.1):
        super().__init__(threshold, 0.0)


class GreedyAthletePolicy(WeightedAthletePolicy):
    """Athlete Policy that performs rep if percieved energy >= threshold """

    def __init__(self,
                 threshold: float = 0.1):
        super().__init__(threshold, 1.0)


class ManualAthletePolicy(AthletePolicy):
    """Athlete Policy where actions controlled by manual keyboard control """

    def __init__(self, print_obs: bool = True):
        self.print_obs = print_obs

    def get_action(self, obs: np.ndarray) -> AthleteAction:
        if self.print_obs:
            print(f"Observation: {athlete_obs_str(obs)}")

        print("\nSelect Athlete Action from:")

        output = ["Actions:"]
        for i in range(len(AthleteAction)):
            output.append(f"{i}={str(AthleteAction(i))}")
        print(" ".join(output))

        while True:
            try:
                idx = int(input("Choose action number: "))
                action = AthleteAction(idx)
                print(f"Performing: {str(action)}")
                return action
            except ValueError:
                print("Invalid choice. Try again.")


class RandomDiscreteAssistantPolicy(AssistantPolicy):
    """Assistant Policy that selects action uniformly at random """

    def get_action(self, obs: np.ndarray) -> DiscreteAssistantAction:
        return np.random.choice(DiscreteAssistantAction)


class GreedyDiscreteAssistantPolicy(AssistantPolicy):
    """Assistant Policy that reports observed athlete energy truthfully and
    recommends athlete performs rep if observed energy >= threshold, otherwise
    recommends athlete end set
    """

    def __init__(self, threshold: float = 0.1):
        self.threshold = threshold

    def get_action(self, obs: np.ndarray) -> DiscreteAssistantAction:
        if obs[0] >= self.threshold:
            return DiscreteAssistantAction.NO_CHANGE_PERFORM_REP
        return DiscreteAssistantAction.NO_CHANGE_END_SET


class ManualAssistantPolicy(AssistantPolicy):
    """Assistant Policy where actions controlled by manual keyboard control """

    def __init__(self, print_obs: bool = True):
        self.print_obs = print_obs

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        if self.print_obs:
            print(f"Observation: {assistant_obs_str(obs)}")

        signal = 0.0
        rcmd = 0.0
        print("\nSelect Assistant Action (energy to report in [0.0, 1.0]:")
        while True:
            try:
                signal = float(input("Energy: "))
                assert 0.0 <= signal <= 1.0
                break
            except (ValueError, AssertionError):
                print("Invalid choice. Try again.")

        print("Select Assistant recommended Action (in [0.0, 1.0]:")
        while True:
            try:
                rcmd = float(input("Recommendation: "))
                assert 0.0 <= rcmd <= 1.0
                break
            except (ValueError, AssertionError):
                print("Invalid choice. Try again.")

        action = np.array([signal, rcmd])
        print(f"Performing: {np.array_str(action, precision=4)}")
        return utils.lmap_array(action, [0.0, 1.0], [-1.0, 1.0])


class ManualDiscreteAssistantPolicy(ManualAssistantPolicy):
    """Assistant Policy where actions controlled by manual keyboard control """

    def get_action(self, obs: np.ndarray) -> DiscreteAssistantAction:
        if self.print_obs:
            print(f"Observation: {assistant_obs_str(obs)}")

        print("\nSelect Assistant Action from:")

        output = ["Actions:"]
        for i in range(len(DiscreteAssistantAction)):
            output.append(f"{i}={str(DiscreteAssistantAction(i))}")
        print(" ".join(output))

        while True:
            try:
                idx = int(input("Choose action number: "))
                action = DiscreteAssistantAction(idx)
                print(f"Performing: {str(action)}")
                return action
            except ValueError:
                print("Invalid choice. Try again.")


ATHLETE_POLICIES = {
    'greedy': GreedyAthletePolicy,
    'random': RandomAthletePolicy,
    'obedient': ObedientAthletePolicy,
    'weighted': WeightedAthletePolicy,
    'random_weighted': RandomWeightedAthletePolicy
}


ASSISTANT_POLICIES = {
    'discrete_greedy': GreedyDiscreteAssistantPolicy,
    'discrete_random': RandomDiscreteAssistantPolicy,
}
