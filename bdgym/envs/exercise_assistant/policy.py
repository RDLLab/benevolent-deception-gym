"""Fixed Athlete Policies """
from typing import Union
from abc import ABC, abstractmethod

import numpy as np

import bdgym.envs.utils as utils
from bdgym.envs.exercise_assistant.manual_control import AssistantEventHandler
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
       perception influence parameter
    2. the athletes action choice is then to perform a repetition if the
       observed energy >= the threshold parameter
    3. Finally, the action performed is based the weighted sum of the athletes
       action choice and the assistant recommendation, again weighted by the
       independence parameter. If the this weighted sum >= 0.5 then athlete
       will END_SET, otherwise athlete will PERFORM_REP.

    Note if independence == 1.0, then the policy == GreedyAthletePolicy, and
    if independence == 0.0, then the policy == ObedientAthletePolicy.

    The perception_influence parameter basically controls how much the provided
    assistant signal of the athletes energy affects the athletes own perception
    about their energy level.

    - A value of 0.0 means the athlete energy level observation will be solely
      their percieved energy level (i.e. their noisy internal observation of
      their energy level).
    - A value of 1.0 means the athlete energy level observation will be solely
      the level provided by the assistant.
    - Values between 0.0 and 1.0 are a weighted linear combination of the two
      sources: the athletes percieved energy level and the assistants reading.

    In this way, depending on the value of the perception_influence parameter
    the Assistant can influence the action of the Athlete using the energy
    level reading it provides.
    """

    def __init__(self,
                 threshold: float = 0.1,
                 perception_influence: float = 0.5,
                 independence: float = 0.5):
        assert 0.0 < threshold < 1.0
        assert 0.0 <= perception_influence <= 1.0
        assert 0.0 <= independence <= 1.0
        self.threshold = threshold
        self.perception_influence = perception_influence
        self.independence = independence

    def get_action(self, obs: np.ndarray) -> AthleteAction:
        energy_obs = (
            (1-self.perception_influence)*obs[0]
            + self.perception_influence*obs[2]
        )
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
    called a new independence value and a new perception_influence value are
    sampled at random from independent normal distributions.
    """

    def __init__(self,
                 threshold_mean: float = 0.15,
                 threshold_std: float = 0.1,
                 perception_influence_mean: float = 0.5,
                 perception_influence_std: float = 0.25,
                 independence_mean: float = 0.5,
                 independence_std: float = 0.25):
        self.threshold_dist = utils.get_truncated_normal(
            threshold_mean, threshold_std, 0.0, 1.0
        )
        self.perception_influence_dist = utils.get_truncated_normal(
            perception_influence_mean, perception_influence_std, 0.0, 1.0
        )
        self.independence_dist = utils.get_truncated_normal(
            independence_mean, independence_std, 0.0, 1.0
        )
        super().__init__(
            self.threshold_dist.rvs(),
            self.perception_influence_dist.rvs(),
            self.independence_dist.rvs()
        )

    def reset(self):
        self.threshold = self.threshold_dist.rvs()
        self.perception_influence = self.perception_influence_dist.rvs()
        self.independence = self.independence_dist.rvs()


class ObedientAthletePolicy(WeightedAthletePolicy):
    """Athlete Policy that selects follows the assistant recommendation.

    Specifically, if the assistant recommendation is >= 0.5 then athlete will
    END_SET, otherwise athlete will PERFORM_REP.
    """

    def __init__(self,
                 threshold: float = 0.1,
                 perception_influence: float = 0.5):
        super().__init__(threshold, perception_influence, 0.0)


class IndependentAthletePolicy(WeightedAthletePolicy):
    """Athlete Policy that performs rep if percieved energy >= threshold

    The threshold for this policy is intentionally set quite low so it's
    necessary for the assistant to employ deception to avoid the athlete
    overexerting themself.
    """

    def __init__(self,
                 threshold: float = 0.05,
                 perception_influence: float = 0.5):
        super().__init__(threshold, perception_influence, 1.0)


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


class DoNothingDiscreteAssistantPolicy(AssistantPolicy):
    """Assistant Policy that reports observed athlete energy truthfully and
    always recommends athlete performs rep
    """

    def get_action(self, obs: np.ndarray) -> DiscreteAssistantAction:
        return DiscreteAssistantAction.NO_CHANGE_PERFORM_REP


class ManualAssistantPolicy(AssistantPolicy):
    """Assistant Policy where actions controlled by manual keyboard control """

    def __init__(self, print_obs: bool = True, event_driven: bool = True):
        self.print_obs = print_obs
        self.event_driven = event_driven

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        if self.print_obs:
            print(f"Observation: {assistant_obs_str(obs)}")

        signal = 0.0
        rcmd = 0.0
        print("\nSelect Assistant Actions:")
        while True:
            try:
                user_input = input(
                    "Energy in [0.0, 1.0] or '' to report observed energy: "
                )
                if user_input == "":
                    signal = obs[0]
                else:
                    signal = float(user_input)
                assert 0.0 <= signal <= 1.0
                break
            except (ValueError, AssertionError):
                print("Invalid choice. Try again.")

        while True:
            try:
                rcmd = float(
                    input(
                        "Recommendation in [0.0, 1.0] "
                        "(0.0=PERFORM_REP, 1.0=END_SET): "
                    )
                )
                assert 0.0 <= rcmd <= 1.0
                break
            except (ValueError, AssertionError):
                print("Invalid choice. Try again.")

        action = np.array([signal, rcmd])
        print(f"Performing: {np.array_str(action, precision=4)}")
        return utils.lmap_array(action, [0.0, 1.0], [-1.0, 1.0])


class ManualDiscreteAssistantPolicy(ManualAssistantPolicy):
    """Assistant Policy where actions controlled by manual keyboard control """

    def get_action(self, obs: np.ndarray, env=None) -> DiscreteAssistantAction:
        if self.print_obs:
            print(f"Observation: {assistant_obs_str(obs)}\n")

        if self.event_driven:
            return self._get_event_action(env)
        return self._get_terminal_action()

    def _get_event_action(self, env) -> DiscreteAssistantAction:
        action = None
        while action is None:
            action = AssistantEventHandler.handle_discrete_events(env)
        return action

    def _get_terminal_action(self) -> DiscreteAssistantAction:
        print("Select Assistant Action from:")

        output = []
        for i in range(len(DiscreteAssistantAction)):
            output.append(f"  {i}={str(DiscreteAssistantAction(i))}")
        print("\n".join(output))

        while True:
            try:
                idx = int(input("Choose action number: "))
                action = DiscreteAssistantAction(idx)
                print(f"Performing: {str(action)}")
                return action
            except ValueError:
                print("Invalid choice. Try again.")


ATHLETE_POLICIES = {
    'random': RandomAthletePolicy,
    'obedient': ObedientAthletePolicy,
    'independent': IndependentAthletePolicy,
    'weighted': WeightedAthletePolicy,
    'random_weighted': RandomWeightedAthletePolicy
}


ASSISTANT_POLICIES = {
    'discrete_greedy': GreedyDiscreteAssistantPolicy,
    'discrete_random': RandomDiscreteAssistantPolicy,
    'discrete_donothing': DoNothingDiscreteAssistantPolicy
}
