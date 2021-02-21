"""The Exercise Assistant Environment. """
from typing import Tuple, Union, Dict

import gym
import numpy as np
from gym import spaces

import bdgym.envs.utils as utils
from bdgym.envs.exercise_assistant.action import AthleteAction
from bdgym.envs.exercise_assistant.graphics import EnvViewer


class ExerciseAssistantEnv(gym.Env):
    """The Exercise Assistant Environment

    Description:
        A multi-agent environment that involves an athlete performing an
        exercise and a exercise assistant that provides feedback to the
        athlete. Both agents share the common goal of the athlete completing
        as many reps of an exercise without the athlete over-exerting themself.

        An episode involves the athlete performing a fixed number of sets
        (MAX_SETS) of the exercise. After each set the athletes is complete
        their energy is replenished a random amount. The episode ends after all
        sets are complete or the athlete becomes over-exerted.

        Each agent alternates performing actions:
          1. The exercise-assistant, after observing athlete energy level,
             sends their signal to the athlete
          2. The athlete, after observing coaches signal, chooses action
             (either perform rep or end set)

        The key interesting aspect of this environment is the asymmetry of
        information and control. The Exercise Assistant has perfect knowledge
        of the current state, i.e. via access to advanced sensors, while the
        athlete has only noisy observations of the current state. Meanwhile,
        the athlete has full direct control over the state of the environment,
        while the exercise assistant can only control the state via the signals
        it sends to the athlete. This creates an interesting dynamic which
        depends a lot on how much the athlete trusts their percieved energy
        level versus what the exercise assistant is recommending.

    State:
        Athlete Energy Level: float in [0.0, 1.0]
            The proportion of energy the athlete has remaining, if this
            reaches 0.0, the athlete is overexerted.
        Sets Completed: int in [0, MAX_SETS]
            The number of sets completed.

    Starting State:
        Athlete Energy Level = uniform from [0.75, 1.0]
             I.e. the athlete starts with energy level sampled uniformly at
             random from [0.75, 1.0]
        Sets Completed = 0

    Transition:
        Exercise Assistant:
            The state is unchanged independent of the exercise-assistant
            action.

        Athlete:
            Perform Rep - athlete energy level is decreased by random amount
                          sampled from exponential distribution with
                          scale=0.05
            End Set - number of sets completed is increased by 1 and the
                      athletes energy level is increased by a random amount
                      (capped at energy=1.0) drawn from an exponential
                      distribution with scale=0.5

    Reward:
        1.0 for each rep performed, 0.0 for ending set, -1.0 for over-exertion

    Termination:
        If Sets Completed == Max_SETS or Athlete Energy Level <= 0.0


    Athlete Properties
    ------------------
    Observation:
        Type: Box(4)
        Num   Observation                           Min       Max
        0     Percieved Energy Level                0.0       1.0
        1     Proportion of sets complete           0.0       1.0
        2     Exercise Assistant Energy Signal      0.0       1.0
        3     Exercise Assistant Recommendation     0.0       1.0

        Percieved Energy Level Observation is noisy with the difference from
        the true level randomly sampled from a normal distribution with
        mean=ATHLETE_OBS_NOISE_MEAN and variance=ATHLETE_OBS_NOISE_VAR

        Proportion of sets complete is fully observed.

        Observations #2 and #3 are the actions of the exercise assistant.

    Actions:
        Type: Discrete(2)
        Num   Action
        0     Perform Rep
        1     End Set

    Assistant Properties
    --------------------
    Observation:
        Type: Box(2)
        Num   Observation                           Min       Max
        0     Athlete Energy Level                  0.0       1.0
        1     Proportion of sets complete           0.0       1.0
        2     Athlete Action                        0.0       1.0

        The exercise assistant recieves a perfect observation of the athlete
        energy level, the proportion of sets complete along with the last
        action completed by the athlete (0.0 for PERFORM_REP and 1.0 for
        END_SET).

    Actions:
        Type: Box(2)
        Num   Action                                Min       Max
        0     Energy Signal                         -1.0       1.0
        1     Recommendation                        -1.0       1.0

        Energy Signal represents what the exercise assistant is communicating
        to the athlete about the athletes energy level.

        Recommendation corresponds to what the exercise assistant recommends
        the athlete should do: -1.0=perform rep, 1.0=end set. It can be thought
        of as the probability that the athlete should end the set.

        Note, actions are linearly mapped from [-1.0, 1.0] to [0.0, 1.0] inside
        the environment (i.e. a recommendation action of 0.0, would correspond
        to sending the recommendation 0.5 to the athlete). We use the range
        [-1.0, 1.0] for better default behaviour with Deep RL algorothms.
    """

    metadata = {
        'render.modes': ['human']
    }

    MAX_SETS = 5
    """Maximum number of sets the athlete can complete """

    INIT_ENERGY_RANGE = [0.75, 1.0]
    """Range of initial energy distribution """

    ENERGY_COST_SCALE = 0.05
    """Scale of rep energy cost exponential distribution """

    ENERGY_RECOVERY_SCALE = 0.5
    """Scale of end of set energy recovery exponential distribution """

    ATHLETE_OBS_NOISE_MEAN = 0.05
    """Mean of athletes percieved energy level obs noise normal dist """

    ATHLETE_OBS_NOISE_VAR = 0.05
    """Variance of athletes percieved energy level obs noise normal dist """

    ASSISTANT_IDX = 0
    """Index of coach agent (i.e. in turn and action and obs spaces) """

    ATHLETE_IDX = 1
    """Index of athlete agent (i.e. in turn and action and obs spaces) """

    NUM_AGENTS = 2
    """Number of agents in the environment """

    MIN_ENERGY = 0.0
    """Minimum athlete energy """

    MAX_ENERGY = 1.0
    """Maximum athlete energy """

    OVEREXERTION_REWARD = -1.0
    """Reward for athlete becoming overexerted """

    REP_REWARD = 1.0
    """Reward for athlete performing a rep """

    def __init__(self):
        self.action_space = {
            self.ASSISTANT_IDX: spaces.Box(low=0.0, high=1.0, shape=(2,)),
            self.ATHLETE_IDX: spaces.Discrete(len(AthleteAction))
        }

        self.observation_space = {
            self.ASSISTANT_IDX: spaces.Box(low=0.0, high=1.0, shape=(3,)),
            self.ATHLETE_IDX: spaces.Box(low=0.0, high=1.0, shape=(4,))
        }

        self.state = [self.MAX_ENERGY, 0]
        self._last_obs = []
        self._last_action = []
        self._last_reward = 0.0
        self.next_agent = self.ASSISTANT_IDX

        # Rendering
        self.viewer = None

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        """Reset the environment

        Returns
        -------
        np.ndarray
            initial observation for each exercise assistant
        np.ndarray
            initial observation for athlete
        """
        init_energy = np.random.uniform(*self.INIT_ENERGY_RANGE)
        self.state = [init_energy, 0]
        assistant_obs = np.array([init_energy, 0.0, 0.0], dtype=np.float32)
        athlete_energy_obs = self._apply_athlete_noise(init_energy)
        athlete_obs = np.array(
            [athlete_energy_obs, 0.0, athlete_energy_obs, 0.0],
            dtype=np.float32
        )
        self._last_obs = [assistant_obs, athlete_obs]
        self._last_action = [None, None]
        self._last_reward = 0.0
        self.next_agent = self.ASSISTANT_IDX
        return assistant_obs, athlete_obs

    def step(self,
             action: Union[np.ndarray, int]
             ) -> Tuple[np.ndarray, float, bool, Dict]:
        """Perform a single step in environment

        NOTE: this will perform the action for the next agent in line to
        perform an action as specified by self.next_agent.

        The step function also returns the observation for the next agent after
        the current one performing the action. I.e. when the athlete performs
        an action the observation returned is for the exercise assistant and
        vice versa.

        Parameters
        ----------
        action: Union[np.ndarray, int]
            the action to be performed

        Returns
        -------
        np.ndarray
            observation for following agent
        float
            reward for following agent
        bool
            whether episode has terminated
        dict
            auxiliary information
        """
        self._validate_action(action)

        if self.next_agent == self.ATHLETE_IDX:
            self.state = self._perform_athlete_action(action)
            obs = self._get_assistant_obs(action)
        else:
            action = self._lmap_assistant_action(action)
            obs = self._get_athlete_obs(action)

        reward = self._get_reward(action)
        done = self._get_done()
        info = self._get_info()

        self._last_action[self.next_agent] = action
        self.next_agent = (self.next_agent + 1) % (self.ATHLETE_IDX+1)
        self._last_obs[self.next_agent] = obs
        self._last_reward = reward
        return obs, reward, done, info

    def render(self, mode: str = 'human'):
        """Render the environment

        Parameters
        ----------
        mode : str
            render mode to use
        """
        assert mode in self.metadata['render.modes']

        if self.viewer is None:
            self.viewer = EnvViewer(self)

        self.viewer.display()

        print(
            f"State: Athlete-Energy={self.state[0]:.4f} "
            f"Sets-Complete={self.state[1]:.4f}"
        )

    def is_terminal(self, state: Tuple[float, int] = None) -> bool:
        """Check if state is terminal

        Parameters
        ----------
        state : Tuple[float, int], optional
            the state to check. If None will check current state defined in
            self.state

        Returns
        -------
        bool
            True if current state is terminal, otherwise False
        """
        return self._get_done(state)

    def is_athletes_turn(self) -> bool:
        """Return if its the athletes turn or not

        Returns
        -------
        bool
            True if the next action to be performed is the athlete's, or False
            if it's the assistant's action next
        """
        return self.next_agent == self.ATHLETE_IDX

    def athlete_performed_rep(self) -> bool:
        """Returns if the last action by athlete was to perform a rep

        Returns
        -------
        bool
            True if last athlete action was PERFORM_REP, otherwise False (i.e.
            if it was END_SET or athlete is yet to perform an action)
        """
        return self._last_action[self.ATHLETE_IDX] == AthleteAction.PERFORM_REP

    @property
    def set_count(self) -> int:
        """The current set number """
        return self.state[1]

    @property
    def last_assistant_obs(self) -> np.ndarray:
        """The last observation of the assistant """
        return self._last_obs[self.ASSISTANT_IDX]

    @property
    def last_athlete_obs(self) -> np.ndarray:
        """The last observation of the athlete """
        return self._last_obs[self.ATHLETE_IDX]

    def discrete_assistant(self) -> bool:
        """Check if assistant is using discrete actions """
        return isinstance(
            self.action_space[self.ASSISTANT_IDX], spaces.Discrete
        )

    def _validate_action(self, action: Union[np.ndarray, int]):
        if self.next_agent == self.ASSISTANT_IDX:
            assert isinstance(action, np.ndarray), \
                (f"Exercise Assistant action must be a np.ndarray. '{action}'"
                 " invalid.")
            assert action.shape == (2,), \
                ("Exercise assistant action shape invalid. "
                 f"{action.shape} != (2,).")
        else:
            if isinstance(action, np.ndarray):
                assert len(action) == 1, \
                    (f"If athlete is np.ndarray it should be of length 1. "
                     f"'{action}' invalid.")
                action = action[0]
            assert isinstance(action, (int, AthleteAction, np.integer)), \
                f"Athlete action must be integer. '{action}' invalid."
            assert 0 <= action < len(AthleteAction), \
                f"Athlete action must be either 0 or 1. '{action}' invalid."

    def _perform_athlete_action(self, action: int) -> Tuple[float, int]:
        if action == AthleteAction.PERFORM_REP:
            new_energy = self._apply_rep_cost(self.state[0])
            sets_done = self.state[1]
        else:
            new_energy = self._apply_recovery(self.state[0])
            sets_done = self.state[1] + 1
        return (new_energy, sets_done)

    def _lmap_assistant_action(self, action: np.ndarray) -> np.ndarray:
        action = np.clip(action, -1.0, 1.0)
        return utils.lmap_array(
            action, [-1, 1.0], [self.MIN_ENERGY, self.MAX_ENERGY]
        )

    def _get_athlete_obs(self, action: np.ndarray) -> np.ndarray:
        sets_completed = self.state[1] / self.MAX_SETS
        percieved_energy = self._apply_athlete_noise(self.state[0])
        return np.array(
            [percieved_energy, sets_completed, action[0], action[1]],
            dtype=np.float32
        )

    def _get_assistant_obs(self, action: int) -> np.ndarray:
        sets_completed = self.state[1] / self.MAX_SETS
        return np.array(
            [self.state[0], sets_completed, action], dtype=np.float32
        )

    def _get_reward(self, action: Union[np.ndarray, int]) -> float:
        if self.next_agent == self.ATHLETE_IDX:
            reward = 0.0
            if self.state[0] <= self.MIN_ENERGY:
                reward = self.OVEREXERTION_REWARD
            elif action == AthleteAction.PERFORM_REP:
                reward = self.REP_REWARD
        else:
            reward = self._last_reward
        return reward

    def _get_done(self, state: Tuple[float, int] = None) -> bool:
        if state is None:
            state = self.state
        return state[0] <= self.MIN_ENERGY or state[1] >= self.MAX_SETS

    def _get_info(self):
        state_info = {
            "athlete energy remaining": self.state[0],
            "sets complete": self.state[1]
        }
        return state_info

    def _apply_athlete_noise(self, energy: float) -> float:
        noise = np.random.normal(
            self.ATHLETE_OBS_NOISE_MEAN, self.ATHLETE_OBS_NOISE_VAR
        )
        return max(self.MIN_ENERGY, min(self.MAX_ENERGY, energy + noise))

    def _apply_rep_cost(self, energy: float) -> float:
        cost = np.random.exponential(self.ENERGY_COST_SCALE)
        return max(self.MIN_ENERGY, min(self.MAX_ENERGY, energy - cost))

    def _apply_recovery(self, energy: float) -> float:
        recovery = np.random.exponential(self.ENERGY_RECOVERY_SCALE)
        return max(self.MIN_ENERGY, min(self.MAX_ENERGY, energy + recovery))
