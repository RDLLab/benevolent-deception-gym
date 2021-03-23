"""The Exercise Assistant Environment. """
from typing import Tuple, Union, Dict, List, Optional

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
        exercise and a assistant that provides feedback to the
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
        information and control. The Assistant has perfect knowledge
        of the current state, i.e. via access to advanced sensors, while the
        athlete has only noisy observations of the current state. Meanwhile,
        the athlete has full direct control over the state of the environment,
        while the assistant can only control the state via the signals
        it sends to the athlete. This creates an interesting dynamic which
        depends a lot on how much the athlete trusts their percieved energy
        level versus what the assistant is recommending.

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
        Assistant:
            The state is unchanged independent of the assistant action.

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
        2     Assistant Energy Signal               0.0       1.0
        3     Assistant Recommendation              0.0       1.0

        Percieved Energy Level Observation is noisy with the difference from
        the true level randomly sampled from a normal distribution with
        mean=ATHLETE_OBS_NOISE_MEAN and variance=ATHLETE_OBS_NOISE_VAR

        Proportion of sets complete is fully observed.

        Observations #2 and #3 are the actions of the assistant.

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

        The assistant recieves a perfect observation of the athlete
        energy level, the proportion of sets complete along with the last
        action completed by the athlete (0.0 for PERFORM_REP and 1.0 for
        END_SET).

    Actions:
        Type: Box(2)
        Num   Action                                Min       Max
        0     Energy Signal                         -1.0       1.0
        1     Recommendation                        -1.0       1.0

        Energy Signal represents what the assistant is communicating
        to the athlete about the athletes energy level.

        Recommendation corresponds to what the assistant recommends the athlete
        should do: -1.0=perform rep, 1.0=end set. It can be thought of as the
        probability that the athlete should end the set.

        Note, actions are linearly mapped from [-1.0, 1.0] to [0.0, 1.0] inside
        the environment (i.e. a recommendation action of 0.0, would correspond
        to sending the recommendation 0.5 to the athlete). We use the range
        [-1.0, 1.0] for better default behaviour with Deep RL algorothms.
    """

    metadata = {
        'render.modes': ['human', 'asci']
    }

    MAX_SETS = 10
    """Maximum number of sets the athlete can complete """

    INIT_ENERGY_RANGE = [0.75, 1.0]
    """Range of initial energy distribution """

    ENERGY_COST_SCALE = 0.1
    """Scale of rep energy cost exponential distribution """

    ENERGY_COST_BOUNDS = [0.05, 0.15]
    """Bound for the energy cost of a single rep """

    ENERGY_RECOVERY_SCALE = 0.5
    """Scale of end of set energy recovery exponential distribution """

    ENERGY_RECOVERY_BOUNDS = [0.25, 0.75]
    """Bound for end of set energy recovery """

    ATHLETE_OBS_NOISE_MEAN = 0.05
    """Mean of athletes percieved energy level obs noise normal dist """

    ATHLETE_OBS_NOISE_VAR = 0.05
    """Variance of athletes percieved energy level obs noise normal dist """

    ATHLETE_OBS_NOISE_BOUNDS = [-0.15, 0.15]
    """Bounds of athletes percieved energy level obs noise distribution """

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

    def __init__(self,
                 render_assistant_info: bool = True,
                 render_athlete_info: bool = True,
                 save_images: bool = False,
                 image_save_directory: Optional[str] = None):
        self.render_assistant_info = render_assistant_info
        self.render_athlete_info = render_athlete_info
        self.save_images = save_images
        self.image_save_directory = image_save_directory

        self.action_space = {
            self.ASSISTANT_IDX: spaces.Box(
                np.float32(0.0), np.float32(1.0), shape=(2,)
            ),
            self.ATHLETE_IDX: spaces.Discrete(len(AthleteAction))
        }

        self.observation_space = {
            self.ASSISTANT_IDX: spaces.Box(
                np.float32(0.0), np.float32(1.0), shape=(3,)
            ),
            self.ATHLETE_IDX: spaces.Box(
                np.float32(0.0), np.float32(1.0), shape=(4,)
            )
        }

        self.state = (self.MAX_ENERGY, 0)
        self._last_obs: List[np.ndarray] = []
        self._last_action: List[Union[int, np.ndarray]] = []
        self._last_reward = 0.0
        self.assistant_deception: List[float] = []
        self.next_agent = self.ASSISTANT_IDX

        # Rendering
        self.viewer: Optional[EnvViewer] = None

        # Reset to fill last obs_config
        self.reset()

    def reset(self) -> np.ndarray:
        """Reset the environment

        Returns
        -------
        np.ndarray
            initial observation for the assistant
        """
        init_energy = np.random.uniform(*self.INIT_ENERGY_RANGE)
        self.state = (init_energy, 0)
        assistant_obs = np.array([init_energy, 0.0, 0.0], dtype=np.float32)
        athlete_obs_noise = utils.get_truncated_normal(
            self.ATHLETE_OBS_NOISE_MEAN,
            self.ATHLETE_OBS_NOISE_VAR,
            *self.ATHLETE_OBS_NOISE_BOUNDS
        ).rvs()
        athlete_energy_obs = np.clip(
            init_energy + athlete_obs_noise, self.MIN_ENERGY, self.MAX_ENERGY
        )

        athlete_obs = np.array(
            [athlete_energy_obs, 0.0, athlete_energy_obs, 0.0],
            dtype=np.float32
        )
        self._last_obs = [assistant_obs, athlete_obs]
        self._last_action = [None, None]
        self._last_reward = 0.0
        self.assistant_deception = []
        self.next_agent = self.ASSISTANT_IDX
        return assistant_obs

    def step(self,
             action: Union[np.ndarray, int]
             ) -> Tuple[np.ndarray, float, bool, Dict]:
        """Perform a single step in environment

        NOTE: this will perform the action for the next agent in line to
        perform an action as specified by self.next_agent.

        The step function also returns the observation for the next agent after
        the current one performing the action. I.e. when the athlete performs
        an action the observation returned is for the assistant and vice versa.

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
            self._track_deception(action)
            obs = self._get_athlete_obs(action)

        reward = self._get_reward(action)
        done = self._get_done()
        info = self._get_info()

        self._last_action[self.next_agent] = action
        self.next_agent = (self.next_agent + 1) % (self.ATHLETE_IDX+1)
        self._last_obs[self.next_agent] = obs
        self._last_reward = reward
        return obs, reward, done, info

    def _validate_action(self, action: Union[np.ndarray, int]):
        if self.next_agent == self.ASSISTANT_IDX:
            assert isinstance(action, np.ndarray), \
                (f"Assistant action must be a np.ndarray. '{action}' invalid.")
            assert action.shape == (2,), \
                ("Assistant action shape invalid. {action.shape} != (2,).")
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
            cost = np.random.exponential(self.ENERGY_COST_SCALE)
            cost = np.clip(cost, *self.ENERGY_COST_BOUNDS)
            new_energy = np.clip(
                self.state[0] - cost, self.MIN_ENERGY, self.MAX_ENERGY
            )
            sets_done = self.state[1]
        else:
            recovery = np.random.exponential(self.ENERGY_RECOVERY_SCALE)
            recovery = np.clip(recovery, *self.ENERGY_RECOVERY_BOUNDS)
            new_energy = np.clip(
                self.state[0] + recovery, self.MIN_ENERGY, self.MAX_ENERGY
            )
            sets_done = self.state[1] + 1
        return (new_energy, sets_done)

    def _get_assistant_obs(self, action: int) -> np.ndarray:
        sets_completed = self.state[1] / self.MAX_SETS
        return np.array(
            [self.state[0], sets_completed, action], dtype=np.float32
        )

    def _lmap_assistant_action(self, action: np.ndarray) -> np.ndarray:
        action = np.clip(action, -1.0, 1.0)
        return utils.lmap_array(
            action, [-1, 1.0], [self.MIN_ENERGY, self.MAX_ENERGY]
        )

    def _track_deception(self, assistant_action: np.ndarray):
        # Per Step Deception is the difference between the energy level that
        # the assistant observed and the energy level it communicates with the
        # athlete.
        obs_energy = self._last_obs[self.ASSISTANT_IDX][0]
        reported_energy = assistant_action[0]
        deception = abs(obs_energy - reported_energy)
        self.assistant_deception.append(deception)

    def _get_athlete_obs(self, action: np.ndarray) -> np.ndarray:
        sets_completed = self.state[1] / self.MAX_SETS

        obs_noise = utils.get_truncated_normal(
            self.ATHLETE_OBS_NOISE_MEAN,
            self.ATHLETE_OBS_NOISE_VAR,
            *self.ATHLETE_OBS_NOISE_BOUNDS
        ).rvs()
        percieved_energy = np.clip(
            self.state[0] + obs_noise, self.MIN_ENERGY, self.MAX_ENERGY
        )
        if self.athlete_performed_rep():
            percieved_energy = min(
                self._last_obs[self.ATHLETE_IDX][0],
                percieved_energy
            )

        return np.array(
            [percieved_energy, sets_completed, action[0], action[1]],
            dtype=np.float32
        )

    def render(self, mode: str = 'human'):
        """Render the environment

        Parameters
        ----------
        mode : str
            render mode to use
        """
        assert mode in self.metadata['render.modes']

        if mode == 'human':
            if self.viewer is None:
                self.viewer = EnvViewer(
                    self,
                    save_images=self.save_images,
                    save_directory=self.image_save_directory
                )
            self.viewer.display()
        elif mode == 'asci':
            print(
                f"State: Athlete-Energy={self.state[0]:.4f} "
                f"Sets-Complete={self.state[1]:.4f}"
            )

    def close(self) -> None:
        """Close the environment.

        Will close the environment viewer if it exists.
        """
        if self.viewer is not None:
            self.viewer.close()
        self.viewer = None

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
            True if state is terminal, otherwise False
        """
        return self._get_done(state)

    def athlete_overexerted(self, state: Tuple[float, int] = None) -> bool:
        """Check if athlete is overexerted in state

        Parameters
        ----------
        state : Tuple[float, int], optional
            the state to check. If None will check current state defined in
            self.state

        Returns
        -------
        bool
            True if athlete is overexerted in state, otherwise False
        """
        if state is None:
            state = self.state
        return state[0] <= self.MIN_ENERGY

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

    @property
    def discrete_assistant(self) -> bool:
        """Check if assistant is using discrete actions """
        return isinstance(
            self.action_space[self.ASSISTANT_IDX], spaces.Discrete
        )

    @property
    def athlete_action_space(self) -> spaces.Space:
        """The athlete's action space """
        return self.action_space[self.ATHLETE_IDX]

    @property
    def assistant_action_space(self) -> spaces.Space:
        """The assistant's action space """
        return self.action_space[self.ASSISTANT_IDX]
