"""PPO Implementation using Stable Baselines3 """
import time

import gym
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common import logger
from stable_baselines3.common.utils import safe_mean


class MAPPO(PPO):
    """A multi-agent env compatible PPO interface

    To get action for given observation use:

        model.predict(observation)

    To train the model using stored data use:

        model.learn_step()
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._callback = None

    def setup_learn(self, total_timesteps: int):
        """Initialize different variables needed for training """
        _, self._callback = self._setup_learn(total_timesteps)

    def end_learn(self):
        """Finilize learning """
        self._callback.on_training_end()

    def init_epoch(self):
        """Initialize agent before start of new epoch

        Adapted from the OnPolicyAlgorithm.collect_rollout() function in
        stable_baselines3.common.on_policy_algorithm module.
        """
        self.rollout_buffer.reset()
        if self.use_sde:
            self.policy.reset_noise(self.env.num_envs)

    def get_step(self, n_steps: int):
        """Get action, etc for next step

        Adapted from the OnPolicyAlgorithm.collect_rollout() function in
        stable_baselines3.common.on_policy_algorithm module.
        """
        if self.use_sde and self.sde_sample_freq > 0 \
           and n_steps % self.sde_sample_freq == 0:
            self.policy.reset_noise(self.env.num_envs)

        with torch.no_grad():
            obs_tensor = torch.as_tensor(self._last_obs).to(self.device)
            actions, values, log_probs = self.policy.forward(obs_tensor)
        actions = actions.cpu().numpy()

        # Rescale and perform action
        clipped_actions = actions
        # Clip the actions to avoid out of bound error
        if isinstance(self.action_space, gym.spaces.Box):
            clipped_actions = np.clip(
                actions, self.action_space.low, self.action_space.high
            )

        return clipped_actions, values, log_probs

    def store_step(self,
                   new_obs,
                   actions,
                   rewards,
                   dones,
                   infos,
                   values,
                   log_probs):
        """Store a step

        Adapted from the OnPolicyAlgorithm.collect_rollout() function in
        stable_baselines3.common.on_policy_algorithm module.
        """
        self.num_timesteps += self.env.num_envs

        self._callback.update_locals(locals())
        # if self._callback.on_step() is False:
        #     return False

        self._update_info_buffer(infos)
        if isinstance(self.action_space, gym.spaces.Discrete):
            actions = actions.reshape(-1, 1)

        self.rollout_buffer.add(
            self._last_obs,
            actions,
            rewards,
            self._last_dones,
            values,
            log_probs
        )
        self._last_obs = new_obs
        self._last_dones = dones

    def finish_rollout(self, new_obs, dones):
        """Finish an episode, computing value for the last timestep """
        with torch.no_grad():
            # Compute value for the last timestep
            obs_tensor = torch.as_tensor(new_obs).to(self.device)
            _, values, _ = self.policy.forward(obs_tensor)

        self.rollout_buffer.compute_returns_and_advantage(
            last_values=values, dones=dones
        )
        self._callback.on_rollout_end()

    def learn_step(self, epoch_num: int, log_step: bool = True):
        """Perform a learning step using stored rollouts

        Adapted from the OnPolicyAlgorithm.learn() function in
        stable_baselines3.common.on_policy_algorithm module.
        """
        self._update_current_progress_remaining(
            self.num_timesteps, self._total_timesteps
        )

        # Display training infos
        if log_step:
            fps = int(self.num_timesteps / (time.time() - self.start_time))
            logger.record("time/iterations", epoch_num, exclude="tensorboard")

            if len(self.ep_info_buffer) > 0 \
               and len(self.ep_info_buffer[0]) > 0:
                logger.record(
                    "rollout/ep_rew_mean",
                    safe_mean(
                        [ep_info["r"] for ep_info in self.ep_info_buffer]
                    )
                )
                logger.record(
                    "rollout/ep_len_mean",
                    safe_mean(
                        [ep_info["l"] for ep_info in self.ep_info_buffer]
                    )
                )
            logger.record("time/fps", fps)
            logger.record(
                "time/time_elapsed",
                int(time.time() - self.start_time),
                exclude="tensorboard"
            )
            logger.record(
                "time/total_timesteps",
                self.num_timesteps,
                exclude="tensorboard"
            )
            logger.dump(step=self.num_timesteps)

        self.train()
