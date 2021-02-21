"""PPO Implementation """
from pprint import pprint

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from bdgym.agents.utils import RLLogger
from bdgym.agents.ppo.buffer import PPOBuffer
from bdgym.agents.ppo.model import PPOActorCritic


class PPO:
    """PPO Implementation """

    def __init__(self,
                 env: gym.Env,
                 seed: float = None,
                 steps_per_epoch: int = 4000,
                 hidden_sizes: list = [32, 32],
                 clip_ratio: float = 0.2,
                 target_kl: float = 0.1,
                 train_actor_iters: int = 80,
                 train_critic_iters: int = 80,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.97,
                 actor_lr: float = 3e-4,
                 critic_lr: float = 1e-3):
        print("\nPPO with config:")
        pprint(locals())

        self.seed = seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.env = env
        self.env_name = env.spec.id
        self.num_actions = self.env.action_space.n
        self.obs_dim = self.env.observation_space.shape

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"Using device={self.device}")
        self.logger = RLLogger(self.env_name, "ppo")

        # Hyper params
        self.steps_per_epoch = steps_per_epoch
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl
        self.train_actor_iters = train_actor_iters
        self.train_critic_iters = train_critic_iters

        self.buffer = PPOBuffer(
            self.steps_per_epoch,
            self.obs_dim,
            gamma,
            gae_lambda,
            self.device
        )
        self.actor_critic = PPOActorCritic(
            self.obs_dim, hidden_sizes, self.num_actions
        )
        self.actor_critic.to(self.device)

        print("\nActorCritic:")
        print(self.actor_critic)

        self.actor_optimizer = optim.Adam(
            self.actor_critic.actor.parameters(), lr=actor_lr
        )
        self.critic_optimizer = optim.Adam(
            self.actor_critic.critic.parameters(), lr=critic_lr
        )
        self.critic_loss_fn = nn.MSELoss()

        self.steps_done = 0

    def _log(self, key, value):
        self.logger.add_scalar(key, value, self.steps_done)

    def get_action(self, obs):
        """Get action to perform for an observation """
        return self.actor_critic.act(obs)

    def _compute_actor_loss(self, data):
        obs, act, adv = data["obs"], data["act"], data["adv"]
        logp_old = data["logp"]

        pi, logp = self.actor_critic.actor(obs, act)
        ratio = torch.exp(logp - logp_old)
        clipped_ratio = torch.clamp(
            ratio, 1-self.clip_ratio, 1+self.clip_ratio
        )
        clip_adv = clipped_ratio * adv
        actor_loss = -(torch.min(ratio * adv, clip_adv)).mean()

        actor_loss_info = dict()
        actor_loss_info["kl"] = (logp_old - logp).mean().item()
        actor_loss_info["entropy"] = pi.entropy().mean().item()
        return actor_loss, actor_loss_info

    def _compute_critic_loss(self, data):
        obs, ret = data["obs"], data["ret"]
        predicted_val = self.actor_critic.critic(obs)
        return self.critic_loss_fn(predicted_val, ret)

    def optimize(self):
        """Optimize Actor and Critic with data in buffer """
        data = self.buffer.get()

        actor_loss_init, actor_loss_init_info = self._compute_actor_loss(data)
        actor_loss_start = actor_loss_init.item()
        critic_loss_start = self._compute_critic_loss(data).item()

        for _ in range(self.train_actor_iters):
            self.actor_optimizer.zero_grad()
            actor_loss, actor_loss_info = self._compute_actor_loss(data)
            if actor_loss_info["kl"] > 1.5*self.target_kl:
                break
            actor_loss.backward()
            self.actor_optimizer.step()

        for _ in range(self.train_critic_iters):
            self.critic_optimizer.zero_grad()
            critic_loss = self._compute_critic_loss(data)
            critic_loss.backward()
            self.critic_optimizer.step()

        # calculate changes in loss, for logging
        actor_loss_delta = (actor_loss.item() - actor_loss_start)
        critic_loss_delta = (critic_loss.item() - critic_loss_start)

        self._log("actor_loss", actor_loss_start)
        self._log("actor_loss_delta", actor_loss_delta)
        self._log("critic_loss", critic_loss_start)
        self._log("critic_loss_delta", critic_loss_delta)
        self._log("kl", actor_loss_init_info["kl"])
        self._log("entropy", actor_loss_init_info["entropy"])

    def train(self, epochs: int = 50):
        """Train the agent """
        print("PPO Starting training")

        for epoch in range(epochs):
            self._log("epoch", epoch)

            obs = self.env.reset()
            epoch_ep_rets = []
            epoch_ep_lens = []
            ep_ret, ep_len = 0, 0
            epoch_vals = []

            for t in range(self.steps_per_epoch):
                action, val, logp = self.actor_critic.step(
                    torch.from_numpy(obs).float().to(self.device)
                )
                next_obs, rew, done, _ = self.env.step(action)

                ep_len += 1
                ep_ret += rew
                epoch_vals.append(val)
                self.buffer.store(obs, action, rew, val, logp)
                obs = next_obs

                timeout = ep_len == self.env.spec.max_episode_steps
                terminal = timeout or done
                epoch_ended = t == self.steps_per_epoch-1

                if terminal or epoch_ended:
                    val = 0
                    if timeout or epoch_ended:
                        self.actor_critic.get_value(
                            torch.from_numpy(obs).float().to(self.device)
                        )
                    self.buffer.finish_path(val)

                if terminal:
                    epoch_ep_rets.append(ep_ret)
                    epoch_ep_lens.append(ep_len)
                    ep_ret, ep_len = 0, 0
                    obs = self.env.reset()

            # update the model
            self.optimize()

            # save model
            # if (epoch+1) % self.model_save_freq == 0:
            #     print(f"Epoch {epoch+1}: saving model")
            #     save_path = self.logger.get_save_path("pth")
            #     self.actor_critic.save_AC(save_path)

            self._log("avg_ep_return", np.mean(epoch_ep_rets))
            self._log("min_ep_return", np.min(epoch_ep_rets))
            self._log("max_ep_return", np.max(epoch_ep_rets))
            self._log("avg_vals", np.mean(epoch_vals))
            self._log("min_vals", np.min(epoch_vals))
            self._log("max_vals", np.max(epoch_vals))
            self._log("avg_ep_len", np.mean(epoch_ep_lens))

        print("PPO Training complete")
