"""Script for training and evaluating PPO versus driver policies """
import time
import os.path as osp
from typing import Tuple
from pprint import pprint

import numpy as np

from stable_baselines3 import PPO

import bdgym.scripts.drl_utils as drl_utils
import bdgym.envs.driver_assistant as da_env
from bdgym.scripts.script_utils import create_dir
import bdgym.scripts.script_utils as script_utils
import bdgym.scripts.driver_assistant.utils as utils
from bdgym.envs.driver_assistant.driver_types import get_driver_config


EVAL_RESULT_DIR = create_dir(
    osp.join(utils.RESULTS_DIR, "sb3_ppo_changing_driver_perf"),
    make_new=True
)
EVAL_RESULTS_FILENAME = osp.join(EVAL_RESULT_DIR, "eval_results.tsv")

print("EVAL_RESULT_DIR:", str(EVAL_RESULT_DIR))


INDEPENDENCES = [0.0]
DRIVER_POLICIES = ['GuidedIDMDriverPolicy']
NUM_EPISODES = 100
SEEDS = list(range(1))
VERBOSITY = 1
RENDER = False
MANUAL = False
DISCRETE = True
FORCE_INDEPENDENT = False
NUM_CPUS = 1

# PPO Parameters
POLICY = "MlpPolicy"
TOTAL_TIMESTEPS = 500000
SAVE_FREQ = -1
BATCH_STEPS = 2048
EVAL_FREQ = BATCH_STEPS*5
SAVE_BEST = True
N_EVAL_EPISODES = 10
N_FINAL_EVAL_EPISODES = 100
PPO_KWARGS = {
    "gamma": 0.999, "ent_coef": 0.1
}


def get_env_creation_fn(independence: float,
                        driver_type: str):
    """Get the get_configured_env_fn """
    def get_config_env_fn(seed):
        return utils.get_configured_env(
            independence,
            driver_type,
            FORCE_INDEPENDENT,
            DISCRETE,
            VERBOSITY,
            seed
        )
    return get_config_env_fn


def run_eval_episode(ppo_model: PPO,
                     env: da_env.FixedDriverDriverAssistantEnv
                     ) -> Tuple[float, int, bool, float]:
    """Run a single eval episode """
    obs = env.reset()
    if RENDER:
        env.render()

    done = False
    total_return = 0
    steps = 0
    while not done:
        action, _ = ppo_model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        total_return += reward
        if RENDER:
            env.render()
        steps += 1

    mean_deception = np.mean(env.assistant_deception, axis=0)
    collision = steps < env.config["duration"]
    return total_return, steps, collision, mean_deception


def eval_best(ppo_model,
              eval_env,
              independence: float,
              driver_type: str,
              seed: int) -> utils.Result:
    """Evaluate the best model """
    ep_returns = []
    ep_steps = []
    ep_collisions = []
    ep_times = []
    ep_deceptions = []
    for _ in range(N_FINAL_EVAL_EPISODES):
        start_time = time.time()
        total_reward, steps, collision, deception = run_eval_episode(
            ppo_model, eval_env
        )
        ep_times.append(time.time() - start_time)
        ep_returns.append(total_reward)
        ep_steps.append(steps)
        ep_collisions.append(collision)
        ep_deceptions.append(deception)

    result = utils.Result(
        assistant="PPO",
        driver=driver_type,
        independence=independence,
        episodes=N_EVAL_EPISODES,
        seed=seed,
        return_mean=np.mean(ep_returns),
        return_std=np.std(ep_returns),
        steps_mean=np.mean(ep_steps),
        steps_std=np.std(ep_steps),
        collision_prob=np.mean(ep_collisions),
        time_mean=np.mean(ep_times),
        time_std=np.std(ep_times),
        deception_mean=np.mean(ep_deceptions),
        deception_std=np.std(ep_deceptions)
    )

    if VERBOSITY > 0:
        utils.display_result(result)

    return result


def perform_run(independence: float,
                driver_type: str,
                seed: int) -> utils.Result:
    """Perform a single run """
    get_config_env_fn = get_env_creation_fn(independence, driver_type)
    env = drl_utils.get_env(get_config_env_fn, False, NUM_CPUS, seed)

    ppo_model = drl_utils.init_ppo_model(
        env,
        batch_steps=BATCH_STEPS,
        verbosity=VERBOSITY,
        result_dir=EVAL_RESULT_DIR,
        policy=POLICY,
        **PPO_KWARGS
    )

    if driver_type == 'GuidedIDMDriverPolicy':
        log_name = (
            f"eval_GuidedIDMDriverPolicy_i{independence:.3f}_s{seed}"
        )
    elif driver_type == 'changing':
        log_name = f"eval_changing_s{seed}"
    else:
        log_name = f"eval_{driver_type}_s{seed}"
    result_dir = osp.join(EVAL_RESULT_DIR, log_name)

    eval_env = drl_utils.get_env(get_config_env_fn, True)
    drl_utils.train_model(
        ppo_model,
        total_timesteps=TOTAL_TIMESTEPS,
        eval_env=eval_env,
        save_frequency=SAVE_FREQ,
        eval_freq=EVAL_FREQ,
        reset_num_timesteps=False,
        log_name=log_name,
        save_best=SAVE_BEST,
        result_dir=result_dir,
        n_eval_episodes=N_EVAL_EPISODES
    )

    eval_env = drl_utils.get_env(get_config_env_fn, True)
    best_model = drl_utils.load_best_model(PPO, result_dir, eval_env)
    result = eval_best(
        best_model, eval_env, independence, driver_type, seed
    )
    return result


def main():
    """Run the evaluation """
    num_runs = len(INDEPENDENCES) * len(DRIVER_POLICIES) * len(SEEDS)
    count = 1
    for i in INDEPENDENCES:
        for d in DRIVER_POLICIES:
            for s in SEEDS:
                print(
                    f"Performing run {count} / {num_runs} with "
                    f"independence={i:.3f} "
                    f"driver_type={d} "
                    f"seed={s}"
                )
                eval_result = perform_run(i, d, s)
                script_utils.append_result_to_file(
                    eval_result, EVAL_RESULTS_FILENAME, True
                )
                count += 1


if __name__ == "__main__":
    main()
