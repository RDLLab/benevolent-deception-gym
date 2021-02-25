"""Script for running evaluation of PPO versus fixed athlete policies """
import time
import os.path as osp

import numpy as np

from stable_baselines3 import PPO

import bdgym.scripts.drl_utils as drl_utils
import bdgym.envs.exercise_assistant as ea_env
from bdgym.scripts.script_utils import create_dir
import bdgym.envs.exercise_assistant.policy as policy
import bdgym.scripts.exercise_assistant.utils as utils


EVAL_RESULT_DIR = create_dir(
    osp.join(utils.RESULTS_DIR, "sb3_ppo_random_fixed_athlete_perf"),
    make_new=True
)
EVAL_RESULTS_FILENAME = osp.join(EVAL_RESULT_DIR, "eval_results.tsv")

print("EVAL_RESULT_DIR:", str(EVAL_RESULT_DIR))

# PERCEPT_INFLUENCES = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
# INDEPENDENCES = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
# ATHLETE_POLICY = 'weighted'
PERCEPT_INFLUENCES = [0.0]
INDEPENDENCES = [0.0]
ATHLETE_POLICY = 'random_weighted'
NUM_EPISODES = 100
SEEDS = list(range(10))
VERBOSITY = 0
RENDER = ""
MANUAL = False
DISCRETE = True
NUM_CPUS = 1

# PPO Parameters
POLICY = "MlpPolicy"
TOTAL_TIMESTEPS = 100000
SAVE_FREQ = -1
BATCH_STEPS = 512
EVAL_FREQ = 5000
SAVE_BEST = True
N_EVAL_EPISODES = 30
N_FINAL_EVAL_EPISODES = 100
PPO_KWARGS = {
    "gamma": 0.999, "ent_coef": 0.1
}


def get_config_env(independence: float,
                   perception_influence: float,
                   seed: int = 0) -> ea_env.FixedAthleteExerciseAssistantEnv:
    """Get the configured Fixed Athlete Exercise Assistant Env """
    if ATHLETE_POLICY == 'weighted':
        athlete_policy = policy.WeightedAthletePolicy(
            perception_influence=perception_influence,
            independence=independence
        )
    elif ATHLETE_POLICY == 'random_weighted':
        athlete_policy = policy.RandomWeightedAthletePolicy()

    env = ea_env.DiscreteFixedAthleteExerciseAssistantEnv(athlete_policy)
    env.seed(seed)
    return env


def get_env_creation_fn(independence: float,
                        perception_influence: float):
    """Get the get_configured_env_fn """
    def get_config_env_fn(seed):
        return get_config_env(independence, perception_influence, seed)
    return get_config_env_fn


def run_eval_episode(ppo_model, env):
    """Run a single eval episode """
    obs = env.reset()
    if RENDER != "":
        env.render()

    done = False
    total_return = 0
    steps = 0
    while not done:
        action, _ = ppo_model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        total_return += reward
        if RENDER != "":
            env.render()
        steps += 1

    mean_deception = np.mean(env.assistant_deception)
    return total_return, steps, env.athlete_overexerted(), mean_deception


def eval_best(ppo_model,
              eval_env,
              independence: float,
              perception_influence: float,
              seed: int) -> utils.Result:
    """Evaluate the best model """
    ep_returns = []
    ep_steps = []
    ep_overexertions = []
    ep_times = []
    ep_deceptions = []
    for _ in range(N_FINAL_EVAL_EPISODES):
        start_time = time.time()
        total_reward, steps, overexerted, deception = run_eval_episode(
            ppo_model, eval_env
        )
        ep_times.append(time.time() - start_time)
        ep_returns.append(total_reward)
        ep_steps.append(steps)
        ep_overexertions.append(overexerted)
        ep_deceptions.append(deception)

    result = utils.Result(
        assistant="PPO",
        athlete=ATHLETE_POLICY,
        independence=independence,
        perception_influence=perception_influence,
        episodes=N_EVAL_EPISODES,
        seed=seed,
        return_mean=np.mean(ep_returns),
        return_std=np.std(ep_returns),
        steps_mean=np.mean(ep_steps),
        steps_std=np.std(ep_steps),
        overexertion_prob=np.mean(ep_overexertions),
        time_mean=np.mean(ep_times),
        time_std=np.std(ep_times),
        deception_mean=np.mean(ep_deceptions),
        deception_std=np.std(ep_deceptions)
    )

    if VERBOSITY > 0:
        utils.display_result(result)

    return result


def perform_run(independence: float,
                perception_influence: float,
                seed: int) -> utils.Result:
    """Perform a single run """
    get_config_env_fn = get_env_creation_fn(independence, perception_influence)
    env = drl_utils.get_env(get_config_env_fn, False, NUM_CPUS, seed)

    ppo_model = drl_utils.init_ppo_model(
        env,
        batch_steps=BATCH_STEPS,
        verbosity=VERBOSITY,
        result_dir=EVAL_RESULT_DIR,
        policy=POLICY,
        **PPO_KWARGS
    )

    if ATHLETE_POLICY == 'weighted':
        log_name = (
            f"eval_i{independence:.3f}_p{perception_influence:.3f}_s{seed}"
        )
    elif ATHLETE_POLICY == 'random_weighted':
        log_name = f"eval_random_weighted_s{seed}"
    else:
        log_name = f"eval_{ATHLETE_POLICY}_s{seed}"
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
        best_model, eval_env, independence, perception_influence, seed
    )
    return result


def main():
    """Run the evaluation """
    num_runs = len(INDEPENDENCES) * len(PERCEPT_INFLUENCES) * len(SEEDS)
    count = 1
    for i in INDEPENDENCES:
        for p in PERCEPT_INFLUENCES:
            for s in SEEDS:
                print(
                    f"Performing run {count} / {num_runs} with "
                    f"independence={i:.3f} "
                    f"perception_influence={p:.3f} "
                    f"seed={s}"
                )
                eval_result = perform_run(i, p, s)
                utils.append_result_to_file(
                    eval_result, EVAL_RESULTS_FILENAME, True
                )
                count += 1


if __name__ == "__main__":
    main()
