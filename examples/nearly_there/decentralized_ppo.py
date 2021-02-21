"""PPO agents for the Nearly There environment trained
decentrally
"""
from multiprocessing import Process, Lock, Barrier

import numpy as np
from stable_baselines3 import PPO

from bdgym.wrappers import MIAWrapper, SingleAgentWrapper
from bdgym.envs import NearlyThereEnv


MODEL_POLICY = "MlpPolicy"
MP_LOCK = Lock()
MP_BARRIER = Barrier(2)


def eval_agent(model, env, num_eps=10):
    """Evaluate model """
    ego_agent = "COACH" if env.ego_agent_id else "ATHLETE"

    with MP_LOCK:
        if env.ego_agent_id:
            print("Running Evaluation")
            print("="*60)

    ep_returns = []
    for e in range(num_eps):
        with MP_LOCK:
            if env.ego_agent_id:
                print(f"\nEpisode={e}\n{'-'*60}")

        MP_BARRIER.wait()
        ep_return = 0
        o = env.reset()
        done = False
        i = 0
        while not done:
            action, _ = model.predict(o, deterministic=True)
            o, reward, done, info = env.step(action)
            with MP_LOCK:
                print(
                    f"Agent={ego_agent}: Step={i}, a={action}, o={o}, "
                    f"r={reward}, done={done}, i={info}"
                )
            ep_return += reward
            i += 1

        ep_returns.append(ep_return)
        with MP_LOCK:
            if env.ego_agent_id:
                print(f"Episode={e} End. Return={ep_return}")

        MP_BARRIER.wait()

    with MP_LOCK:
        print(f"\n{'-'*60}{ego_agent} Training Complete{'-'*60}")
        print(
            f"Mean return = {np.mean(ep_returns):.4f} +/- "
            f"{np.std(ep_returns):.4f}"
        )


def train_agent(env: SingleAgentWrapper,
                eval_env: SingleAgentWrapper,
                model_kwargs: dict,
                learn_kwargs: dict):
    """independently train an agent """
    model = PPO(
        MODEL_POLICY,
        env,
        device='cpu',
        **model_kwargs
    )
    model.learn(
        eval_env=eval_env,
        **learn_kwargs
    )
    print("Training complete")
    # ensure child connections terminated correctly
    env.close()
    print("Training Env closed")

    eval_agent(model, eval_env)
    eval_env.close()
    print("Eval Env closed")
    return model


if __name__ == "__main__":
    ne_env = NearlyThereEnv()
    ne_env = MIAWrapper(ne_env, ne_env.num_agents)
    ne_eval_env = NearlyThereEnv()
    ne_eval_env = MIAWrapper(ne_eval_env, ne_env.num_agents)

    athlete_env = ne_env.get_agent_env(NearlyThereEnv.ATHLETE)
    coach_env = ne_env.get_agent_env(NearlyThereEnv.COACH)

    athlete_eval_env = ne_eval_env.get_agent_env(NearlyThereEnv.ATHLETE)
    coach_eval_env = ne_eval_env.get_agent_env(NearlyThereEnv.COACH)

    ppo_model_kwargs = {
        "verbose": 1,
        "tensorboard_log": "./results/"
    }

    learn_fn_kwargs = {
        "total_timesteps": 250000,
        "eval_freq": 2048,
        "n_eval_episodes": 100,
        "tb_log_name": "PPO_Athlete"
    }
    coach_learn_kwargs = dict(learn_fn_kwargs)
    coach_learn_kwargs['tb_log_name'] = "PPO_Coach"

    athlete_proc = Process(
        target=train_agent,
        args=(athlete_env, athlete_eval_env, ppo_model_kwargs, learn_fn_kwargs)
    )
    coach_proc = Process(
        target=train_agent,
        args=(coach_env, coach_eval_env, ppo_model_kwargs, coach_learn_kwargs)
    )
    athlete_proc.start()
    coach_proc.start()

    print("Joining agent processes")
    athlete_proc.join()
    coach_proc.join()

    print("Closing eval env")
    ne_eval_env.close()
    print("Closing main env")
    ne_env.close()
