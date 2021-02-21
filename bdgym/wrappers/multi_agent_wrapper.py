"""A Wrapper to handle independent agents in a multi-agent environment.

This environment follows a worker-master architecture, where each worker
corresponds to a single-agent environment, one for each independent agent,
where agents interact with their version of the environment in the usual
single-agent OpenAI gym environment manner. The master waits for all
workers to call the same function (i.e. step or reset) before collecting
the actions of each agent (in case of step) and performing the function
in the environment and then returning the obs, etc to each agent
 """
from typing import List
from multiprocessing import Process, Pipe
from multiprocessing.connection import wait

import gym

from bdgym.wrappers.single_agent_wrapper import SingleAgentWrapper


class MIAWrapper(gym.Wrapper):
    """Multiple Independent agent gym wrapper """

    PARENT_PIPE = 0
    CHILD_PIPE = 1

    def __init__(self, env: gym.Env, num_agents: int):
        super().__init__(env)
        self.num_agents = num_agents

        self._step_pipes = []
        self._reset_pipes = []
        for _ in range(num_agents):
            self._step_pipes.append(Pipe())
            self._reset_pipes.append(Pipe())

        self._env_process = Process(
            target=self._env_handler,
            args=(env, self._reset_pipes, self._step_pipes)
        )
        self._env_process.start()

        # close child end of pipe in the parent process
        self._close_pipe_ends(
            [self._step_pipes, self._reset_pipes], self.CHILD_PIPE
        )

    def step_agent(self, action, agent_id: int):
        """Perform step for a single agent """
        step_conn = self._step_pipes[agent_id][self.PARENT_PIPE]
        step_conn.send(action)
        return step_conn.recv()

    def reset_agent(self, agent_id: int, **kwargs):
        """Single agent reset """
        reset_conn = self._reset_pipes[agent_id][self.PARENT_PIPE]
        reset_conn.send(kwargs)
        return reset_conn.recv()

    def get_agent_env(self, agent_id: int) -> SingleAgentWrapper:
        """Get single agent environment interface for given a agent """
        return SingleAgentWrapper(self, agent_id)

    def close(self):
        """Close the environment """
        self.close_conns()
        self._env_process.join()

    def _env_handler(self, env: gym.Env,
                     reset_pipes: List[Pipe],
                     step_pipes: List[Pipe]):
        """Handles interaction with a single version of environment """
        # close all unused pipe ends in the child process
        # to ensure process can detect when pipe is closed
        self._close_pipe_ends(
            [reset_pipes, step_pipes], self.PARENT_PIPE
        )

        reset_conns = [pipe[self.CHILD_PIPE] for pipe in reset_pipes]
        step_conns = [pipe[self.CHILD_PIPE] for pipe in step_pipes]
        conns = reset_conns + step_conns

        connection_ended = False
        reset_performed = False
        step_performed = False
        waiting_reset_conns = list(reset_conns)
        waiting_step_conns = list(step_conns)
        action_n = [None] * self.num_agents

        while not connection_ended:
            if reset_performed:
                waiting_reset_conns = list(reset_conns)
                reset_performed = False

            if step_performed:
                waiting_step_conns = list(step_conns)
                step_performed = False
                action_n = [None] * self.num_agents

            # wait for all agents to reset or step, whichever comes first
            while (
                    waiting_reset_conns
                    and waiting_step_conns
                    and not connection_ended
            ):

                # print("Waiting for connections")
                for conn in wait(conns):
                    try:
                        action = conn.recv()
                    except EOFError:
                        connection_ended = True
                        break
                    else:
                        if conn in waiting_reset_conns:
                            waiting_reset_conns.remove(conn)
                        else:
                            waiting_step_conns.remove(conn)
                            conn_idx = step_conns.index(conn)
                            action_n[conn_idx] = action

            if connection_ended:
                break

            if not waiting_reset_conns:
                self._perform_reset(env, reset_conns)
                reset_performed = True

            if not waiting_step_conns:
                self._perform_step(env, action_n, step_conns)
                step_performed = True

        # connection ended, so close this end of pipes
        self._close_pipe_ends(
            [reset_pipes, step_pipes], self.CHILD_PIPE
        )

    @staticmethod
    def _perform_reset(env, reset_conns):
        # print("reseting env")
        obs_n = env.reset()
        # print(f"init obs={obs_n}")
        for i, conn in enumerate(reset_conns):
            conn.send(obs_n[i])

    @staticmethod
    def _perform_step(env, action_n, step_conns):
        # print(f"\nPerforming step:\naction={action_n}")
        obs_n, rew_n, done_n, info_n = env.step(action_n)
        # print(f"obs={obs_n}\nrew={rew_n}\ndone={done_n}")
        for i, conn in enumerate(step_conns):
            conn.send((obs_n[i], rew_n[i], done_n[i], info_n[i]))

    def _close_pipe_ends(self, pipe_lists, pipe_end):
        for i in range(self.num_agents):
            for pipes in pipe_lists:
                pipes[i][pipe_end].close()

    def close_conns(self):
        """Close any pipes """
        self._close_pipe_ends(
            [self._step_pipes, self._reset_pipes], self.PARENT_PIPE
        )
