import gymnasium as gym
import numpy as np
from gymnasium import spaces

class JobShopEnv(gym.Env):
    def __init__(self, jobs=3, machines=3):
        super().__init__()

        self.jobs = jobs
        self.machines = machines

        self.processing_time = np.random.randint(1, 8, (jobs, machines))

        self.job_step = np.zeros(jobs, dtype=int)
        self.machine_time = np.zeros(machines)
        self.time = 0

        self.schedule_log = []  # (job, machine, start, end)

        self.action_space = spaces.Discrete(jobs)
        self.observation_space = spaces.Box(
            low=0, high=100, shape=(jobs + machines + 1,), dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.job_step[:] = 0
        self.machine_time[:] = 0
        self.time = 0
        self.schedule_log.clear()
        return self._get_state(), {}

    def _get_state(self):
        return np.concatenate([
            self.job_step,
            self.machine_time,
            [self.time]
        ]).astype(np.float32)

    def step(self, action):
        if self.job_step[action] >= self.machines:
            return self._get_state(), -10, False, False, {}

        machine = self.job_step[action]

        start = self.machine_time[machine]
        duration = self.processing_time[action, machine]
        end = start + duration

        self.machine_time[machine] = end
        self.job_step[action] += 1
        self.time = min(self.machine_time)

        # LOG FOR GANTT
        self.schedule_log.append((action, machine, start, end))

        done = bool(np.all(self.job_step >= self.machines))
        reward = -end

        return self._get_state(), reward, done, False, {}