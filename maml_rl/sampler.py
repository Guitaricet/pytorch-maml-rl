import gym
import torch
import multiprocessing as mp

from maml_rl.envs.subproc_vec_env import SubprocVecEnv
from maml_rl.episode import BatchEpisodes

from metaworld.benchmarks import ML1, ML10, ML45
from maml_rl.envs.metaworld import MetaworldWrapper


def make_env(env_name):
    # def _make_env():
    #     # import pdb; pdb.set_trace()
    #     return gym.make(env_name)
    # return _make_env

    # TODO: hardcode
    def _make_env():
        if env_name == 'ml10':
            env = ML10.get_train_tasks()
        elif env_name == 'ml45':
            env = ML45.get_train_tasks()
        else:
            env = ML1.get_train_tasks(env_name)

        tasks = env.sample_tasks(1)  # Sample a task
        env.set_task(tasks[0])  # Set task
        env.__class__ = MetaworldWrapper  # Wrap env (add some methods)
        return env
    return _make_env


class BatchSampler(object):
    def __init__(self, env_name, batch_size, num_workers=None):
        self.env_name = env_name
        self.batch_size = batch_size
        self.num_workers = num_workers or mp.cpu_count() - 1

        self.queue = mp.Queue()
        self.envs = SubprocVecEnv([make_env(env_name) for _ in range(num_workers)],
            queue=self.queue)
        self._env = make_env(env_name)()

    def sample(self, policy, params=None, gamma=0.95, device='cpu'):
        episodes = BatchEpisodes(batch_size=self.batch_size, gamma=gamma, device=device)
        for i in range(self.batch_size):
            self.queue.put(i)
        for _ in range(self.num_workers):
            self.queue.put(None)
        observations, batch_ids = self.envs.reset()
        dones = [False]
        while (not all(dones)) or (not self.queue.empty()):
            with torch.no_grad():
                observations_tensor = torch.from_numpy(observations).to(device=device, dtype=torch.float32)
                actions_tensor = policy(observations_tensor, params=params).sample()
                actions = actions_tensor.cpu().numpy()
            new_observations, rewards, dones, new_batch_ids, info = self.envs.step(actions)
            # info keys: reachDist, pickRew, epRew, goalDist, success, goal, task_name
            episodes.append(observations, actions, rewards, batch_ids, info)
            observations, batch_ids = new_observations, new_batch_ids
        return episodes

    def reset_task(self, task):
        tasks = [task for _ in range(self.num_workers)]
        reset = self.envs.reset_task(tasks)
        return all(reset)

    def sample_tasks(self, num_tasks):
        tasks = self._env.unwrapped.sample_tasks(num_tasks)
        return tasks
