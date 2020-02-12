import gym
import torch
import multiprocessing as mp

from maml_rl.envs.subproc_vec_env import SubprocVecEnv
from maml_rl.episode import BatchEpisodes

from maml_rl.envs.metaworld import ML1, ML10, ML45, ML3


def make_env(env_name, test_env=False):
    def _make_env():

        if env_name == 'ml10':
            env_factory = ML10
        elif env_name == 'ml45':
            env_factory = ML45
        elif env_name == 'ml3':
            env_factory = ML3
        else:
            print(f'[INFO] using ML1 with task {env_name}')
            print(f'[INFO] train and test task are the same in ML1')
            return ML1.get_train_tasks(env_name)

        env = (env_factory.get_test_tasks() if test_env
               else env_factory.get_train_tasks())
        return env

    return _make_env


class BatchSampler(object):
    def __init__(self, env_name, batch_size, num_workers=None, test_env=False):
        self.env_name = env_name
        self.batch_size = batch_size
        self.num_workers = num_workers or mp.cpu_count() - 1
        self.test_env = test_env

        self.queue = mp.Queue()
        self.envs = SubprocVecEnv([make_env(env_name, test_env=test_env) for _ in range(num_workers)],
                                  queue=self.queue)
        self._env = make_env(env_name, test_env=test_env)()

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
            new_observations, rewards, dones, new_batch_ids, infos = self.envs.step(actions)
            # info keys: reachDist, pickRew, epRew, goalDist, success, goal, task_name

            # NOTE: last infos will be absent if batch_size % num_workers != 0

            episodes.append(observations, actions, rewards, batch_ids, infos)
            observations, batch_ids = new_observations, new_batch_ids
        return episodes

    def reset_task(self, task):
        tasks = [task for _ in range(self.num_workers)]
        reset = self.envs.reset_task(tasks)
        return all(reset)

    def sample_tasks(self, num_tasks, task2prob=None):
        tasks = self._env.unwrapped.sample_tasks(num_tasks, task2prob)
        return tasks
