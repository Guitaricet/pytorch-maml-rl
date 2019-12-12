"""
Wrappers for Metaworld compatibility
"""
import numpy as np

from metaworld import benchmarks

from metaworld.benchmarks.base import Benchmark
from metaworld.core.serializable import Serializable
from metaworld.envs.mujoco.multitask_env import MultiClassMultiTaskEnv

from metaworld.envs.mujoco.sawyer_xyz.sawyer_door import SawyerDoorEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_door_close import SawyerDoorCloseEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_drawer_open import SawyerDrawerOpenEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_reach_push_pick_place import SawyerReachPushPickPlaceEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_peg_insertion_side import SawyerPegInsertionSideEnv


def sample_tasks(mcmt_env, meta_batch_size, task2prob=None):
    """
    Overrides MultiClassMultiTaskEnv method allowing to sample
    tasks with given probabilities

    :param mcmt_env: Metaworld MultiClassMultiTaskEnv
    :param meta_batch_size: number of task-goal pairs to sample
    :param task2prob: task index to probability,
                        probabilities over all tasks should sum to 1
    """
    if mcmt_env._sampled_all:
        assert meta_batch_size >= len(mcmt_env._task_envs)
        tasks = [i for i in range(meta_batch_size)]
        return tasks

    tasks = np.random.randint(
            0, mcmt_env.num_tasks, size=meta_batch_size).tolist()

    if task2prob is not None:
        task_ids = list(range(mcmt_env.num_tasks))
        tasks = np.random.choice(task_ids, size=meta_batch_size, p=task2prob)

    if not mcmt_env._sample_goals: return tasks  # noqa: E701

    goals = [
        mcmt_env._task_envs[t % len(mcmt_env._task_envs)].sample_goals_(1)[0]
        for t in tasks
    ]
    tasks_with_goal = [
        dict(task=t, goal=g)
        for t, g in zip(tasks, goals)
    ]
    return tasks_with_goal


class ML1(benchmarks.ML1):
    def reset_task(self, task):
        self.set_task(task)
    
    def sample_tasks(self, meta_batch_size, task2prob=None):
        return sample_tasks(self, meta_batch_size, task2prob)


class ML10(benchmarks.ML10):
    def reset_task(self, task):
        self.set_task(task)

    def sample_tasks(self, meta_batch_size, task2prob=None):
        return sample_tasks(self, meta_batch_size, task2prob)


class ML45(benchmarks.ML45):
    def reset_task(self, task):
        self.set_task(task)

    def sample_tasks(self, meta_batch_size, task2prob=None):
        return sample_tasks(self, meta_batch_size, task2prob)

# Very small (2 train 2 test tasks) environment for debug purposes


DEBUG_MODE_CLS_DICT = dict(
    train={
        'reach-v1': SawyerReachPushPickPlaceEnv,
        'door-v1': SawyerDoorEnv,
        'peg-insert-side-v1': SawyerPegInsertionSideEnv,
    },
    test={
        'drawer-open-v1': SawyerDrawerOpenEnv,
        'door-close-v1': SawyerDoorCloseEnv,
    }
)

debug_mode_train_args_kwargs = {
    key: dict(args=[], kwargs={'obs_type': 'plain', 'random_init': True})
    for key, _ in DEBUG_MODE_CLS_DICT['train'].items()
}
debug_mode_test_args_kwargs = {
    key: dict(args=[], kwargs={'obs_type': 'plain'})
    for key, _ in DEBUG_MODE_CLS_DICT['test'].items()
}
debug_mode_train_args_kwargs['reach-v1']['kwargs']['task_type'] = 'reach'

DEBUG_MODE_ARGS_KWARGS = dict(
    train=debug_mode_train_args_kwargs,
    test=debug_mode_test_args_kwargs,
)


class ML3(MultiClassMultiTaskEnv, Benchmark, Serializable):
    """Benchmark-like environment for debugging"""

    def __init__(self, env_type='train', sample_all=False):
        assert env_type == 'train' or env_type == 'test'
        Serializable.quick_init(self, locals())

        cls_dict = DEBUG_MODE_CLS_DICT[env_type]
        args_kwargs = DEBUG_MODE_ARGS_KWARGS[env_type]

        super().__init__(
            task_env_cls_dict=cls_dict,
            task_args_kwargs=args_kwargs,
            sample_goals=True,
            obs_type='plain',
            sample_all=sample_all)

    def reset_task(self, task):
        self.set_task(task)

    def sample_tasks(self, meta_batch_size, task2prob=None):
        return sample_tasks(self, meta_batch_size, task2prob)
