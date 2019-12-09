import json
from datetime import datetime
from collections import defaultdict

import gym
import torch
import wandb
import numpy as np

from tensorboardX import SummaryWriter

import maml_rl.envs
from maml_rl.metalearner import MetaLearner
from maml_rl.policies import CategoricalMLPPolicy, NormalMLPPolicy
from maml_rl.baseline import LinearFeatureBaseline
from maml_rl.sampler import BatchSampler


def get_date_str():
    d = datetime.now()
    return f'{d.month}_{d.day}'


def total_rewards(episodes_rewards, aggregation=torch.mean):
    rewards = torch.mean(torch.stack([aggregation(torch.sum(rewards, dim=0))
        for rewards in episodes_rewards], dim=0))
    return rewards.item()


def get_success_rate(episodes_infos, per_task=False):
    """
    :param episodes_infos: nested lists (n_tasks, n_episodes, n_timestemps) of dicts
                           i.e. (meta_batch_size, fast_batch_size, n_timestemps)
    """

    # info keys: reachDist, pickRew, epRew, goalDist, success, goal, task_name
    # n_tasks = len(episodes_infos)
    n_episodes = len(episodes_infos[0])
    task_success_rate = dict()

    n_successes = 0

    episodes_total = 0
    for task_infos in episodes_infos:
        if 'task_name' not in task_infos[0][0]: continue  # noqa: E701
        # first episode, first timestamp
        task = task_infos[0][0]['task_name']
        task = task.replace('Env', '')
        task = task.replace('Sawyer', '')
        task_success_rate[task] = 0

        for episode_infos in task_infos:
            episodes_total += 1
            for timestamp in episode_infos:
                if timestamp and timestamp['success']:
                    task_success_rate[task] += 1
                    n_successes += 1
                    break

    for task, success in task_success_rate.items():
        task_success_rate[task] = success / n_episodes

    n_successes /= episodes_total
    return n_successes, task_success_rate


def main(args):
    continuous_actions = (args.env_name in {'AntVel-v1', 'AntDir-v1',
        'AntPos-v0', 'HalfCheetahVel-v1', 'HalfCheetahDir-v1',
        '2DNavigation-v0', 'pick-place-v1', 'ml10', 'ml45', 'ml2'})

    save_folder = './saves/{0}'.format(args.output_folder + get_date_str())
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    with open(os.path.join(save_folder, 'config.json'), 'w') as f:
        config = {k: v for (k, v) in vars(args).items() if k != 'device'}
        config.update(device=args.device.type)
        json.dump(config, f, indent=2)

    print('Initializing samplers...')

    sampler = BatchSampler(args.env_name, batch_size=args.fast_batch_size,
                           num_workers=args.num_workers)

    test_sampler = BatchSampler(args.env_name, test_env=True, batch_size=args.fast_batch_size,
                                num_workers=args.num_workers)

    if continuous_actions:
        policy = NormalMLPPolicy(
            int(np.prod(sampler.envs.observation_space.shape)),
            int(np.prod(sampler.envs.action_space.shape)),
            hidden_sizes=(args.hidden_size,) * args.num_layers)
    else:
        policy = CategoricalMLPPolicy(
            int(np.prod(sampler.envs.observation_space.shape)),
            sampler.envs.action_space.n,
            hidden_sizes=(args.hidden_size,) * args.num_layers)
    baseline = LinearFeatureBaseline(
        int(np.prod(sampler.envs.observation_space.shape)))

    print('Initializing meta-learners...')

    metalearner = MetaLearner(sampler, policy, baseline, gamma=args.gamma,
        fast_lr=args.fast_lr, tau=args.tau, device=args.device)

    # NOTE: we need this metalearner only for sampling
    test_metalearner = MetaLearner(test_sampler, policy, baseline, gamma=args.gamma,
        fast_lr=args.fast_lr, tau=args.tau, device=args.device)

    print('Starting the training')

    # Initialize logging
    wandb.init()
    wandb.config.update(args)
    wandb.watch(policy)

    # outer loop (meta-training)
    for i in range(args.num_batches):
        print(f'Batch {i}')

        # sample trajectories from random tasks
        print(f'\tSampling a batch of {args.meta_batch_size} training tasks')
        tasks = sampler.sample_tasks(num_tasks=args.meta_batch_size)

        # inner loop (adaptation)
        # returns list of tuples (train_episodes, valid_episodes)
        print(f'\tTraining')
        episodes = metalearner.sample(tasks, first_order=args.first_order)

        print(f'\tUpdating the meta-model')
        metalearner.step(episodes, max_kl=args.max_kl, cg_iters=args.cg_iters,
            cg_damping=args.cg_damping, ls_max_steps=args.ls_max_steps,
            ls_backtrack_ratio=args.ls_backtrack_ratio)

        # Logging
        r_before = total_rewards([ep.rewards for ep, _ in episodes])
        r_after = total_rewards([ep.rewards for _, ep in episodes])

        test_episode_infos = [ep._info_list for _, ep in episodes]
        success_rate, task_success_rate = get_success_rate(
            test_episode_infos, per_task=True
        )
        wandb.log({'total_rewards/before_update': r_before,
                   'total_rewards/after_update': r_after,
                   'success_rate/total': success_rate})
        wandb.log({f'success_rate/{task}': rate for task, rate in task_success_rate.items()})

        # meta-test
        if i and i % args.eval_every == 0:
            print(f'Evaluating on meta-test')

            # save policy network
            with open(os.path.join(save_folder,
                      'policy-{0}.pt'.format(i)), 'wb') as f:
                torch.save(policy.state_dict(), f)

            # Evaluate on meta-test
            tasks = test_sampler.sample_tasks(num_tasks=5 * args.meta_batch_size)

            episodes = test_metalearner.sample(tasks, first_order=args.first_order)

            r_before = total_rewards([ep.rewards for ep, _ in episodes])
            r_after = total_rewards([ep.rewards for _, ep in episodes])

            test_episode_infos = [ep._info_list for _, ep in episodes]
            success_rate, task_success_rate = get_success_rate(
                test_episode_infos, per_task=True
            )

            wandb.log({'total_rewards_test/before_update': r_before,
                       'total_rewards_test/after_update': r_after,
                       'success_rate_test/total': success_rate})
            wandb.log({f'success_rate_test/{task}': rate for task, rate in task_success_rate.items()})

    print('Saving the final model')
    # save final policy
    with open(os.path.join(save_folder,
              'policy-final.pt'), 'wb') as f:
        torch.save(policy.state_dict(), f)


if __name__ == '__main__':
    import argparse
    import os
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description='Reinforcement learning with '
                                     'Model-Agnostic Meta-Learning (MAML)')

    # General
    parser.add_argument('--env-name', type=str,
        help='name of the environment')
    parser.add_argument('--gamma', type=float, default=0.95,
        help='value of the discount factor gamma')
    parser.add_argument('--tau', type=float, default=1.0,
        help='value of the discount factor for GAE')
    parser.add_argument('--first-order', action='store_true',
        help='use the first-order approximation of MAML')

    # Policy network (relu activation function)
    parser.add_argument('--hidden-size', type=int, default=100,
        help='number of hidden units per layer')
    parser.add_argument('--num-layers', type=int, default=2,
        help='number of hidden layers')

    # Task-specific
    parser.add_argument('--fast-batch-size', type=int, default=20,
        help='batch size for each individual task')
    parser.add_argument('--fast-lr', type=float, default=0.5,
        help='learning rate for the 1-step gradient update of MAML')

    # Optimization
    parser.add_argument('--num-batches', type=int, default=200,
        help='number of batches to train on')
    parser.add_argument('--eval-every', type=int, default=100,
        help='number of batches between evaluation on meta-test')

    parser.add_argument('--meta-batch-size', type=int, default=40,
        help='number of tasks per batch')

    parser.add_argument('--max-kl', type=float, default=1e-2,
        help='maximum value for the KL constraint in TRPO')
    parser.add_argument('--cg-iters', type=int, default=10,
        help='number of iterations of conjugate gradient')
    parser.add_argument('--cg-damping', type=float, default=1e-5,
        help='damping in conjugate gradient')
    parser.add_argument('--ls-max-steps', type=int, default=15,
        help='maximum number of iterations for line search')
    parser.add_argument('--ls-backtrack-ratio', type=float, default=0.8,
        help='maximum number of iterations for line search')

    # Miscellaneous
    parser.add_argument('--output-folder', type=str, default='maml',
        help='name of the output folder')
    parser.add_argument('--num-workers', type=int, default=mp.cpu_count() - 1,
        help='number of workers for trajectories sampling')
    parser.add_argument('--device', type=str, default='cpu',
        help='set the device (cpu or cuda)')

    args = parser.parse_args()

    # Create logs and saves folder if they don't exist
    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    if not os.path.exists('./saves'):
        os.makedirs('./saves')
    # Device
    args.device = torch.device(args.device
        if torch.cuda.is_available() else 'cpu')
    # Slurm
    if 'SLURM_JOB_ID' in os.environ:
        args.output_folder += '-{0}'.format(os.environ['SLURM_JOB_ID'])

    main(args)
