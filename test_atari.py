#!/usr/bin/env python

# Modifications Copyright (c) 2019 Uber Technologies, Inc.

import argparse
import os
import gym
import numpy as np

def test(game_name, num_timesteps, policy, load_path, save_path, noops=False, sticky=False, epsgreedy=False):
    import tensorflow as tf
    import horovod.tensorflow as hvd
    hvd.init()
    print('initialized worker %d' % hvd.rank(), flush=True)
    from baselines.common import set_global_seeds
    set_global_seeds(hvd.rank())
    from baselines import bench
    from baselines.common import set_global_seeds
    from atari_reset.wrappers import VecFrameStack, VideoWriter, my_wrapper,\
        EpsGreedyEnv, StickyActionEnv, NoopResetEnv, SubprocVecEnv, PreventSlugEnv
    from atari_reset.ppo import learn
    from atari_reset.policies import CnnPolicy, GRUPolicy

    set_global_seeds(hvd.rank())
    ncpu = 2
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    tf.Session(config=config).__enter__()

    print('SAVE PATH', save_path)

    def make_env(rank):
        def env_fn():
            np.random.seed(rank)
            env = gym.make(game_name + 'NoFrameskip-v4')
            env = PreventSlugEnv(env)
            env = bench.Monitor(env, "{}.monitor.json".format(rank), allow_early_resets=True)
            if False and rank%nenvs == 0 and hvd.local_rank()==0:
                os.makedirs(save_path + '/vids/' + game_name, exist_ok=True)
                videofile_prefix = save_path + '/vids/' + game_name
                env = VideoWriter(env, videofile_prefix)
            if noops:
                os.makedirs(save_path, exist_ok=True)
                env = NoopResetEnv(env, 30, nenvs, save_path)
            env = my_wrapper(env, clip_rewards=True, sticky=sticky)
            if epsgreedy:
                env = EpsGreedyEnv(env)
            # def print_rec(e):
            #     print(e.__class__.__name__)
            #     if hasattr(e, 'env'):
            #         print_rec(e.env)
            # import time
            # import random
            # time.sleep(random.random() * 10)
            # print('\tSHOWING STUFF')
            # print_rec(env)
            # print('\n\n\n')
            return env
        return env_fn

    nenvs = 12
    env = SubprocVecEnv([make_env(i + nenvs * hvd.rank()) for i in range(nenvs)])
    env = VecFrameStack(env, 4)

    policy = {'cnn': CnnPolicy, 'gru': GRUPolicy}[policy]
    learn(policy=policy, env=env, nsteps=256, log_interval=1, save_interval=100, total_timesteps=num_timesteps,
          load_path=load_path, save_path=save_path, game_name=game_name, test_mode=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', type=str, default='MontezumaRevenge')
    parser.add_argument('--num_timesteps', type=int, default=1e8)
    parser.add_argument('--policy', default='gru')
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--save_path', type=str, default='', help='Where to save results to')
    parser.add_argument("--noops", help="Use 0 to 30 random noops at the start of each episode", action="store_true")
    parser.add_argument("--sticky", help="Use sticky actions", action="store_true")
    parser.add_argument("--epsgreedy", help="Take random action with probability 0.01", action="store_true")
    args = parser.parse_args()

    test(args.game, args.num_timesteps, args.policy, args.load_path, args.save_path, args.noops, args.sticky, args.epsgreedy)
