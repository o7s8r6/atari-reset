#!/usr/bin/env python
import argparse
import os
import gym
import numpy as np

def test(args):
    import filelock
    with filelock.FileLock('/tmp/robotstify.lock'):
        import gym
        import sys
        try:
            import goexplore_py.complex_fetch_env
        except Exception:
            print('Could not import complex_fetch_env, is goexplore_py in PYTHONPATH?')

    import tensorflow as tf
    import horovod.tensorflow as hvd
    hvd.init()
    print('initialized worker %d' % hvd.rank(), flush=True)
    from baselines.common import set_global_seeds
    set_global_seeds(hvd.rank())
    from baselines import bench
    from baselines.common import set_global_seeds
    from atari_reset.wrappers import VecFrameStack, VideoWriter, my_wrapper,\
        EpsGreedyEnv, StickyActionEnv, NoopResetEnv, SubprocVecEnv, PreventSlugEnv, FetchSaveEnv, TanhWrap
    from atari_reset.ppo import learn
    from atari_reset.policies import CnnPolicy, GRUPolicy, FFPolicy

    set_global_seeds(hvd.rank())
    ncpu = 2
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    tf.Session(config=config).__enter__()

    max_noops = 30 if args.noops else 0
    print('SAVE PATH', args.save_path)

    def make_env(rank):
        def env_fn():
            if args.game == 'fetch':
                assert args.fetch_target_location is not None, 'For now, we require a target location for fetch'
                kwargs = {}
                dargs = vars(args)
                for attr in dargs:
                    if attr.startswith('fetch_'):
                        if attr == 'fetch_type':
                            kwargs['model_file'] = f'teleOp_{args.fetch_type}.xml'
                        elif attr != 'fetch_total_timestep':
                            kwargs[attr[len('fetch_'):]] = dargs[attr]

                env = goexplore_py.complex_fetch_env.ComplexFetchEnv(
                    **kwargs
                )
            elif args.game == 'fetch_dumb':
                env = goexplore_py.dumb_fetch_env.ComplexFetchEnv()
            else:
                env = gym.make(args.game + 'NoFrameskip-v4')
                if args.seed_env:
                    env.seed(0)
                # if args.unlimited_score:
                #     # This removes the TimeLimit wrapper around the env
                #     env = env.env
                # env = PreventSlugEnv(env)
            # change for long runs
            # env._max_episode_steps *= 1000
            env = bench.Monitor(env, "{}.monitor.json".format(rank), allow_early_resets=True)
            if False and rank%nenvs == 0 and hvd.local_rank()==0:
                os.makedirs(args.save_path + '/vids/' + args.game, exist_ok=True)
                videofile_prefix = args.save_path + '/vids/' + args.game
                env = VideoWriter(env, videofile_prefix)
            if 'fetch' not in args.game:
                if args.noops:
                    os.makedirs(args.save_path, exist_ok=True)
                    env = NoopResetEnv(env, 30, nenvs, args.save_path, num_per_noop=args.num_per_noop, unlimited_score=args.unlimited_score)
                    env = my_wrapper(env, clip_rewards=True, sticky=args.sticky)
                if args.epsgreedy:
                    env = EpsGreedyEnv(env)
            else:
                os.makedirs(f'{args.save_path}', exist_ok=True)
                env = FetchSaveEnv(env, rank=rank, n_ranks=nenvs, save_path=f'{args.save_path}/', demo_path=args.demo)
                env = TanhWrap(env)
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

    nenvs = args.nenvs
    env = SubprocVecEnv([make_env(i + nenvs * hvd.rank()) for i in range(nenvs)])
    env = VecFrameStack(env, 1 if 'fetch' in args.game else 4)

    if 'fetch' in args.game:
        print('Fetch environment, using the feedforward policy.')
        args.policy = FFPolicy
    else:
        args.policy = {'cnn': CnnPolicy, 'gru': GRUPolicy}[args.policy]

    args.sil_pg_weight_by_value = False
    args.sil_vf_relu = False
    args.sil_vf_coef = 0
    args.sil_coef = 0
    args.sil_ent_coef = 0
    args.ent_coef = 0
    args.vf_coef = 0
    args.cliprange = 1
    args.l2_coef = 0
    args.adam_epsilon = 1e-8
    args.gamma = 0.99
    args.lam = 0.10
    args.scale_rewards = 1.0
    args.sil_weight_success_rate = True
    args.norm_adv = 1.0
    args.log_interval = 1
    args.save_interval = 100
    args.subtract_rew_avg = True
    args.clip_rewards = False
    learn(env, args, True)
    # learn(policy=policy, env=env, nsteps=256, log_interval=1, save_interval=100, total_timesteps=args.num_timesteps,
    #       load_path=args.load_path, save_path=args.save_path, game_name=args.game, test_mode=True, max_noops=max_noops)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', type=str, default='MontezumaRevenge')
    parser.add_argument('--num_timesteps', type=int, default=1e8)
    parser.add_argument('--num_per_noop', type=int, default=500)
    parser.add_argument('--policy', default='gru')
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--save_path', type=str, default='', help='Where to save results to')
    parser.add_argument("--noops", help="Use 0 to 30 random noops at the start of each episode", action="store_true")
    parser.add_argument("--sticky", help="Use sticky actions", action="store_true")
    parser.add_argument("--seed_env", help="Seed the environment", action="store_true")
    parser.add_argument("--unlimited_score", help="Run with no time limit and fix the issue with score rollover", action="store_true")
    parser.add_argument('--nsubsteps', type=int, default=40)
    parser.add_argument("--epsgreedy", help="Take random action with probability 0.01", action="store_true")
    parser.add_argument('--demo', type=str, default=None)
    parser.add_argument('--nenvs', type=int, default=32)
    parser.add_argument("--ffshape", type=str, default='1x1024',
                        help="Shape of fully connected network: NLAYERxWIDTH")
    parser.add_argument('--fetch_nsubsteps', type=int, default=20)
    parser.add_argument('--fetch_timestep', type=float, default=0.002)
    parser.add_argument('--fetch_total_timestep', type=float, default=None)
    parser.add_argument("--inc_entropy_threshold", type=int, default=100,
                        help="Increase entropy when at this stage in the demo")
    parser.add_argument('--fetch_incl_extra_full_state', action='store_true', default=False)
    parser.add_argument('--fetch_state_is_pixels', action='store_true', default=False)
    parser.add_argument('--fetch_force_closed_doors', action='store_true', default=False)
    parser.add_argument('--fetch_include_proprioception', action='store_true', default=False)
    parser.add_argument('--fetch_state_azimuths', type=str, default='145_215')
    parser.add_argument('--fetch_type', type=str, default='boxes')
    parser.add_argument('--fetch_target_location', type=str, default=None)
    parser.add_argument('--fetch_state_wh', type=int, default=96)
    parser.add_argument('--nsteps', type=int, default=256)
    parser.add_argument('--ffmemsize', type=int, default=800)
    args = parser.parse_args()

    if args.fetch_total_timestep is not None:
        args.fetch_timestep = args.fetch_total_timestep / args.fetch_nsubsteps

    args.im_cells = {}

    # assert not os.path.exists(args.save_path)

    import atari_reset.policies
    atari_reset.policies.FFSHAPE = args.ffshape
    atari_reset.policies.MEMSIZE = args.ffmemsize


    def check_done():
        import pickle, glob
        all_episodes = []
        for e in glob.glob(f'{args.save_path}/*.pickle'):
            all_episodes += pickle.load(open(e, 'rb'))
        all_episodes.sort(key=lambda x: x['start_step'])
        if len(all_episodes) < args.num_per_noop:
            return False
        n_to_consider = args.num_per_noop
        while n_to_consider < len(all_episodes) - 1 and all_episodes[n_to_consider - 1]['start_step'] == \
                all_episodes[n_to_consider]['start_step']:
            n_to_consider += 1
        return all([('score' in e) for e in all_episodes[:n_to_consider]])

    if not check_done():
        test(args)
