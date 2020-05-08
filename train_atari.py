#!/usr/bin/env python
import argparse
import os
import numpy as np
import itertools

# import sys
# sys.path.append('/home/work')
# import goexplore_py.complex_fetch_env
from baselines import logger

def discounted_rewards(rewards, gamma):
    results = [0.0] * len(rewards)
    for i in reversed(range(len(rewards))):
        next_discounted = 0 if i + 1 >= len(results) else results[i + 1]
        results[i] = rewards[i] + gamma * next_discounted
    return results

def get_mean_reward(demo, fn, frameskip, gamma):
    if os.path.isdir(demo):
        import glob
        demos = glob.glob(demo + '/*.demo')
    else:
        demos = [demo]
    rewards = []
    for demo in demos:
        import pickle
        cur_rewards = pickle.load(open(demo, 'rb'))['rewards']
        cur_rewards = [sum(cur_rewards[i:i + frameskip]) for i in range(0, len(cur_rewards), frameskip)]
        if gamma is None:
            rewards += [abs(e) for e in cur_rewards if e != 0]
        else:
            rewards += [abs(e) for e in discounted_rewards(cur_rewards, gamma)]
    assert len(rewards) > 0
    if fn == 'median':
        return np.median(rewards)
    if fn == 'max':
        return np.max(rewards)
    if fn == 'mean':
        return np.mean(rewards)
    assert False, f'Unknown aggregation function: {fn}'


PROCS = []

def train(args, extra_data):
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
    if hvd.rank() == 0:
        while os.path.exists(args.save_path + '/progress.csv'):
            while args.save_path[-1] == '/':
                args.save_path = args.save_path[:-1]
            args.save_path += '_retry'
            # assert False, 'The save path already exists, something is wrong. If retrying the job, please clear this manually first.'
        logger.configure(args.save_path)
        os.makedirs(args.save_path + '/' + args.game, exist_ok=True)
        for k in list(extra_data):
            if 'prev_progress' in k:
                extra_data[k].to_csv(args.save_path + '/' + k + '.csv', index=False)
                del extra_data[k]

    frameskip = 1 if 'fetch' in args.game else 4

    if args.autoscale is not None:
        max_reward = get_mean_reward(args.demo, args.autoscale_fn, frameskip, (args.gamma if args.autoscale_value else None)) / args.autoscale
        args.scale_rewards = 1.0 / max_reward
        print(f'Autoscaling with scaling factor 1 / {max_reward} ({args.scale_rewards})')
        args.clip_rewards = False

    if 'Pitfall' in args.game and not args.scale_rewards:
        print('Forcing reward scaling because game is Pitfall!')
        args.scale_rewards = 0.001
        args.clip_rewards = False

    import json
    os.makedirs(args.save_path, exist_ok=True)
    json.dump(vars(args), open(args.save_path + '/kwargs.json', 'w'), indent=True, sort_keys=True)
    from baselines.common import set_global_seeds
    set_global_seeds(hvd.rank())
    from atari_reset.ppo import learn
    from atari_reset.policies import CnnPolicy, GRUPolicy, FFPolicy, FetchCNNPolicy
    from atari_reset.wrappers import ReplayResetEnv, ResetManager, SubprocVecEnv, VideoWriter, VecFrameStack, SuperDumbVenvWrapper, my_wrapper, MyResizeFrame, WarpFrame, MyResizeFrameOld, TanhWrap, SubprocWrapper, prepare_subproc

    if args.frame_resize == "MyResizeFrame":
        frame_resize_wrapper = MyResizeFrame
    elif args.frame_resize == "WarpFrame":
        frame_resize_wrapper = WarpFrame
    elif args.frame_resize == "MyResizeFrameOld":
        frame_resize_wrapper = MyResizeFrameOld
    else:
        raise NotImplementedError("No such frame-size wrapper: " + args.frame_resize)
    ncpu = 2
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    tf.Session(config=config).__enter__()

    # nrstartsteps = 320  # number of non frameskipped steps to divide workers over
    nrworkers = hvd.size() * args.nenvs
    workers_per_sp = int(np.ceil(nrworkers / args.nrstartsteps))

    if args.demo is None:
        args.demo = 'demos/' + args.game + '.demo'
    print('Using demo', args.demo)

    subproc_data = None

    def make_env(rank, is_extra_sil, subproc_idx):
        # print('WOW', rank, is_extra_sil)
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
                env = goexplore_py.dumb_fetch_env.ComplexFetchEnv(incl_extra_full_state=args.fetch_incl_extra_full_state)
            else:
                env = gym.make(args.game + 'NoFrameskip-v4')
            env = ReplayResetEnv(env,
                                 args,
                                 seed=rank,
                                 workers_per_sp=workers_per_sp,
                                 is_extra_sil=is_extra_sil,
                                 frameskip=frameskip
                                 )
            if 'fetch' not in args.game:
                if rank%args.nenvs == 0 and hvd.local_rank() == 0: # write videos during training to track progress
                    dir = os.path.join(args.save_path, args.game)
                    os.makedirs(dir, exist_ok=True)
                    if args.videos:
                        videofile_prefix = os.path.join(dir, 'episode')
                        env = VideoWriter(env, videofile_prefix)
                env = my_wrapper(env,
                                #  clip_rewards=args.clip_rewards,
                                 frame_resize_wrapper=frame_resize_wrapper,
                                #  scale_rewards=args.scale_rewards,
                                 sticky=args.sticky)
            else:
                env = TanhWrap(env)
            return env
        return env_fn

    env_types = [(i, False) for i in range(args.nenvs)]
    if args.n_sil_envs:
        # For cases where we start from the current starting points
        env_types += [(args.nenvs - 1, True)] * args.n_sil_envs
        # For cases where we start from the beginning
        # env_types += [(0, True)] * n_sil_envs

    env = SubprocVecEnv([make_env(i + args.nenvs * hvd.rank(), is_extra_sil, subproc_idx) for subproc_idx, (i, is_extra_sil) in enumerate(env_types)])
    env = ResetManager(env, move_threshold=args.move_threshold, steps_per_demo=args.steps_per_demo, fast_increase_starting_point=args.fast_increase_starting_point)
    if args.starting_points is not None:
        for i, e in enumerate(args.starting_points.split(',')):
            env.set_max_starting_point(int(e), i, args.move_threshold if args.sp_set_mt else 0)
    if 'fetch' not in args.game:
        env = VecFrameStack(env, frameskip)
    else:
        env = SuperDumbVenvWrapper(env)

    print('About to start PPO')
    if 'fetch' in args.game:
        if args.fetch_state_is_pixels:
            args.policy = FetchCNNPolicy
        else:
            print('Fetch environment, using the feedforward policy.')
            args.policy = FFPolicy
    else:
        args.policy = {'cnn' : CnnPolicy, 'gru': GRUPolicy}[args.policy]
    args.im_cells = extra_data.get('im_cells')
    learn(env, args, False)


if __name__ == '__main__':
    if 'CUSTOM_DOCKER_IMAGE' in os.environ:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        print('FILE LIMITS', soft, hard)
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (hard * 64, hard * 64))
        except Exception as e:
            print(f'Couldn\'t set file limit because of {e}. Hopefully that\'s ok')
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        print('FILE LIMITS', soft, hard)

        print('RUNNING ON MA DECTECTED')
        os.environ['FIBER_IMAGE'] = os.environ['CUSTOM_DOCKER_IMAGE'] + ':' + os.environ['CUSTOM_DOCKER_IMAGE_TAG']
        os.environ['FIBER_BACKEND'] = 'ma'
        os.chdir('/root/')

    parser = argparse.ArgumentParser()

    current_group = parser

    # TODO: this boolarg logic is copied from goexplore_py/main.py. Extract it into a library!
    def boolarg(arg, *args, default=False, help='', neg=None, dest=None):
        def extract_name(a):
            dashes = ''
            while a[0] == '-':
                dashes += '-'
                a = a[1:]
            return dashes, a

        if dest is None:
            _, dest = extract_name(arg)

        group = current_group.add_mutually_exclusive_group()
        group.add_argument(arg, *args, dest=dest, action='store_true', help=help + (' (DEFAULT)' if default else ''), default=default)
        not_args = []
        for a in [arg] + list(args):
            dashes, name = extract_name(a)
            not_args.append(f'{dashes}no_{name}')
        if isinstance(neg, str):
            not_args[0] = neg
        if isinstance(neg, list):
            not_args = neg
        group.add_argument(*not_args, dest=dest, action='store_false', help=f'Opposite of {arg}' + (' (DEFAULT)' if not default else ''), default=default)

    def add_argument(*args, **kwargs):
        if 'help' in kwargs and kwargs.get('default') is not None:
            kwargs['help'] += f' (default: {kwargs.get("default")})'

        current_group.add_argument(*args, **kwargs)

    # TODO: add boolargs to this.
    current_group = parser.add_argument_group('General')
    add_argument('--game', type=str, default='MontezumaRevenge')
    add_argument('--demo', type=str, default=None)
    add_argument('--start_frame', type=int, default=0,
                        help='Training will start this many frames back')
    add_argument('--load_path', type=str, default=None, help='Path to load existing model from')
    add_argument('--autoload_path', type=str, default=None, help='Path to load existing model and starting points from')
    add_argument('--save_path', type=str, default='results', help='Where to save results to')
    add_argument('--starting_points', type=str, default=None,
                        help='Demo-step to start training from, if not the last')
    boolarg('--sp_set_mt', default=True)
    add_argument('--num_timesteps', type=int, default=1e12)
    add_argument('--policy', default='gru')
    add_argument('--learning_rate', type=float, default=1e-4)
    add_argument('--ent_coef', type=float, default=1e-4)
    add_argument('--vf_coef', type=float, default=0.5)
    add_argument('--sil_coef', type=float, default=0.0)
    add_argument('--sil_vf_coef', type=float, default=0.0, help='The value function coef for SIL. Good values: 0.0 or 0.01.')
    add_argument('--sil_ent_coef', type=float, default=0.0, help='The entropy coef for SIL.')
    add_argument('--allowed_lag', type=int, default=50)
    add_argument('--gamma', type=float, default=0.999)
    add_argument('--move_threshold', type=float, default=0.2)
    boolarg('--game_over_on_life_loss', default=True,
                        help='Whether the agent is allowed to continue after a life loss.')
    boolarg('--always_run_till_done', default=False,
                        help='Whether to always run the demo until the done signal or (otherwise terminating early is ok).')
    add_argument('--allowed_score_deficit', type=int, default=0,
                        help='The imitator is allowed to be this many points worse than the example')
    add_argument('--frame_resize', type=str, default="MyResizeFrame",
                        help='Resize wrapper to be used for the game.')
    add_argument('--steps_per_demo', type=int, default=1024)
    add_argument('--nrstartsteps', type=int, default=320)
    add_argument('--n_sil_envs', type=int, default=1)
    add_argument('--ffmemsize', type=int, default=800)
    add_argument('--nenvs', type=int, default=32)
    boolarg('--clip_rewards', default=True)
    boolarg('--noops', default=True)
    add_argument('--scale_rewards', type=float, default=None)
    add_argument('--fast_increase_starting_point', action='store_true', default=False)
    add_argument("--sticky", help="Use sticky actions", action="store_true")
    add_argument("--test_from_start", action="store_true", default=False,
                        help="Add a virtual demo that always runs from the start")
    add_argument("--autoscale", type=float, default=None,
                        help="Automatically scale rewards based on the max absolute reward in the demos so that the value passed to --autoscale is the max possible reward")
    add_argument("--autoscale_fn", type=str, default='mean',
                        help="Function to use for computing autoscale factor. Options are: mean, median, max")
    boolarg('--autoscale_value', default=False, help='Whether to use the value function for autoscaling. Otherwise just use the absolute value of non-zero rewards.')
    boolarg("--videos",
                        help="Add a virtual demo that always runs from the start")
    add_argument("--from_start_prior", type=int, default=0,
                        help="Prior expected number of frames for the demonstration from the start")
    add_argument("--demo_selection", type=str, default='uniform',
                        help="How to select which demonstration to train on."
                             " Options are: uniform, normalize, normalize_from_start and normalize_by_target (note: normalize_by_target also normalizes from start).")
    add_argument("--avg_frames_window_size", type=int, default=0,
                        help="Size of the rolling window used to estimate the number of frames produced by each demo")
    add_argument("--noptepochs", type=int, default=4,
                        help="Number of epochs for which to train the network")
    add_argument("--ffshape", type=str, default='2x256',
                        help="Shape of fully connected network: NLAYERxWIDTH")
    boolarg('--sil_weight_success_rate', default=False)
    boolarg('--sil_vf_relu', default=False)
    boolarg('--sil_pg_weight_by_value', default=False)
    add_argument("--sd_multiply_explore", type=int, default=2,
                        help="Multiply the sd by this amount when increasing entropy")
    add_argument('--nsteps', type=int, default=128)
    add_argument('--extra_sil_from_start_prob', type=float, default=-1)
    add_argument('--extra_sil_before_demo_max', type=int, default=10)
    add_argument('--max_demo_len', type=int, default=10000000000000)
    add_argument('--fake_target_probability', type=float, default=0.0)
    add_argument('--lam', type=float, default=0.95)
    add_argument('--l2_coef', type=float, default=1e-7)
    add_argument('--cliprange', type=float, default=0.1)
    add_argument('--adam_epsilon', type=float, default=1e-6)
    add_argument('--intrinsic_reward_weight', type=float, default=-1)
    boolarg('--im_count_all', default=False)
    boolarg('--im_reward_all', default=False)
    add_argument('--log_interval', type=int, default=1)
    add_argument('--save_interval', type=int, default=100)
    boolarg('--norm_adv', default=True)
    boolarg('--subtract_rew_avg', default=False)
    boolarg('--only_forward', default=False)
    add_argument('--cell_rewards', type=float, default=0)
    add_argument('--cell_reward_window', type=int, default=10)
    boolarg('--cell_from_demo_unique', default=True)
    boolarg('--montezuma_checkpoint_domain_knowledge', default=False)
    boolarg('--sil_from_start_only_best_demo', default=False)
    boolarg('--sil_from_start_prob_by_selection', default=False, help='If true, everytime we select SIL we will randomly sample according to extra_sil_from_start_prob. If false (default), we will normalize by the number of steps it will take to do SIL from start vs regular SIL.')
    add_argument('--from_start_demo_reward_interval_factor', type=float, default=200000, help='When doing demo from start, we will stop early if we haven\'t collected any rewards in an interval longer than the maximum interval between two rewards in the demos times this factor')
    boolarg('--sil_finish_demo', default=False)
    add_argument('--from_start_proportion', type=float, default=None, help='The proportion of from start. By default 1/(n_demos + 1).')
    boolarg('--correct_score_counter', default=False)
    add_argument('--extra_frames_exp_factor', type=int, default=7)

    current_group = parser.add_argument_group('Fetch')
    add_argument('--fetch_nsubsteps', type=int, default=20)
    add_argument('--fetch_timestep', type=float, default=0.002)
    add_argument('--fetch_max_steps', type=int, default=4000)
    add_argument('--fetch_total_timestep', type=float, default=None)
    add_argument("--inc_entropy_threshold", type=int, default=100,
                        help="Increase entropy when at this stage in the demo")
    add_argument('--fetch_incl_extra_full_state', action='store_true', default=False)
    add_argument('--fetch_state_is_pixels', action='store_true', default=False)
    add_argument('--fetch_force_closed_doors', action='store_true', default=False)
    add_argument('--fetch_include_proprioception', action='store_true', default=False)
    add_argument('--fetch_state_azimuths', type=str, default='145_215')
    add_argument('--fetch_type', type=str, default='boxes')
    add_argument('--fetch_target_location', type=str, default=None)
    add_argument('--fetch_state_wh', type=int, default=96)
    args = parser.parse_args()

    args.im_cells = None

    if args.fetch_total_timestep is not None:
        args.fetch_timestep = args.fetch_total_timestep / args.fetch_nsubsteps

    if 'fetch' in args.game:
        args.noops = False

    import atari_reset.policies
    atari_reset.policies.FFSHAPE = args.ffshape
    atari_reset.policies.SD_MULTIPLY_EXPLORE = args.sd_multiply_explore
    atari_reset.policies.MEMSIZE = args.ffmemsize

    assert not args.autoload_path or not args.load_path, 'Autoload path and load path are mutually exclusive'
    assert not args.autoload_path or not args.starting_points, 'Autoload path and starting points are mutually exclusive'
    extra_data = {}
    if args.autoload_path:
        import glob, pandas as pd, pickle
        if args.demo == '__nodemo__':
            best_frames = -1
            cur_path = args.autoload_path
            args.load_path = None
            while os.path.exists(cur_path + '/progress.csv'):
                timesteps = list(pd.read_csv(cur_path + '/progress.csv')['total_timesteps'])[-1]
                if timesteps > best_frames:
                    args.load_path = cur_path
                    best_frames = timesteps
                    print('BEST:', best_frames)
                while cur_path[-1] == '/':
                    cur_path = cur_path[:-1]
                cur_path += '_retry'

            print('Restarting from', args.load_path)
            args.autoload_path = args.load_path
            args.load_path = max(glob.glob(f'{args.autoload_path}/{args.game}/*'))
            index = int(args.load_path.split('/')[-1])
            progress_number = 0
            for prog_file in sorted(glob.glob(args.autoload_path + '/prev_progress*.csv')):
                extra_data[f'prev_progress_{progress_number:03}'] = pd.read_csv(prog_file)
                progress_number += 1
            extra_data[f'prev_progress_{progress_number:03}'] = pd.read_csv(args.autoload_path + '/progress.csv').iloc[:index]
            if args.intrinsic_reward_weight != 0:
                all_cells = pickle.load(open(glob.glob(args.autoload_path + '/im_cells.p*k*l*')[0], 'rb'))
                # import ipdb; ipdb.set_trace()
                if isinstance(all_cells[0], dict):
                    extra_data['im_cells'] = all_cells[index]
                else:
                    assert all_cells[0][-1][0] == index
                    extra_data['im_cells'] = all_cells[0][-1][1]
        else:
            args.load_path = max(glob.glob(f'{args.autoload_path}/{args.game}/*'))
            print('Loading model:', args.load_path)
            progress_file = args.autoload_path + '/progress.csv'
            df = pd.read_csv(progress_file)
            starting_points = []
            for e in df.columns:
                if e.startswith('max_starting_point'):
                    starting_points.append((int(e.split('_')[-1]), df[e][len(df)-1]))
            starting_points = sorted(starting_points)
            args.starting_points = ','.join(str(e) for _, e in starting_points)
            assert [e for e, _ in starting_points] == list(range(len(starting_points)))
            print('Using starting points:', args.starting_points)

    train(args, extra_data)
