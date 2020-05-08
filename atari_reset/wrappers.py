import tempfile
import os
import random
import pickle
import gym
from collections import deque
from PIL import Image
from gym import spaces
import copy
import imageio
import numpy as np
from multiprocessing import Process, Pipe
import mpi4py.rc
mpi4py.rc.initialize = False
from mpi4py import MPI
import traceback
try:
    from goexplore_py.utils import imdownscale
except Exception:
    print('Warning: goexplore_py not in PYTHONPATH, some features will be unavailable')
reset_for_batch = False


# change for long runs
SCORE_THRESHOLD = 500_000_000_000
SUBPROCPOOL = None
MAP_THREAD = None


def prepare_subproc(subproctype, n_envs):
    if subproctype == 'fiber':
        from fiber import Pool, SimpleQueue
    else:
        from multiprocessing import Pool, SimpleQueue

    global SUBPROCPOOL
    print('Creating Queues')
    queues = [(SimpleQueue(), SimpleQueue()) for _ in range(n_envs)]
    print('pool')
    # SUBPROCPOOL = Pool(n_envs, subproc_initializer, (queues,))
    SUBPROCPOOL = Pool(n_envs)
    print('map')
    import threading
    global MAP_THREAD
    def run_map():
        print('RUN MAP')
        # SUBPROCPOOL.map(process_subproc_wrapper, list(range(n_envs)))
        SUBPROCPOOL.map(process_subproc_wrapper, enumerate(queues), 1)
    print('starting thread')
    MAP_THREAD = threading.Thread(target=run_map)
    MAP_THREAD.start()
    print('Checking...')
    for i, (_, returns) in enumerate(queues):
        print('Checking queue', i, n_envs)
        assert returns.get() == 'READY'
        print('Checking queue done')
    print('DONE')
    return queues


def subproc_wrapper_skip_return(location, name):
    # This is to make functions for which skipping return is OK faster.
    if location == ['unwrapped', 'ale'] and name == 'act':
        return True
    return False


def subproc_wrapper_get_from_dummy(location, name):
    if location == ['unwrapped', 'ale'] and name == 'lives':
        return True
    if name in ('action_space', 'observation_space', 'reward_range', 'metadata'):
        return True
    if location == ['unwrapped'] and name == '_action_set':
        return True


SUBPROC_QUEUES = None

def subproc_initializer(queues):
    print('INITIALIZING')
    global SUBPROC_QUEUES
    SUBPROC_QUEUES = queues


def process_subproc_wrapper(args):
    i, (instructions, returns) = args
    print('LOOPING', i)
    # instructions, returns = SUBPROC_QUEUES[i]
    assert instructions is not None
    assert returns is not None
    returns.put('READY')
    print('READY')
    create_env, create_args, create_kwargs = instructions.get()
    env = create_env(*create_args, **create_kwargs)
    print('ENV')
    while True:
        typ, location, name, args, kwargs = instructions.get()
        print(typ, location, name)
        cur_obj = env
        for sub in location:
            cur_obj = getattr(cur_obj, sub)
        if typ == 'fn':
            # if name != 'act':
            #     print('Calling', name)
            res = getattr(cur_obj, name)(*args, **kwargs)
            if not subproc_wrapper_skip_return(location, name):
                returns.put(res)
            # if name != 'act':
            #     print('Success calling', name)
        elif typ == 'at':
            returns.put(getattr(cur_obj, name))


class SubprocFunctor:
    def __init__(self, name, instructions, returns, location):
        self.name = name
        self.instructions = instructions
        self.returns = returns
        self.location = location

    def __call__(self, *args, **kwargs):
        self.instructions.put(('fn', self.location, self.name, args, kwargs))
        if subproc_wrapper_skip_return(self.location, self.name):
            ret = None
        else:
            # print('Calling', self.location, self.name)
            ret = self.returns.get()
            # print('Calling', self.location, self.name, 'returned to master')
        return ret


class SubprocObject:
    def __getattr__(self, item):
        if subproc_wrapper_get_from_dummy(self.location, item):
            # print('Skipping multiprocessing for', self.location, item)
            return getattr(self.dummy_env, item)

        # print('Trying to find IN', self.location, item)
        if not callable(getattr(self.dummy_env, item)):
            self.instructions.put(('at', self.location, item, None, None))
            ret = self.returns.get()
        else:
            ret = SubprocFunctor(item, self.instructions, self.returns, self.location)
        # print('Got value IN', self.location, item)
        return ret


class SubprocUnwrapped(SubprocObject):
    def __init__(self, instructions, returns, dummy_env, location):
        self.instructions = instructions
        self.returns = returns
        self.dummy_env = dummy_env
        self.location = location
        if hasattr(dummy_env, 'ale'):
            self.ale = SubprocUnwrapped(instructions, returns, dummy_env.ale, location + ['ale'])


class SubprocWrapper(SubprocObject):
    def __init__(self, create_env, create_args, create_kwargs, queues):
        import goexplore_py.complex_fetch_env
        assert create_env is goexplore_py.complex_fetch_env.ComplexFetchEnv, \
            'Envs other than fetch not yet supported (some optimizations assume fetch env, would need to reduce get_from_dummy to handle other envs).'
        create_args = create_args or []
        create_kwargs = create_kwargs or {}

        self.instructions, self.returns = queues
        self.instructions.put((create_env, create_args, create_kwargs))
        self.location = []

        self.dummy_env = create_env(*create_args, **create_kwargs)

        self.unwrapped = SubprocUnwrapped(self.instructions, self.returns, self.dummy_env.unwrapped, ['unwrapped'])


class MyWrapper(gym.Wrapper):
    def __init__(self, env):
        super(MyWrapper, self).__init__(env)

    def decrement_starting_point(self, nr_steps, idx):
        return self.env.decrement_starting_point(nr_steps, idx)

    def recursive_getattr(self, name):
        if hasattr(self, name):
            return getattr(self, name)
        else:
            try:
                return self.env.recursive_getattr(name)
            except AttributeError:
                raise Exception(f'Couldn\'t get attr: {name}')

    def batch_reset(self):
        global reset_for_batch
        reset_for_batch = True
        obs = self.env.reset()
        reset_for_batch = False
        return obs

    def reset(self):

        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def step_async(self, actions):
        return self.env.step_async(actions)

    def step_wait(self):
        return self.env.step_wait()

    def reset_task(self):
        return self.env.reset_task()

    @property
    def num_envs(self):
        return self.env.num_envs


class SuperDumbVenvWrapper:
    def __init__(self, venv):
        self.venv = venv

    def __getattr__(self, item):
        return getattr(self.venv, item)


class VecFrameStack(MyWrapper):
    """
    Vectorized environment base class
    """
    def __init__(self, venv, nstack):
        self.venv = venv
        self.nstack = nstack
        wos = venv.observation_space # wrapped ob space
        low = np.repeat(wos.low, self.nstack, axis=-1)
        high = np.repeat(wos.high, self.nstack, axis=-1)
        self.stackedobs = np.zeros((venv.num_envs,)+low.shape, low.dtype)
        self._observation_space = spaces.Box(low=low, high=high)
        self._action_space = venv.action_space

    def step(self, vac):
        """
        Apply sequence of actions to sequence of environments
        actions -> (observations, rewards, news)
        where 'news' is a boolean vector indicating whether each element is new.
        """
        obs, rews, news, infos = self.venv.step(vac)
        # self.stackedobs = np.roll(self.stackedobs, shift=-1, axis=-1)  # Old code
        self.stackedobs = np.roll(self.stackedobs, shift=-obs.shape[-1], axis=-1)  # New code
        for (i, new) in enumerate(news):
            if new:
                self.stackedobs[i] = 0
        self.stackedobs[..., -obs.shape[-1]:] = obs
        return self.stackedobs, rews, news, infos

    def reset(self):
        """
        Reset all environments
        """
        obs = self.venv.reset()
        self.stackedobs[...] = 0
        self.stackedobs[..., -obs.shape[-1]:] = obs
        return self.stackedobs

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    def close(self):
        self.venv.close()

    @property
    def num_envs(self):
        return self.venv.num_envs


N_PROFILE = 1
from collections import defaultdict
PROFILE_DATA = defaultdict(list)

class Profile:
    def __init__(self, name):
        self.start = None
        self.name = name

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        total = (time.time() - self.start) * 1000
        PROFILE_DATA[self.name].append(total)
        if len(PROFILE_DATA[self.name]) > N_PROFILE:
            print(self.name, np.mean(PROFILE_DATA[self.name]))
            PROFILE_DATA[self.name] = []


def mydownscale(state, target_shape, max_pix_value):
    import cv2
    state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    return imdownscale(state, target_shape, max_pix_value).tobytes()


class DemoReplayInfo:
    def __init__(self, env, args, demo_file_name, seed, workers_per_sp, frameskip, transform_reward):
        # Added to allow for the creation of "fake" replay information
        self.args = args
        self.demo_file_name = demo_file_name
        if demo_file_name is None:
            self.actions = None
            self.returns = [0]
            self.rewards = []
            self.checkpoints = None
            self.checkpoint_action_nr = None
            self.starting_point = 0
            self.starting_point_current_ep = None
            self.states_cache = None
            self.obs = None
            self.target = None
            self.max_returns = None
            self.discounted_rewards = None
            self.state_checkpoints = []
            self.reward_augmentations = []
            self.next_state_checkpoint_idx = 0
        else:
            with open(demo_file_name, "rb") as f:
                dat = pickle.load(f)
            self.actions = dat['actions'][:args.max_demo_len]
            rewards = dat['rewards'][:args.max_demo_len]
            self.rewards = rewards
            assert len(rewards) == len(self.actions)
            self.returns = np.cumsum(rewards)
            self.max_returns = np.max(self.returns)
            self.checkpoints = dat['checkpoints']
            self.checkpoint_action_nr = dat['checkpoint_action_nr']
            self.starting_point = max(len(self.actions) - 1 - seed//workers_per_sp, 0)
            self.starting_point_current_ep = None
            self.obs = dat.get('obs')
            self.target = dat.get('target')

            cells = dat.get('cells', [])
            self.cells_at = cells
            self.target_cells = []
            for i in range(0, len(cells)):
                cell = cells[i]
                if cell not in self.target_cells[-args.cell_reward_window:] and (not args.cell_from_demo_unique or cell not in self.target_cells):
                    self.target_cells.append(cell)
            self.current_target_idx = 0
            self.target_idx_for_starting_point = []
            self.reward_augmentations = []
            done = False
            if hasattr(env, 'target_location'):
                old_target_location = env.target_location
                env.target_location = dat.get('target')
            for i, action in enumerate(self.actions):
                assert not done
                if self.args.cell_rewards == 0:
                    self.reward_augmentations.append(0.0)
                else:
                    state, reward, done, _ = env.step(action)
                    assert reward == self.rewards[i]
                    reward_augmentation = self.get_reward_augmentation_and_move_window(state)

                    self.reward_augmentations.append(reward_augmentation)
                self.target_idx_for_starting_point.append(self.current_target_idx)
            if hasattr(env, 'target_location'):
                env.target_location = old_target_location

            self.current_target_idx = 0

            # print(f'Found {len(self.target_cells)} checkpoints for a trajectory of length {len(self.reward_augmentations)} ({len(self.target_cells) * 100 / len(self.reward_augmentations):.1f}%)')

            self.discounted_rewards = [None] * len(self.rewards)
            for cur_skip in range(frameskip):
                for i in reversed(range(cur_skip, len(self.discounted_rewards), frameskip)):
                    discounted_next = 0 if i + frameskip >= len(self.discounted_rewards) else self.discounted_rewards[i + frameskip]
                    self.discounted_rewards[i] = (transform_reward(sum(self.rewards[i:i + frameskip])) + sum(self.reward_augmentations[i:i+frameskip])) + args.gamma * discounted_next
            assert None not in self.discounted_rewards

    def get_reward_augmentation_and_move_window(self, state):
        reward_augmentation = 0.0
        if self.args.cell_rewards == 0:
            return 0.0
        while self.current_target_idx < len(self.target_cells):
            window = self.target_cells[self.current_target_idx:self.current_target_idx + self.args.cell_reward_window]
            for i, (params, window_cell) in enumerate(window):
                mycell = mydownscale(state, *params)
                if mycell == window_cell:
                    self.current_target_idx += i + 1
                    reward_augmentation += (i + 1) * self.args.cell_rewards
                    break
            else:  # Else occurs if we DIDN'T break, ie if we didn't find anything in the window, so we can give up.
                break
        return reward_augmentation


class ReplayResetEnv(MyWrapper):
    """
        Randomly resets to states from a replay
    """

    def __init__(self,
                 env,
                 args,
                 seed,
                 workers_per_sp,
                 is_extra_sil,
                 frameskip,
                 frac_sample=0.2,
                 ):
        super(ReplayResetEnv, self).__init__(env)
        assert args.gamma is not None
        assert args.scale_rewards != -1
        assert args.clip_rewards is not None
        assert frameskip is not None
        assert not (args.clip_rewards and args.scale_rewards), "Clipping and scaling rewards makes no sense"
        assert not (args.test_from_start and args.only_forward), "test_from_start is useless if only_forward is on"
        assert args.only_forward is not None
        assert args.cell_rewards is not None
        assert args.cell_reward_window is not None

        self.args = args

        def transform_reward(r):
            if self.args.clip_rewards:
                r = np.clip(r, -1, 1)
            elif self.args.scale_rewards is not None:
                r *= self.args.scale_rewards
            return r

        # Get the initial state with seed 0
        self.env.seed(0)
        self.env.reset()
        self.initial_state = self.env.unwrapped.clone_state()

        self.frameskip = frameskip

        # TODO: using os.getpid for the random seed makes this non-reproducible, but .seed is used for too many other
        # things. Separate the meaning of seed.
        ns = int(time.time() * 1e9) % int(1e9)
        seed_max = 2**32 - 1
        self.rng = np.random.RandomState((os.getpid() * ns) % seed_max) #seed)
        np.random.seed((os.getpid() * ns * 2) % seed_max)
        random.seed((os.getpid() * ns * 3) % seed_max)
        self.actions_to_overwrite = []
        self.frac_sample = frac_sample
        self.demo_replay_info = []
        self.demo_action = None
        self.is_extra_sil=is_extra_sil
        self.steps_since_last_reward = 0
        if args.test_from_start:
            self.env.reset()
            self.env.unwrapped.restore_state(self.initial_state)
            self.demo_replay_info.append(DemoReplayInfo(env, args, None, seed, workers_per_sp, frameskip, transform_reward))
        if os.path.isdir(args.demo):
            import glob
            for f in sorted(glob.glob(args.demo + '/*.demo')):
                self.env.reset()
                self.env.unwrapped.restore_state(self.initial_state)
                self.demo_replay_info.append(DemoReplayInfo(env, args, f, seed, workers_per_sp, frameskip, transform_reward))
        elif args.demo != '__nodemo__':
            self.env.reset()
            self.env.unwrapped.restore_state(self.initial_state)
            self.demo_replay_info.append(DemoReplayInfo(env, args, args.demo, seed, workers_per_sp, frameskip, transform_reward))
        else:
            assert args.test_from_start
            assert not is_extra_sil
        self.best_demo_idx = np.argmax([(e.returns[-1] if e.demo_file_name is not None else - float('inf')) for e in self.demo_replay_info])
        self.max_demo_reward_interval = 1
        for demo in self.demo_replay_info:
            cur_interval = 0
            for reward in demo.rewards:
                cur_interval += 1
                if cur_interval > self.max_demo_reward_interval:
                    self.max_demo_reward_interval = cur_interval
                if reward != 0:
                    cur_interval = 0
        # print('Max demo reward interval:', self.max_demo_reward_interval)
        # print(f'Best demo: {self.demo_replay_info[self.best_demo_idx].demo_file_name} ({self.best_demo_idx}) with score of {self.demo_replay_info[self.best_demo_idx].returns[-1]}')
        self.cur_demo_replay = None
        self.cur_demo_idx = -1
        self.extra_frames_counter = -1
        self.action_nr = -1
        self.score = -1
        self.intrinsic_score = -1
        self.is_fake_target = False
        self.infinite_window_size = False
        if not args.avg_frames_window_size > 0:
            self.args.avg_frames_window_size = 1
            self.infinite_window_size = True
        self.times_demos_chosen = np.zeros(len(self.demo_replay_info), dtype=np.int)
        self.steps_taken_per_demo = np.zeros((len(self.demo_replay_info), self.args.avg_frames_window_size), dtype=np.int)
        self.target_masks = defaultdict(lambda: np.zeros(len(self.demo_replay_info), dtype=np.bool))
        self.possible_targets = []
        for i in range(len(self.demo_replay_info)):
            if args.from_start_prior > 0 and args.test_from_start and i == 0:
                self.steps_taken_per_demo[i, :] = args.from_start_prior
                self.times_demos_chosen[i] = self.args.avg_frames_window_size
            else:
                self.steps_taken_per_demo[i, 0] = 1

            target = self.demo_replay_info[i].target
            if args.test_from_start and i == 0:
                self.target_masks[(True, target)][i] = True
            else:
                self.target_masks[(False, target)][i] = True
                if target is not None and target not in self.possible_targets:
                    self.possible_targets.append(target)
        # print(f'Found {len(self.target_masks)} categories: {self.target_masks.keys()}.')


    def recursive_getattr(self, name):
        prefix = 'starting_point_'
        if name[:len(prefix)] == prefix:
            idx = int(name[len(prefix):])
            return self.demo_replay_info[idx].starting_point
        elif name == 'n_demos':
            # print('Received n_demos, ret', len(self.demo_replay_info))
            return len(self.demo_replay_info)
        else:
            return super(ReplayResetEnv, self).recursive_getattr(name)

    @property
    def demo_targets(self):
        return [demo.target for demo in self.demo_replay_info]

    def _get_window_index(self):
        window_index = (self.times_demos_chosen[self.cur_demo_idx] - 1) % self.args.avg_frames_window_size
        assert window_index >= 0
        assert window_index < self.args.avg_frames_window_size
        return window_index

    def _substep(self, action):
        prev_lives = self.env.unwrapped.ale.lives()
        obs, reward, done, info = self.env.step(action)
        if self.args.correct_score_counter and reward < -900_000:
            # We assume that if the reward is less than -900_000, it is actually due to the score counter rolling over, and compensate accordingly
            reward += 1_000_000
        lives = self.env.unwrapped.ale.lives()
        result = (obs, reward, done, info, prev_lives, lives)

        return result

    def _get_cell(self):
        return self.env._get_state()

    def step(self, action):
        took_extra_sil_action = False
        if len(self.actions_to_overwrite) > 0:
            action = self.actions_to_overwrite.pop(0)
            valid = False
        else:
            valid = True
            if self.is_extra_sil:
                took_extra_sil_action = True
                # TODO: maybe implement caching of state for this?
                action = self.demo_action['action']
                valid = False

        # TODO: get from cache?
        obs, reward, done, info, prev_lives, lives = self._substep(action)
        if 'fetch' in self.args.game:
            info['env_cell'] = self._get_cell()
        # if took_extra_sil_action:
        #     print(self.cur_demo_idx, self.demo_action['action_idx'] + 1, np.allclose(obs, self.cur_demo_replay.obs[self.demo_action['action_idx'] + 1]))
        #     print()

        info['idx'] = self.cur_demo_idx
        self.steps_taken_per_demo[self.cur_demo_idx, self._get_window_index()] += 1
        if reward == 0:
            self.steps_since_last_reward += 1
        else:
            self.steps_since_last_reward = 0
        self.action_nr += 1
        self.score += reward
        # if done and self.score >= self.cur_demo_replay.returns[-1]:
        #     print(f'Normal done! {self.score}')

        done_reason = ['done'] if done else []

        # game over on loss of life, to speed up learning
        if self.args.game_over_on_life_loss:
            if lives < prev_lives and lives > 0:
                # print(f"Done because of lives: {prev_lives} -> {lives}")
                done = True
                done_reason.append('lives')

        if self.args.only_forward:
            pass
        elif (self.args.test_from_start and self.cur_demo_idx == 0):
            if self.steps_since_last_reward > self.max_demo_reward_interval * self.args.from_start_demo_reward_interval_factor:
                done_reason.append('slowstart')
                done = True
        # kill if we have achieved the final score, or if we're laggging the demo too much
        elif self.score >= self.cur_demo_replay.max_returns:
            self.extra_frames_counter -= 1
            if self.extra_frames_counter <= 0:
                # print(f'Done due to score >= max_returns: {self.score} >= {self.cur_demo_replay.max_returns}')
                done = True
                done_reason.append('success')
                info['replay_reset.random_reset'] = True # to distinguish from actual game over
        elif self.action_nr > self.args.allowed_lag:  # TODO: always run till done???
            min_index = self.action_nr - self.args.allowed_lag
            if min_index < 0:
                min_index = 0
            if min_index >= len(self.cur_demo_replay.returns):
                min_index = len(self.cur_demo_replay.returns) - 1
            max_index = self.action_nr + self.args.allowed_lag
            threshold = min(self.cur_demo_replay.returns[min_index: max_index]) - self.args.allowed_score_deficit
            if self.score < threshold:
                done_reason.append('badscore')
                # print(f'Done due to low score: {self.score} < {threshold}')
                done = True

        # output flag to increase entropy if near the starting point of this episode
        if self.action_nr < self.cur_demo_replay.starting_point + self.args.inc_entropy_threshold and not self.is_extra_sil:
            info['increase_entropy'] = True

        if not valid:
            info['replay_reset.invalid_transition'] = True

            if took_extra_sil_action:
                assert reward == self.cur_demo_replay.rewards[self.demo_action['action_idx']]
                info['replay_reset.demo_action'] = self.demo_action

                cur_action_idx = self.demo_action['action_idx'] + 1
                if (cur_action_idx < self.cur_demo_replay.starting_point + self.args.nenvs or self.args.sil_finish_demo) and cur_action_idx < len(self.cur_demo_replay.actions):
                        self.demo_action = {
                        'action': self.cur_demo_replay.actions[cur_action_idx],
                        'discounted_rewards': self.cur_demo_replay.discounted_rewards[cur_action_idx],#self.get_discounted_rewards(self.cur_demo_replay.rewards[cur_action_idx:]),
                        'expected_state': self.cur_demo_replay.obs[cur_action_idx] if self.cur_demo_replay.obs is not None else None,
                        'action_idx': cur_action_idx,
                        'demo_idx': self.cur_demo_idx
                    }
                else:
                    self.demo_action = None
                    done = True
                    info['replay_reset.random_reset'] = True  # to distinguish from actual game over
        else:
            info['replay_reset.demo_action'] = self.demo_action
            self.demo_action = None

        if done and self.is_extra_sil:
            info['sil_episode'] = {
                'from_start': self.cur_demo_replay.starting_point_current_ep == 0,
                'nsteps': cur_action_idx - self.cur_demo_replay.starting_point_current_ep
            }

        if done and not self.is_extra_sil and not self.is_fake_target:
            ep_info = {'l': self.action_nr,
                       'as_good_as_demo': (self.score >= (self.cur_demo_replay.returns[-1] - self.args.allowed_score_deficit)),
                       'r': self.score,
                       'intrinsic_score': self.intrinsic_score,
                       'starting_point': self.cur_demo_replay.starting_point_current_ep,
                       'idx': self.cur_demo_idx,
                       'done_reasons': info.get('done_reasons', []) + done_reason,
                       'is_extra_sil': self.is_extra_sil
                       }
            info['episode'] = ep_info

        if self.args.scale_rewards is not None:
            reward *= self.args.scale_rewards
        elif self.args.clip_rewards:
            reward = np.sign(reward)

        intrinsic_reward = self.check_intrinsic_reward(obs)
        reward += intrinsic_reward

        self.intrinsic_score += intrinsic_reward

        return obs, reward, done, info

    def check_intrinsic_reward(self, obs):
        if self.args.cell_rewards == 0:
            return 0
        return self.cur_demo_replay.get_reward_augmentation_and_move_window(obs)

    def decrement_starting_point(self, nr_steps, demo_idx):
        if self.demo_replay_info[demo_idx].starting_point>0:
            self.demo_replay_info[demo_idx].starting_point = int(np.maximum(self.demo_replay_info[demo_idx].starting_point - nr_steps, 0))

    @property
    def expected_steps(self):
        ones = np.ones(len(self.demo_replay_info))
        norm = np.where(self.times_demos_chosen == 0, ones, self.times_demos_chosen)
        if not self.infinite_window_size:
            norm = np.where(norm > self.args.avg_frames_window_size, self.args.avg_frames_window_size, norm)
        expected_steps = np.sum(self.steps_taken_per_demo, axis=1) / norm
        return expected_steps

    def reset(self):
        start = time.time()
        self.steps_since_last_reward = 0
        self.is_fake_target = False
        obs = self.env.reset()
        if self.args.always_run_till_done:
            self.extra_frames_counter = 100000000
        else:
            self.extra_frames_counter = int(np.exp(self.rng.rand()*self.args.extra_frames_exp_factor))

        # Select demo
        expected_steps = self.expected_steps
        inverse_expected = 1 / expected_steps
        if self.args.demo_selection == 'normalize_from_start':
            logits = inverse_expected
            logits[1:] = np.mean(logits[1:])
        elif self.args.demo_selection == 'normalize_by_target':
            logits = inverse_expected
            for mask in self.target_masks.values():
                logits[mask] = np.mean(logits[mask])
        elif self.args.demo_selection == 'normalize':
            logits = inverse_expected
        elif self.args.demo_selection == 'uniform':
            logits = np.ones(len(self.demo_replay_info))
        else:
            raise NotImplementedError(f"Unknown operation: {self.args.demo_selection}")
        if self.args.from_start_proportion is not None and self.args.test_from_start:
            logits[0] = logits[0] * len(logits) * self.args.from_start_proportion
        logits = logits / logits.sum()
        # np.set_printoptions(suppress=True)
        # print('expected_steps:', expected_steps)
        # print('logits:', logits)
        extra_sil_starting_point = None
        while True:
            self.cur_demo_idx = np.random.choice(len(self.demo_replay_info), p=logits)
            # Keep choosing demo until we do NOT have a fake demo if we are in is_extra_sil
            if self.is_extra_sil and self.demo_replay_info[self.cur_demo_idx].actions is None:
                continue
            self.cur_demo_replay = self.demo_replay_info[self.cur_demo_idx]
            if self.is_extra_sil:
                if self.args.extra_sil_from_start_prob < 0:
                    assert False, 'extra_sil_from_start_prob < 0 is deprecated'
                    extra_sil_starting_point = self.cur_demo_replay.starting_point + 100
                    # TODO: this is basically complete bullshit
                    while extra_sil_starting_point > self.cur_demo_replay.starting_point:
                        extra_sil_starting_point = np.random.geometric(min(2 / (self.cur_demo_replay.starting_point + 1), 0.5)) - 1  # TODO: make this value customizable
                else:
                    # We provisionally select NOT from start
                    extra_sil_starting_point = max(0, self.cur_demo_replay.starting_point - random.randint(0, min(self.args.extra_sil_before_demo_max, self.cur_demo_replay.starting_point)))

                    # We compute the number of steps if we were to start from the beginning or from the previously found starting point
                    if self.args.sil_from_start_only_best_demo:
                        best_demo = self.demo_replay_info[self.best_demo_idx]
                        if self.args.sil_finish_demo:
                            max_steps_sil = len(best_demo.actions)
                        else:
                            max_steps_sil = min(best_demo.starting_point + self.args.nenvs, len(best_demo.actions))
                    else:
                        if self.args.sil_finish_demo:
                            max_steps_sil = len(self.cur_demo_replay.actions)
                        else:
                            max_steps_sil = min(self.cur_demo_replay.starting_point + self.args.nenvs, len(self.cur_demo_replay.actions))
                    num_steps_not_from_start = max_steps_sil - extra_sil_starting_point

                    # Random threshold: if we do the probability is by selection, we simply use extra_sil_from_start_prob, otherwise,
                    # we normalize so that we expect to spend the same number of steps starting from the beginning vs not, adjusted
                    # by extra_sil_from_start_prob.
                    from_start_prob = self.args.extra_sil_from_start_prob
                    if not self.args.sil_from_start_prob_by_selection:
                        from_start_prob = self.args.extra_sil_from_start_prob * num_steps_not_from_start / (max_steps_sil * (1-self.args.extra_sil_from_start_prob) + self.args.extra_sil_from_start_prob * num_steps_not_from_start)

                    # Finally, we randomly select whether to do from start
                    if random.random() < from_start_prob:
                        extra_sil_starting_point = 0
                        if self.args.sil_from_start_only_best_demo:
                            self.cur_demo_idx = self.best_demo_idx
                            self.cur_demo_replay = self.demo_replay_info[self.cur_demo_idx]
            break
        self.times_demos_chosen[self.cur_demo_idx] += 1
        if not self.infinite_window_size:
            self.steps_taken_per_demo[self.cur_demo_idx, self._get_window_index()] = 0
        if self.cur_demo_replay.target is not None:
            assert hasattr(self.env, 'target_location')
            self.env.target_location = self.cur_demo_replay.target
            if not self.is_extra_sil and self.rng.rand() < self.args.fake_target_probability and len(set(self.possible_targets)) > 1:
                while self.env.target_location == self.cur_demo_replay.target:
                    self.env.target_location = self.rng.choice(self.possible_targets)
                self.is_fake_target = True
        elif self.cur_demo_idx == 0 and self.args.test_from_start and len(self.possible_targets) >= 1:
            self.env.target_location = self.rng.choice(self.possible_targets)

        # Select starting point
        if (self.args.test_from_start and self.cur_demo_idx == 0) or self.args.only_forward:
            for demo in self.demo_replay_info:
                demo.next_state_checkpoint_idx = 0
            self.cur_demo_replay.starting_point_current_ep = 0
            self.cur_demo_replay.current_target_idx = 0
            self.actions_to_overwrite = []
            self.action_nr = 0
            self.score = 0
            self.intrinsic_score = 0
            obs = self.env.reset()
            if self.args.noops and not self.is_extra_sil:
                self.demo_action = None
                noops = random.randint(0, 30)
                for _ in range(noops):
                    obs, _, _, _ = self.env.step(0)
            elif self.is_extra_sil:
                self.demo_action = {
                    'action': self.cur_demo_replay.actions[0],
                    'discounted_rewards': self.cur_demo_replay.discounted_rewards[0],#self.get_discounted_rewards(self.cur_demo_replay.rewards[self.cur_demo_replay.starting_point_current_ep:]),
                    'expected_state': self.cur_demo_replay.obs[0] if self.cur_demo_replay.obs is not None else None,
                    'action_idx': 0,
                    'demo_idx': self.cur_demo_idx
                }
            return obs

        elif reset_for_batch:
            self.cur_demo_replay.starting_point_current_ep = 0
            self.cur_demo_replay.current_target_idx = 0
            self.actions_to_overwrite = self.cur_demo_replay.actions[:]
            self.action_nr = 0
            self.score = 0
            self.intrinsic_score = 0
            self.env.unwrapped.restore_state(self.initial_state)
        else:
            # print("Total number of actions:", len(self.cur_demo_replay.actions))
            if self.is_extra_sil:
                self.cur_demo_replay.starting_point_current_ep = extra_sil_starting_point
            elif self.rng.rand() <= 1.-self.frac_sample:
                self.cur_demo_replay.starting_point_current_ep = self.cur_demo_replay.starting_point
            else:
                self.cur_demo_replay.starting_point_current_ep = self.rng.randint(low=self.cur_demo_replay.starting_point, high=len(self.cur_demo_replay.actions))

            start_action_nr = 0
            start_ckpt = None
            for nr, ckpt in zip(self.cur_demo_replay.checkpoint_action_nr[::-1], self.cur_demo_replay.checkpoints[::-1]):
                if nr < (self.cur_demo_replay.starting_point_current_ep - self.args.start_frame):
                    start_action_nr = nr
                    start_ckpt = ckpt
                    break
            if start_action_nr > 0:
                self.env.unwrapped.restore_state(start_ckpt)
            else:
                # Note: this is because some environments use the random seed to set their initial state, so we need this to be the same.
                # This is necessary no matter what because of SIL.
                self.env.unwrapped.restore_state(self.initial_state)
            nr_to_start_lstm = np.maximum(self.cur_demo_replay.starting_point_current_ep - self.args.start_frame, start_action_nr)
            assert nr_to_start_lstm>start_action_nr or nr_to_start_lstm == 0
            if nr_to_start_lstm>start_action_nr:
                for a in self.cur_demo_replay.actions[start_action_nr:nr_to_start_lstm]:
                    action = self.env.unwrapped._action_set[a]
                    self.env.unwrapped.ale.act(action)
            self.cur_demo_replay.actions_to_overwrite = self.cur_demo_replay.actions[nr_to_start_lstm:self.cur_demo_replay.starting_point_current_ep]
            if not self.is_fake_target and self.cur_demo_replay.actions is not None:
                self.demo_action = {
                    'action': self.cur_demo_replay.actions[self.cur_demo_replay.starting_point_current_ep],
                    'discounted_rewards': self.cur_demo_replay.discounted_rewards[self.cur_demo_replay.starting_point_current_ep],#self.get_discounted_rewards(self.cur_demo_replay.rewards[self.cur_demo_replay.starting_point_current_ep:]),
                    'expected_state': self.cur_demo_replay.obs[self.cur_demo_replay.starting_point_current_ep] if self.cur_demo_replay.obs is not None else None,
                    'action_idx': self.cur_demo_replay.starting_point_current_ep,
                    'demo_idx': self.cur_demo_idx
                }
            else:
                self.demo_action = None
            if nr_to_start_lstm>0:
                obs = self.env.unwrapped._get_image()
            self.action_nr = nr_to_start_lstm
            # TODO: check that it is indeed correct to use nr_to_start_lstm here and not starting_point_current_ep
            self.cur_demo_replay.current_target_idx = self.cur_demo_replay.target_idx_for_starting_point[nr_to_start_lstm]
            self.score = self.cur_demo_replay.returns[nr_to_start_lstm - 1] if nr_to_start_lstm > 0 else 0
            # TODO: reasonable value for this
            self.intrinsic_score = 0
            if not self.is_extra_sil and self.cur_demo_replay.starting_point_current_ep == 0 and self.cur_demo_replay.actions_to_overwrite == []:
                if self.args.test_from_start:
                    # If we start from the very beginning, we don't set a time limit to the demo.
                    # This way more effort is spent fine tuning from the start, and also our graphs
                    # don't look weird.
                    self.extra_frames_counter = 1_000_000_000
                if self.args.noops:
                    noops = random.randint(0, 30)
                    for _ in range(noops):
                        self.demo_action = None
                        obs, _, _, _ = self.env.step(0)

        return obs


class MaxAndSkipEnv(MyWrapper):
    def __init__(self, env, skip=4, maxlen=2):
        """Return only every `skip`-th frame"""
        MyWrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = deque(maxlen=maxlen)
        self._skip       = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        combined_info = {}
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            combined_info.update(info)
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)

        return max_frame, total_reward, done, combined_info

    def reset(self):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs

class ClipRewardEnv(MyWrapper):
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward = np.sign(reward)
        return obs, reward, done, info

class IgnoreNegativeRewardEnv(MyWrapper):
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward = max(reward, 0)
        return obs, reward, done, info

class ScaledRewardEnv(MyWrapper):
    def __init__(self, env, scale=1):
        MyWrapper.__init__(self, env)
        self.scale = scale

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward = reward*self.scale
        return obs, reward, done, info

class EpsGreedyEnv(MyWrapper):
    def __init__(self, env, eps=0.01):
        MyWrapper.__init__(self, env)
        self.eps = eps

    def step(self, action):
        if np.random.uniform()<self.eps:
            action = np.random.randint(self.env.action_space.n)
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info

class StickyActionEnv(MyWrapper):
    def __init__(self, env, p=0.25):
        MyWrapper.__init__(self, env)
        self.p = p
        self.last_action = 0

    def step(self, action):
        if np.random.uniform() < self.p:
            action = self.last_action
        self.last_action = action
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info

class TanhWrap(MyWrapper):
    def step(self, action):
        return self.env.step(np.tanh(action))

class Box(gym.Space):
    """
    A box in R^n.
    I.e., each coordinate is bounded.
    Example usage:
    self.action_space = spaces.Box(low=-10, high=10, shape=(1,))
    """
    def __init__(self, low, high, shape=None, dtype=np.uint8):
        """
        Two kinds of valid input:
            Box(-1.0, 1.0, (3,4)) # low and high are scalars, and shape is provided
            Box(np.array([-1.0,-2.0]), np.array([2.0,4.0])) # low and high are arrays of the same shape
        """
        if shape is None:
            assert low.shape == high.shape
            self.low = low
            self.high = high
        else:
            assert np.isscalar(low) and np.isscalar(high)
            self.low = low + np.zeros(shape)
            self.high = high + np.zeros(shape)
        self.dtype = dtype
    def contains(self, x):
        return x.shape == self.shape and (x >= self.low).all() and (x <= self.high).all()
    def to_jsonable(self, sample_n):
        return np.array(sample_n).tolist()
    def from_jsonable(self, sample_n):
        return [np.asarray(sample) for sample in sample_n]
    @property
    def shape(self):
        return self.low.shape
    @property
    def size(self):
        return self.low.shape
    def __repr__(self):
        return "Box" + str(self.shape)
    def __eq__(self, other):
        return np.allclose(self.low, other.low) and np.allclose(self.high, other.high)

class WarpFrame(MyWrapper):
    def __init__(self, env):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        MyWrapper.__init__(self, env)
        self.res = 84
        self.observation_space = Box(low=0, high=255, shape=(self.res, self.res, 1), dtype = np.uint8)

    def reshape_obs(self, obs):
        obs = np.dot(obs.astype('float32'), np.array([0.299, 0.587, 0.114], 'float32'))
        # print(obs.shape)
        obs = np.array(Image.fromarray(obs).resize((self.res, self.res),
                                                   resample=Image.BILINEAR), dtype=np.uint8)
        # print(obs.shape)
        return obs.reshape((self.res, self.res, 1))

    def reset(self):
        return self.reshape_obs(self.env.reset())

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self.reshape_obs(obs), reward, done, info


class MyResizeFrame(MyWrapper):
    def __init__(self, env):
        """Warp frames to 105x80"""
        MyWrapper.__init__(self, env)
        self.res = (105, 80, 3)
        self.net_res = (self.res[1], self.res[0], self.res[2])
        # self.res = (80, 105, 3)
        self.observation_space = Box(low=0, high=255, shape=self.net_res, dtype=np.uint8)

    def reshape_obs(self, obs):
        # print("MyResizeFrame:", obs.shape)
        obs = np.array(Image.fromarray(obs).resize((self.res[0], self.res[1]),
                                                   resample=Image.BILINEAR), dtype=np.uint8)
        # print("MyResizeFrame:", obs.shape)
        # obs = obs.reshape(self.res)
        # obs = obs.transpose([1, 0, 2])

        # print("MyResizeFrame:", obs.shape)
        return obs
        # return obs.reshape(self.res)

    def reset(self):
        return self.reshape_obs(self.env.reset())

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self.reshape_obs(obs), reward, done, info


class MyResizeFrameOld(MyWrapper):
    def __init__(self, env):
        """Warp frames to 105x80"""
        MyWrapper.__init__(self, env)
        self.res = (105, 80, 3)
        # self.res = (80, 105, 3)
        self.observation_space = Box(low=0, high=255, shape=self.res, dtype=np.uint8)

    def reshape_obs(self, obs):
        obs = np.array(Image.fromarray(obs).resize((self.res[0], self.res[1]),
                                                   resample=Image.BILINEAR), dtype=np.uint8)
        return obs.reshape(self.res)

    def reset(self):
        return self.reshape_obs(self.env.reset())

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self.reshape_obs(obs), reward, done, info


class FireResetEnv(MyWrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        MyWrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self):
        self.env.reset()
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset()
        return obs

class PreventSlugEnv(MyWrapper):
    def __init__(self, env, max_no_rewards=10000):
        """Abort if too much time without getting reward."""
        MyWrapper.__init__(self, env)
        self.last_reward = 0
        self.steps = 0
        self.max_no_rewards = max_no_rewards

    def step(self, *args, **kwargs):
        obs, reward, done, info = self.env.step(*args, **kwargs)
        self.steps += 1
        if reward > 0:
            self.last_reward = self.steps
        if self.steps - self.last_reward > self.max_no_rewards:
            done = True
        return obs, reward, done, info

    def reset(self):
        self.got_reward = False
        self.steps = 0
        return self.env.reset()

class VideoWriter(MyWrapper):
    def __init__(self, env, file_prefix):
        MyWrapper.__init__(self, env)
        self.file_prefix = file_prefix
        fd, self.temp_filename = tempfile.mkstemp('.mp4', 'tmp', '/'.join(self.file_prefix.split('/')[:-1]))
        os.close(fd)
        self.video_writer = None
        self.counter = 0
        self.orig_frames = None
        self.cur_step = 0
        self.is_train = hasattr(self.env, 'demo_replay_info')
        self.score = 0
        self.randval = 0

    def process_frame(self, frame):
        f_out = np.zeros((224, 160, 3), dtype=np.uint8)
        f_out[7:-7, :] = np.cast[np.uint8](frame)
        return f_out

    def get_orig_frame(self, shape):
        if self.orig_frames is None:
            self.orig_frames = {}
            tmp = self.env.env.unwrapped.clone_state()

            self.env.env.reset()
            for i, a in enumerate(self.env.actions):
                self.orig_frames[i + self.env.reset_steps_ignored] = self.env.env.render(mode='rgb_array')
                self.env.unwrapped.ale.act(a)

            self.orig_frames[len(self.env.actions)] = self.env.env.render(mode='rgb_array')

            self.env.env.unwrapped.restore_state(tmp)

        if self.cur_step not in self.orig_frames:
            frame = np.zeros(shape, dtype=np.uint8)
            frame[:, :, 1] = 255
            return frame

        return self.orig_frames[self.cur_step]

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.cur_step += 1

        # if reward <= -999000:
        #     reward = 0
        self.score += reward
        if int(self.score) // SCORE_THRESHOLD != int(self.score - reward) // SCORE_THRESHOLD and not self.is_train:
            self.video_writer.close()
            n_noops = -1
            try:
                n_noops = self.recursive_getattr('cur_noops')
            except Exception:
                pass
            os.rename(self.temp_filename, self.file_prefix + str(n_noops) + '_' + str(self.randval) + '_' + ('%08d' % self.score) + '.mp4')
            self.video_writer = imageio.get_writer(self.temp_filename, mode='I', fps=120)

        # if self.is_train:
        #     orig_frame = self.process_frame(self.get_orig_frame(obs.shape))
        #     cur_frame = self.process_frame(obs)
        #     final_frame = np.zeros((224, 160 * 2 + 16, 3), dtype=np.uint8)
        #     final_frame[:, :160, :] = cur_frame
        #     final_frame[:, 160+16:, :] = orig_frame
        #     self.video_writer.append_data(final_frame)
        #     if done:
        #         if info['episode']['as_good_as_demo']:
        #             color_idx = 1 # Green
        #         else:
        #             color_idx = 0 # Red
        #
        #         frame = np.zeros((224, 160 * 2 + 16, 3), dtype=np.uint8)
        #         frame[:, :160, color_idx] = 255
        #         frame[:, 160+16:, :] = orig_frame
        #         for _ in range(120):
        #             self.video_writer.append_data(frame)
        # else:
        # print("Current step:", self.cur_step)
        self.video_writer.append_data(self.process_frame(obs))
        return obs, reward, done, info

    def reset(self):
        self.score = 0
        if self.video_writer is not None:
            # print("Writing video")
            # traceback.print_stack()
            self.video_writer.close()
            if self.is_train:
                demo_idx = getattr(self.env, 'cur_demo_idx', 0)
                filename = f'{self.file_prefix}_demo-{demo_idx}_start-{self.env.demo_replay_info[demo_idx].starting_point_current_ep if self.is_train else 0}_{random.randint(0, 2)}.mp4'
                os.rename(self.temp_filename, filename)
            else:
                n_noops = -1
                try:
                    n_noops = self.recursive_getattr('cur_noops')
                except Exception:
                    pass
                os.rename(self.temp_filename, self.file_prefix + str(n_noops) + '_'+ str(self.randval) + '.mp4')
            self.counter += 1
        res = self.env.reset()
        self.randval = random.randint(0, 1000000)
        self.video_writer = imageio.get_writer(self.temp_filename, mode='I', fps=120)
        if self.is_train:
            self.cur_step = self.env.demo_replay_info[self.env.cur_demo_idx].starting_point_current_ep
        return res


def my_wrapper(env,
               clip_rewards=True,
               frame_resize_wrapper=MyResizeFrame,
               scale_rewards=None,
               sticky=True):
    assert 'NoFrameskip' in env.spec.id
    # assert not (clip_rewards and scale_rewards), "Clipping and scaling rewards makes no sense"
    # if scale_rewards is not None:
    #     env = ScaledRewardEnv(env, scale_rewards)
    # if clip_rewards:
    #     env = ClipRewardEnv(env)
    if sticky:
        env = StickyActionEnv(env)
    maxlen = 2
    if 'Venture' in env.spec.id or 'Gravitar' in env.spec.id:
        print('Maxing over 4 frames instead of 2 for Venture and Gravitar because some frames flicker at a rate of 1 / 4')
        maxlen = 4
    env = MaxAndSkipEnv(env, skip=4, maxlen=maxlen)
    if 'Pong' in env.spec.id:
        env = FireResetEnv(env)
    env = frame_resize_wrapper(env)
    return env


class ResetDemoInfo:
    def __init__(self, env, idx):
        self.env = env
        self.idx = idx
        self.target = self.env.recursive_getattr('demo_targets')[0][idx]
        starting_points = self.env.recursive_getattr(f'starting_point_{idx}')
        all_starting_points = flatten_lists(MPI.COMM_WORLD.allgather(starting_points))
        self.min_starting_point = min(all_starting_points)
        self.max_starting_point = max(all_starting_points)
        # JH: This seems off-by-one. I believe the number of start steps is supposed to be all start steps in our window
        # but, by taking the difference between the min and the max, it will be the number of steps in our window - 1.
        # Fixing with a + 1.
        self.nrstartsteps = (self.max_starting_point - self.min_starting_point) + 1

        # TODO: Why does the number of start-steps need to be greater than 10?
        # JH: I don't think there is any reason why the number of start-steps (which is basically the window in which
        # we try different starting points) should be larger than 10. The algorithm is expected to work better when the
        # number of start steps is greater than 10, but it should not be an assert.
        # The number of start steps should be greater than 0, to prevent a divide by 0 error though.
        # assert(self.nrstartsteps > 10)
        assert (self.nrstartsteps >= 1)
        self.max_max_starting_point = self.max_starting_point
        # TODO: Why does this vector have 10,000 extra numbers?
        self.starting_point_success = np.zeros(self.max_starting_point+10000)
        self.infos = []


class ResetManager(MyWrapper):
    def __init__(self, env, move_threshold=0.2, steps_per_demo=1024, fast_increase_starting_point=False):
        super(ResetManager, self).__init__(env)
        self.n_demos = self.recursive_getattr('n_demos')[0]
        self.demos = [ResetDemoInfo(self.env, idx) for idx in range(self.n_demos)]
        self.counter = 0
        self.fast_increase_starting_point = fast_increase_starting_point
        self.move_threshold = move_threshold
        self.steps_per_demo = steps_per_demo

    def proc_infos(self):
        # debug = True
        # if debug:
        #     # Count the number of episode frames dedicated to each demo
        #     steps_taken_by_each_worker = self.recursive_getattr('steps_taken_per_demo')
        #     if hvd.size() > 1:
        #         steps_taken_by_each_worker = flatten_lists(steps_taken_by_each_worker)
        #     steps_taken_per_demo = np.sum(sum(steps_taken_by_each_worker), axis=1)
        #     print("steps_taken_per_demo:", steps_taken_per_demo)
        #
        # if debug:
        #     # Count the number of workers currently running the various demos
        #     cur_demo_idx = self.recursive_getattr('cur_demo_idx')
        #     if hvd.size() > 1:
        #         cur_demo_idx = flatten_lists(cur_demo_idx)
        #     current_demo_counts = np.zeros(self.n_demos)
        #     for idx in cur_demo_idx:
        #         current_demo_counts[idx] += 1
        #     print('current_demo_counts:', current_demo_counts)

        for idx in range(self.n_demos):
            epinfos = [info['episode'] for info in self.demos[idx].infos if 'episode' in info]

            import horovod.tensorflow as hvd
            if hvd.size()>1:
                epinfos = flatten_lists(MPI.COMM_WORLD.allgather(epinfos))

            new_sp_wins = {}
            new_sp_counts = {}
            for epinfo in epinfos:
                sp = epinfo['starting_point']
                if sp in new_sp_counts:
                    new_sp_counts[sp] += 1
                    if epinfo['as_good_as_demo']:
                        new_sp_wins[sp] += 1
                else:
                    new_sp_counts[sp] = 1
                    if epinfo['as_good_as_demo']:
                        new_sp_wins[sp] = 1
                    else:
                        new_sp_wins[sp] = 0

            for sp,wins in new_sp_wins.items():
                self.demos[idx].starting_point_success[sp] = np.cast[np.float32](wins)/new_sp_counts[sp]

            # move starting point, ensuring at least 20% of workers are able to complete the demo
            csd = np.argwhere(np.cumsum(self.demos[idx].starting_point_success) / self.demos[idx].nrstartsteps >= self.move_threshold)
            # If fast_increase_starting_point is True, we always increase the starting point using the else clause,
            # not this clause. If it is False, we may increase the starting point using this clause, which is likely
            # slower.
            if len(csd) > 0 and (csd[0][0] <= self.demos[idx].max_starting_point or not self.fast_increase_starting_point):
                new_max_start = csd[0][0]
            else:
                new_max_start = np.minimum(self.demos[idx].max_starting_point + 100, self.demos[idx].max_max_starting_point)
            n_points_to_shift = self.demos[idx].max_starting_point - new_max_start
            self.decrement_starting_point(n_points_to_shift, idx)
            self.demos[idx].infos = []

    def decrement_starting_point(self, n_points_to_shift, idx):
        self.env.decrement_starting_point(n_points_to_shift, idx)
        starting_points = self.env.recursive_getattr(f'starting_point_{idx}')
        all_starting_points = flatten_lists(MPI.COMM_WORLD.allgather(starting_points))
        self.demos[idx].max_starting_point = max(all_starting_points)
        self.demos[idx].min_starting_point = min(all_starting_points)

    def set_max_starting_point(self, starting_point, idx, threshold):
        n_points_to_shift = self.demos[idx].max_starting_point - starting_point
        self.demos[idx].starting_point_success[starting_point:] = threshold
        self.decrement_starting_point(n_points_to_shift, idx)

    def step(self, action):
        obs, rews, news, infos = self.env.step(action)
        for info in infos:
            if info.get('replay_reset.demo_action') is not None:
                idx = info['replay_reset.demo_action']['demo_idx']
                sp = info['replay_reset.demo_action']['action_idx']
                info['replay_reset.demo_action']['reset_manager.starting_point_success'] = self.demos[idx].starting_point_success[sp]
            self.demos[info['idx']].infos.append(info)
        self.counter += 1
        if (self.counter > (self.demos[0].max_max_starting_point - self.demos[0].max_starting_point) / 2 and
                self.counter % (self.steps_per_demo * self.n_demos) == 0):
            self.proc_infos()
        return obs, rews, news, infos

    def step_wait(self):
        obs, rews, news, infos = self.env.step_wait()
        for info in infos:
            self.demos[info['idx']].infos.append(info)
        self.counter += 1
        if self.counter > (self.demos[0].max_max_starting_point - self.demos[0].max_starting_point) / 2 and self.counter % (self.steps_per_demo * self.n_demos) == 0:
            self.proc_infos()
        return obs, rews, news, infos

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]

def worker(remote, env_fn_wrapper):
    import os, cProfile, time
    filename = f'/mnt/share/adrienle/shared_data/subproc_{os.getpid()}'
    # profile = cProfile.Profile()
    env = env_fn_wrapper.x()
    start_time = time.time()
    # profile.enable()
    while True:
        cmd, data = remote.recv().data
        if cmd == 'step':
            pid = os.getpid()
            start = time.time()
            ob, reward, done, info = env.step(data)
            if done:
                ob = env.reset()
            # print('step', pid, (time.time() - start) * 1000, 'ms')

            # print(len(pickle.dumps(TimedPickle((ob, reward, done, info), 'step_data_only', enabled=False))))
            remote.send(TimedPickle((ob, reward, done, info), 'step_data', enabled=False))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.action_space, env.observation_space))
        elif cmd == 'get_history':
            senv = env
            while not hasattr(senv, 'get_history'):
                senv = senv.env
            remote.send(senv.get_history(data))
        elif cmd == 'recursive_getattr':
            remote.send(env.recursive_getattr(data))
        elif cmd == 'decrement_starting_point':
            env.decrement_starting_point(*data)
        else:
            raise NotImplementedError

        # cur_time = time.time()
        # if cur_time - start_time > 10:
        #     profile.dump_stats(filename)

class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

import time

class TimedPickle:
    def __init__(self, data, name, enabled=False):
        self.data = data
        self.name = name
        self.enabled = enabled

    def __getstate__(self):
        return (time.time(), self.data, self.name, self.enabled)

    def __setstate__(self, s):
        tstart, self.data, self.name, self.enabled = s
        if self.enabled:
            print(f'pickle time for {self.name} = {(time.time() - tstart) * 1000} ms')

class SubprocVecEnv(MyWrapper):
    def __init__(self, env_fns):
        """
        envs: list of gym environments to run in subprocesses
        """
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, CloudpickleWrapper(env_fn)))
            for (work_remote, env_fn) in zip(self.work_remotes, env_fns)]
        for p in self.ps:
            p.start()

        self.remotes[0].send(TimedPickle(('get_spaces', None), 'get_spaces'))
        self.action_space, self.observation_space = self.remotes[0].recv()

    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(TimedPickle(('step', action), 'step', enabled=False))
        def convert_data(data):
            ob, rew, done, infos = data
            return ob, rew, done, infos
        results = [convert_data(remote.recv().data) for remote in self.remotes]
        obs, rews, dones, infos = zip(*results)

        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(TimedPickle(('step', action), 'step_async'))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        # raise Exception("From where am I called.")
        # traceback.print_stack()
        for remote in self.remotes:
            remote.send(TimedPickle(('reset', None), 'reset'))
        return np.stack([remote.recv() for remote in self.remotes])

    def reset_task(self):
        for remote in self.remotes:
            remote.send(TimedPickle(('reset_task', None), 'reset_task'))
        return np.stack([remote.recv() for remote in self.remotes])

    def get_history(self, nsteps):
        for remote in self.remotes:
            remote.send(TimedPickle(('get_history', nsteps), 'get_history'))
        results = [remote.recv() for remote in self.remotes]
        obs, acts, dones = zip(*results)
        obs = np.stack(obs)
        acts = np.stack(acts)
        dones = np.stack(dones)
        return obs, acts, dones

    def recursive_getattr(self, name):
        for remote in self.remotes:
            remote.send(TimedPickle(('recursive_getattr',name), 'recursive_getattr'))
        return [remote.recv() for remote in self.remotes]

    def decrement_starting_point(self, n, idx):
        for remote in self.remotes:
            remote.send(TimedPickle(('decrement_starting_point', (n, idx)), 'decrement_starting_point'))

    def close(self):
        for remote in self.remotes:
            remote.send(TimedPickle(('close', None), 'close'))
        for p in self.ps:
            p.join()

    @property
    def num_envs(self):
        return len(self.remotes)


import pickle, pickletools, gzip

class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, cur_noops=0, n_envs=1, save_path=None, num_per_noop=10, unlimited_score=False):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.override_num_noops = None
        self.num_per_noop = num_per_noop
        self.noop_action = 0
        self.unlimited_score = unlimited_score
        self.cur_noops = cur_noops - n_envs
        self.n_envs = n_envs
        self.save_path = save_path
        self.score = 0
        self.levels = 0
        self.in_treasure = False
        self.rewards = []
        self.actions = []
        ns = int(time.time() * 1e9) % int(1e9)
        seed_max = 2**32 - 1
        self.rng = np.random.RandomState((os.getpid() * ns) % seed_max) #seed)
        np.random.seed((os.getpid() * ns * 2) % seed_max)
        random.seed((os.getpid() * ns * 3) % seed_max)
        self.cur_n_steps = 0
        self.episode_start_step = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'
        import uuid
        self.uuid = uuid.uuid4().hex

    def choose_noops(self):
        return self.rng.randint(0, 31)
        # n_done = []
        # for i in range(0, 31):
        #     pickle_path = self.save_path + '/' + str(i) + '.pickle'
        #     n_data = 0
        #     try:
        #         import pickle
        #         n_data = len(pickle.load(open(pickle_path, 'rb')))
        #     except Exception:
        #         pass
        #     n_done.append(n_data)

        # weights = np.array([(max(0.00001, self.num_per_noop - e)) for e in n_done])
        # div = np.sum(weights)
        # if div == 0:
        #     weights = np.array([1 for e in n_done])
        #     div = np.sum(weights)
        # return np.random.choice(list(range(len(n_done))), p=weights/div)

    def _get_filename(self):
        return f'{self.save_path}/{self.uuid}_{os.getpid()}_{id(self)}.pickle'

    def _get_file_data(self):
        filename = self._get_filename()
        try:
            return pickle.load(open(filename, 'rb'))
        except Exception:
            return []

    def _set_file_data(self, data):
        filename = self._get_filename()
        pickle.dump(data, open(filename, 'wb'))

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        noops = self.choose_noops()
        obs = self.env.reset(**kwargs)
        assert noops >= 0
        self.cur_noops = noops
        self.env.cur_noops = noops
        self.score = 0
        self.rewards = []
        self.actions = [self.noop_action] * self.cur_noops
        self.in_treasure = False
        self.levels = 0
        self.episode_start_step = self.cur_n_steps
        data = self._get_file_data()
        data.append({'start_step': self.episode_start_step, 'noops': self.cur_noops})
        self._set_file_data(data)
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        self.cur_n_steps += 1
        a, reward, done, c = self.env.step(ac)
        if reward < -900_000 and self.unlimited_score:
            # We assume that a reward of less than 900k is actually a counter rollover from the counter
            # reaching over 1 million, so we add 1 million to the reward to compensate. This is true in
            # Montezuma.
            reward += 1_000_000
        from collections import Counter
        in_treasure = Counter(a[:, :, 2].flatten()).get(136, 0) > 20_000
        if self.in_treasure and not in_treasure:
            self.levels += 1
        self.in_treasure = in_treasure
        self.actions.append(ac)
        self.rewards.append(reward)
        self.score += reward

        if self.save_path and (done or len(self.actions) % 50_000 == 0):
            prefix = ''
            if done:
                if 'episode' not in c:
                    c['episode'] = {}
                # if 'write_to_pickle' not in c['episode']:
                #     c['episode']['write_to_pickle'] = []
                #     c['episode']['pickle_path'] = pickle_path
                c['episode']['l'] = len(self.actions)
                c['episode']['r'] = self.score
                c['episode']['as_good_as_demo'] = False
                c['episode']['starting_point'] = 0
                c['episode']['idx'] = 0
            else:
                prefix = 'temporary_'
            data = self._get_file_data()
            for k in list(data[-1]):
                if k.startswith('temporary_'):
                    del data[-1][k]
            assert data[-1]['start_step'] == self.episode_start_step
            data[-1][prefix + 'score'] = self.score
            data[-1][prefix + 'levels'] = self.levels
            data[-1][prefix + 'actions'] = gzip.compress(pickletools.optimize(pickle.dumps([int(e) for e in self.actions])))
            self._set_file_data(data)
            # c['episode']['write_to_pickle'].append({'score': self.score, 'levels': self.levels, 'actions': gzip.compress(pickletools.optimize(pickle.dumps([int(e) for e in self.actions])))})
        return a, reward, done, c



class FetchSaveEnv(gym.Wrapper):
    def __init__(self, env, rank, n_ranks, save_path=None, demo_path=None):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.save_path = save_path
        self.demos = [('None', [])]
        self.cur_demo_idx = -1
        self.cur_action_idx = -1
        self.max_steps = 0
        self.rank = rank
        self.n_ranks = n_ranks
        # if demo_path:
        #     print('Getting demos from', demo_path)
        #     self.demos = []
        #     import glob, pickle
        #     for demo in glob.glob(demo_path + '/*.demo'):
        #         suffix = demo.split('/')[-1].split('.')[0]
        #         data = pickle.load(open(demo, 'rb'))
        #         self.demos.append((suffix, data['actions']))
        self.actions = []
        self.rewards = []
        self.writer = None

    def get_frame(self, cur_frame, reward, total, done, is_replaying):
        frame = self.env.render(mode='rgb_array')

        from PIL import Image
        from PIL import ImageFont
        from PIL import ImageDraw

        frame = Image.fromarray(frame).resize((frame.shape[1] * 2, frame.shape[0] * 2), Image.BICUBIC)
        draw = ImageDraw.Draw(frame)
        # font = ImageFont.truetype(<font-file>, <font-size>)
        dir_path = os.path.dirname(os.path.realpath(__file__))

        font = ImageFont.truetype(dir_path + "/../helvetica.ttf", 28)
        # draw.text((x, y),"Sample Text",(r,g,b))
        info_text = f"Frame: {cur_frame}\nScore: {total + reward}"
        # info_text += f'\nGripped: {self.env.unwrapped._get_state().gripped_info}'
        # if is_replaying:
        #     info_text += '\nREPLAYING'
        draw.text((0, 0), info_text, (255, 255, 255), font=font)
        return np.array(frame)

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        obs = self.env.reset(**kwargs)
        import imageio
        self.actions = []
        self.rewards = []
        if self.cur_demo_idx == -1:
            self.cur_demo_idx = 0
            self.cur_action_idx = self.rank * 5
        else:
            self.cur_action_idx += self.n_ranks * 5

        print('Setting large max_steps')
        self.max_steps = 1_000

        self.writer = imageio.get_writer(f'{self.save_path}/{self.demos[self.cur_demo_idx][0]}_{self.cur_action_idx}_{random.randint(0, 10000)}.mp4', fps=1/(0.002 * 20))

        self.writer.append_data(self.get_frame(0, 0, 0, False, True))

        # for i in range(0, self.cur_action_idx):
        #     ac = self.demos[self.cur_demo_idx][1][i]
        #     obs, reward, d, _ = self.env.step(ac)
        #     self.actions.append(ac)
        #     self.rewards.append(reward)
        #     self.writer.append_data(self.get_frame(len(self.actions), reward, sum(self.rewards[:-1]), d, True))
        #     self.max_steps -= 1
        #     assert not d
        return obs

    def step(self, ac):
        a, reward, done, c = self.env.step(ac)

        if reward > 0:
            print('FOUND REWARD', reward)
        self.actions.append(ac)
        self.rewards.append(reward)
        frame = self.get_frame(len(self.actions), reward, sum(self.rewards[:-1]), done, False)
        self.writer.append_data(frame)
        self.max_steps -= 1
        if self.max_steps < 0:
            done = True

        # TODO: REMOVE THIS SILLY LOGIC
        if sum(self.rewards[:-1]) == 1 and sum(self.rewards[-20:-1]) == 0:
            done = True
            print('KILLING BECAUSE REACHED REWARD OF 2. THIS SHOULD BE REMOVED IN THE FUTURE')

        if done:
            for _ in range(100):
                self.writer.append_data(frame)
            self.writer.close()
            import pickle

            # def aslist(l):
            #     if isinstance(l, np.ndarray):
            #         return [aslist(e) for e in l]
            #     if isinstance(l, (np.float16, np.float32, np.float64)):
            #         return float(l)
            #     if isinstance(l, (np.int32)):
            #         return int(l)
            #     return l
            data = {
                'actions': np.array(self.actions).tolist(),
                'rewards': np.array(self.rewards).tolist(),
                'demo': self.demos[self.cur_demo_idx][0],
                'idx': self.cur_action_idx
            }
            import gzip
            pickle.dump(data, open(f'{self.save_path}/{self.demos[self.cur_demo_idx][0]}_{self.cur_action_idx}.pickle', 'wb'))

        return a, reward, done, c



