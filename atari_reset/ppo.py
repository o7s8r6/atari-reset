'''
Proximal policy optimization with a few tricks. Adapted from the implementation in baselines.
'''

import os.path as osp
import os
import sys
import time
import joblib
import numpy as np
from baselines import logger
from collections import deque
import tensorflow as tf
import horovod.tensorflow as hvd
import mpi4py.rc
mpi4py.rc.initialize = False
from mpi4py import MPI
from baselines.common import explained_variance
from baselines.common.mpi_moments import mpi_moments
import json

class Model(object):
    def __init__(self, args, ob_space, ac_space, nenv, nsteps, test_mode):
        sess = tf.get_default_session()

        act_model = args.policy(sess, ob_space, ac_space, nenv, 1, test_mode=test_mode, reuse=False)
        train_model = args.policy(sess, ob_space, ac_space, nenv, nsteps, test_mode=test_mode, reuse=True)

        SIL_A = train_model.pdtype.sample_placeholder([nenv*nsteps], name='sil_action')
        SIL_VALID = tf.placeholder(tf.float32, [nenv*nsteps], name='sil_valid')
        SIL_R = tf.placeholder(tf.float32, [nenv*nsteps], name='sil_return')

        A = train_model.pdtype.sample_placeholder([nenv*nsteps], name='action')
        ADV = tf.placeholder(tf.float32, [nenv*nsteps], name='advantage')
        VALID = tf.placeholder(tf.float32, [nenv*nsteps], name='valid')
        R = tf.placeholder(tf.float32, [nenv*nsteps], name='return')
        OLDNEGLOGPAC = tf.placeholder(tf.float32, [nenv*nsteps], name='neglogprob')
        OLDVPRED = tf.placeholder(tf.float32, [nenv*nsteps], name='valuepred')
        LR = tf.placeholder(tf.float32, [], name='lr')

        neglogpac = train_model.pd.neglogp(A)
        entropy = tf.reduce_mean(VALID * train_model.pd.entropy())
        vpred = train_model.vf

        neglogp_sil_ac = train_model.pd.neglogp(SIL_A)

        sil_pg_value_weight = 1.0
        if args.sil_pg_weight_by_value:
            sil_pg_value_weight = tf.nn.relu(SIL_R - OLDVPRED)
        sil_pg_loss = tf.reduce_mean(neglogp_sil_ac * sil_pg_value_weight * SIL_VALID)
        sil_vf_td = SIL_R - vpred
        if args.sil_vf_relu:
            sil_vf_td = tf.nn.relu(sil_vf_td)
        sil_vf_loss = .5 * tf.reduce_mean(tf.square(sil_vf_td) * SIL_VALID)
        sil_entropy = tf.reduce_mean(SIL_VALID * train_model.pd.entropy())
        sil_loss = sil_pg_loss + args.sil_vf_coef * sil_vf_loss - args.sil_ent_coef * sil_entropy

        sil_valid_min = tf.reduce_min(SIL_VALID)
        sil_valid_max = tf.reduce_max(SIL_VALID)
        sil_valid_mean = tf.reduce_mean(SIL_VALID)

        neglop_sil_min = tf.reduce_min(neglogp_sil_ac)
        neglop_sil_max = tf.reduce_max(neglogp_sil_ac)
        neglop_sil_mean = tf.reduce_mean(neglogp_sil_ac)

        vpredclipped = OLDVPRED + tf.clip_by_value(vpred - OLDVPRED, - args.cliprange, args.cliprange)
        vf_losses1 = tf.square(vpred - R)
        vf_losses2 = tf.square(vpredclipped - R)
        vf_loss = .5 * tf.reduce_mean(VALID * tf.maximum(vf_losses1, vf_losses2))
        ratio = tf.exp(OLDNEGLOGPAC - neglogpac)
        pg_losses = -ADV * ratio
        pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - args.cliprange, 1.0 + args.cliprange)
        pg_loss = tf.reduce_mean(VALID * tf.maximum(pg_losses, pg_losses2))
        mv = tf.reduce_mean(VALID)
        approxkl = .5 * tf.reduce_mean(VALID * tf.square(neglogpac - OLDNEGLOGPAC)) / mv
        clipfrac = tf.reduce_mean(VALID * tf.to_float(tf.greater(tf.abs(ratio - 1.0), args.cliprange))) / mv
        params = tf.trainable_variables()
        l2_loss = .5 * sum([tf.reduce_sum(tf.square(p)) for p in params])
        loss = pg_loss - entropy * args.ent_coef + vf_loss * args.vf_coef + args.l2_coef * l2_loss + args.sil_coef * sil_loss

        opt = tf.train.AdamOptimizer(LR, epsilon=args.adam_epsilon)
        opt = hvd.DistributedOptimizer(opt)
        train_op = opt.minimize(loss)

        def train(lr, obs, returns, advs, masks, actions, values, neglogpacs, valids, increase_ent, sil_actions, sil_rew, sil_valid, states=None):
            td_map = {LR: lr, A: actions, ADV: advs, VALID: valids, R: returns,
                      OLDNEGLOGPAC: neglogpacs, OLDVPRED: values, train_model.E: increase_ent,
                      SIL_A: sil_actions, SIL_R: sil_rew, SIL_VALID: sil_valid}
            if hasattr(train_model, 'X'):
                td_map[train_model.X] = obs
            else:
                train_model.add_X(td_map, obs)
            # print(sil_valid.tolist())
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            return sess.run([pg_loss, vf_loss, l2_loss, entropy, approxkl, clipfrac, sil_pg_loss, sil_vf_loss, sil_loss, sil_entropy, sil_valid_min, sil_valid_max, sil_valid_mean, neglop_sil_min, neglop_sil_max, neglop_sil_mean, train_op], feed_dict=td_map)[:-1]
        self.loss_names = ['policy_loss', 'value_loss', 'l2_loss', 'policy_entropy', 'approxkl', 'clipfrac', 'sil_pg_loss', 'sil_vf_loss', 'sil_loss', 'sil_entropy', 'sil_valid_min', 'sil_valid_max', 'sil_valid_mean', 'neglop_sil_min', 'neglop_sil_max', 'neglop_sil_mean']

        def save(save_path):
            ps = sess.run(params)
            joblib.dump(ps, save_path)

        def load(load_path):
            loaded_params = joblib.load(load_path)
            restores = []
            for p, loaded_p in zip(params, loaded_params):
                restores.append(p.assign(loaded_p))
            sess.run(restores)

        self.train = train
        self.train_model = train_model
        self.act_model = act_model
        self.step = act_model.step
        self.value = act_model.value
        self.initial_state = act_model.initial_state
        self.save = save
        self.load = load
        sess.run(tf.global_variables_initializer())
        if args.load_path and hvd.rank()==0:
            self.load(args.load_path)
        sess.run(hvd.broadcast_global_variables(0))
        tf.get_default_graph().finalize()

class Runner(object):

    def __init__(self, args, env, model, nsteps):
        self.env = env
        self.model = model
        nenv = env.num_envs
        self.gamma = args.gamma
        self.args = args
        self.lam = args.lam
        self.scale_rewards = args.scale_rewards
        self.clip_rewards = args.clip_rewards
        self.norm_adv = args.norm_adv
        self.sil_weight_success_rate = args.sil_weight_success_rate
        self.subtract_rew_avg = args.subtract_rew_avg
        self.nsteps = nsteps
        self.num_steps_to_cut_left = nsteps//2
        self.num_steps_to_cut_right = 0
        if hasattr(model.train_model, 'X'):
            self.X_typename = model.train_model.X.dtype.name
        else:
            self.X_typename = model.train_model.Xs[0].dtype.name
        # obs = [(env.reset())]
        obs = [np.cast[self.X_typename](env.reset())]
        states = [model.initial_state]
        dones = [np.array([False for _ in range(nenv)])]
        random_res = [np.array([False for _ in range(nenv)])]
        # mb_obs, mb_increase_ent, mb_rewards, mb_reward_avg, mb_actions, mb_values, mb_valids, mb_random_resets, mb_dones, mb_neglogpacs, mb_states, mb_sil_actions, mb_sil_rew, mb_sil_valid
        self.mb_stuff = [obs, [np.zeros(obs[0].shape[0], dtype=np.uint8)], [], [], [], [], [], [random_res], dones, [], states, [], [], []]

        self.recent_sil_episodes = deque(maxlen=50_000)
        self.recent_intrinsic_rewards = deque(maxlen=50_000)

        self.prev_cells = [None] * nenv
        self.total_intrinsic_reward = [0.0] * nenv
        import copy
        self.intrinsic_reward_counts = copy.deepcopy(args.im_cells) if args.im_cells is not None else {}
        self.intrinsic_reward_count_updates = {}
        try:
            from goexplore_py.goexplore import GridDimension, GridEquality
            door_dists = GridDimension('door_dists', 0.2,
                                       0.195) if self.args.fetch_target_location == '0001' else GridDimension(
                                           'door_dists', 1000, 500)
            door1_dists = GridDimension('door1_dists', 0.2,
                                        0.195) if self.args.fetch_target_location == '0010' else GridDimension(
                                            'door1_dists', 1000, 500)
            self.grid_resolution = (
                door_dists, door1_dists,
                GridDimension('gripped_info', 1), GridDimension('gripped_pos', 1000, 500),
                GridEquality('object_pos', self.args.fetch_target_location, sort=True),
                GridDimension('gripper_pos', 0.5)  # TODO: check this is correct!!!
            )
        except Exception:
            self.grid_resolution = tuple()

    def update_intrinsic_reward_counts(self):
        all_update_dicts = flatten_lists(MPI.COMM_WORLD.allgather(list(self.intrinsic_reward_count_updates.items())))
        for k, v in all_update_dicts:
            self.intrinsic_reward_counts[k] = self.intrinsic_reward_counts.get(k, 0) + v
        self.intrinsic_reward_count_updates = {}

    def save_intrinsic_reward_counts(self, update):
        if hvd.rank() == 0:
            previous_cells = []
            import pickle
            filename = self.args.save_path + '/im_cells.pkl'
            try:
                previous_cells = pickle.load(open(filename, 'rb'))
            except Exception:
                pass
            previous_cells.append((update, self.intrinsic_reward_counts))
            pickle.dump([previous_cells], open(filename, 'wb'))

    def get_im_key(self, raw_cell):
        res = {}
        for dimension in self.grid_resolution:
            res[dimension.attr] = dimension.apply(raw_cell)
        im_key = raw_cell.__class__(**res)
        return im_key

    def run(self):
        # shift forward
        if len(self.mb_stuff[2]) >= self.nsteps+self.num_steps_to_cut_left+self.num_steps_to_cut_right:
            self.mb_stuff = [l[self.nsteps:] for l in self.mb_stuff]

        mb_obs, mb_increase_ent, mb_rewards, mb_reward_avg, mb_actions, mb_values, mb_valids, mb_random_resets, \
            mb_dones, mb_neglogpacs, mb_states, mb_sil_actions, mb_sil_rew, mb_sil_valid = self.mb_stuff
        epinfos = []
        while len(mb_rewards) < self.nsteps+self.num_steps_to_cut_left+self.num_steps_to_cut_right:
            # print(len(mb_rewards), "/", self.nsteps+self.num_steps_to_cut_left+self.num_steps_to_cut_right)
            actions, values, states, neglogpacs = self.model.step(mb_obs[-1], mb_states[-1], mb_dones[-1], mb_increase_ent[-1])
            mb_actions.append(actions)
            mb_values.append(values)
            mb_states.append(states)
            mb_neglogpacs.append(neglogpacs)

            obs, rewards, dones, infos = self.env.step(actions)
            # mb_obs.append((obs))
            mb_obs.append(np.cast[self.X_typename](obs))
            mb_increase_ent.append(np.asarray([info.get('increase_entropy', False) for info in infos], dtype=np.uint8))
            if 'fetch' in self.args.game:
                for i in range(len(rewards)):
                    raw_cell = infos[i]['env_cell']
                    im_key = self.get_im_key(raw_cell)
                    if dones[i]:
                        self.prev_cells[i] = None
                        self.recent_intrinsic_rewards.append(self.total_intrinsic_reward[i])
                        self.total_intrinsic_reward[i] = 0.0
                    else:
                        im_count = self.intrinsic_reward_counts.get(im_key, 1)
                        intrinsic_reward = 0
                        if self.args.im_reward_all or im_key != self.prev_cells[i]:
                            intrinsic_reward = self.args.intrinsic_reward_weight / np.sqrt(im_count)
                        self.total_intrinsic_reward[i] += intrinsic_reward
                        rewards[i] += intrinsic_reward
                    if self.args.im_count_all or im_key != self.prev_cells[i]:
                        self.intrinsic_reward_count_updates[im_key] = self.intrinsic_reward_count_updates.get(im_key, 0) + 1
                    self.prev_cells[i] = im_key
                self.update_intrinsic_reward_counts()
            mb_rewards.append(rewards)
            mb_dones.append(dones)
            mb_valids.append([(not info.get('replay_reset.invalid_transition', False)) for info in infos])
            mb_random_resets.append(np.array([info.get('replay_reset.random_reset', False) for info in infos]))

            def get_sil_valid(info):
                is_valid = float(info.get('replay_reset.demo_action') is not None)
                if self.sil_weight_success_rate and is_valid:
                    is_valid *= 1.0 - info['replay_reset.demo_action']['reset_manager.starting_point_success']
                return is_valid

            mb_sil_valid.append([get_sil_valid(info) for info in infos])
            sil_actions = np.zeros_like(actions)
            for cur_info_id, info in enumerate(infos):
                cur_action = info.get('replay_reset.demo_action')
                if cur_action is not None:
                    sil_actions[cur_info_id] = cur_action['action']

            # Note: this is somewhat hacky but basically we assume that if the action is not discrete, then it ranges
            # from -1 to 1 but needs to be output as the logit of tanh by the neural net, so we need to arctanh it here.
            if not np.issubdtype(actions.dtype, np.integer):
                # Note: arctanh fails for values that are exactly 1 or -1. In fact, due to 32 bit floating point
                # precision, the values need to be between -1 + 1e-7 and 1 - 1e-7. Clipping to those values here.
                # Further, to prevent the loss from blowing up we actually clip to between -1 + 1e-6 and 1 - 1e-6
                # instead.
                sil_actions = np.arctanh(np.clip(sil_actions, -1 + 1e-6, 1 - 1e-6))
            mb_sil_actions.append(sil_actions)
            # TODO: this might be a bit inefficient, perhaps should be computed on the other side to limit amount of
            # data transfered
            mb_sil_rew.append([(info.get('replay_reset.demo_action') or {'discounted_rewards': 0.0})['discounted_rewards'] for info in infos])

            for info, orig_ob in zip(infos, mb_obs[-2]):
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
                maybe_sil_epinfo = info.get('sil_episode')
                if maybe_sil_epinfo: self.recent_sil_episodes.append(maybe_sil_epinfo)

        # GAE
        mb_advs = [np.zeros_like(mb_values[0])] * (len(mb_rewards) + 1)
        for t in reversed(range(len(mb_rewards))):
            if t < self.num_steps_to_cut_left:
                mb_valids[t] = np.zeros_like(mb_valids[t])
            else:
                if t == len(mb_values)-1:
                    next_value = self.model.value(mb_obs[-1], mb_states[-1], mb_dones[-1])
                else:
                    next_value = mb_values[t+1]
                use_next = np.logical_not(mb_dones[t+1])
                adv_mask = np.logical_not(mb_random_resets[t+1])
                delta = mb_rewards[t] + self.gamma * use_next * next_value - mb_values[t]
                mb_advs[t] = adv_mask * (delta + self.gamma * self.lam * use_next * mb_advs[t + 1])

        # extract arrays
        end = self.nsteps + self.num_steps_to_cut_left
        ar_mb_obs = np.asarray(mb_obs[:end], dtype=self.X_typename)
        ar_mb_ent = np.stack(mb_increase_ent[:end], axis=0)
        ar_mb_valids = np.asarray(mb_valids[:end], dtype=np.float32)
        ar_mb_actions = np.asarray(mb_actions[:end])
        ar_mb_values = np.asarray(mb_values[:end], dtype=np.float32)
        ar_mb_neglogpacs = np.asarray(mb_neglogpacs[:end], dtype=np.float32)
        ar_mb_dones = np.asarray(mb_dones[:end], dtype=np.bool)
        ar_mb_advs = np.asarray(mb_advs[:end], dtype=np.float32)
        ar_mb_rets = ar_mb_values + ar_mb_advs
        ar_mb_sil_valid = np.asarray(mb_sil_valid[:end], dtype=np.float32)
        # print(ar_mb_sil_valid.tolist())
        ar_mb_sil_actions = np.asarray(mb_sil_actions[:end])
        ar_mb_sil_rew = np.asarray(mb_sil_rew[:end], dtype=np.float32)

        if self.norm_adv:
            adv_mean, adv_std, _ = mpi_moments(ar_mb_advs.ravel())
            ar_mb_advs = (ar_mb_advs - adv_mean) / (adv_std + 1e-7)

        # obs, increase_ent, advantages, masks, actions, values, neglogpacs, valids, returns, states, epinfos = runner.run()
        return (*map(sf01, (ar_mb_obs, ar_mb_ent, ar_mb_advs, ar_mb_dones, ar_mb_actions, ar_mb_values, ar_mb_neglogpacs, ar_mb_valids, ar_mb_rets, ar_mb_sil_actions, ar_mb_sil_rew, ar_mb_sil_valid)),
            mb_states[0], epinfos)

def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])


def learn(env, args, test_mode):
    print('In PPO')
    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    # TODO: this could be adjusted to ignore SIL!!!
    nbatch = nenvs * args.nsteps
    nsteps_train = args.nsteps + args.nsteps // 2

    model = Model(args=args, ob_space=ob_space, ac_space=ac_space, nenv=nenvs, nsteps=nsteps_train, test_mode=test_mode)
    runner = Runner(args=args, env=env, model=model, nsteps=args.nsteps)

    print('Model and Runner built')
    tfirststart = time.time()
    nupdates = args.num_timesteps // (nbatch*hvd.size())
    update = 0
    epinfobuf = deque(maxlen=100)

    # Initialize the per-episode information buffers
    per_ep_info_buffers = None
    if hasattr(env.venv, 'n_demos'):
        per_ep_info_buffers = []
        for idx in range(env.venv.n_demos):
            per_ep_info_buffers.append(deque(maxlen=100))

    while update < nupdates:
        if test_mode:
            def check_done():
                import pickle, glob
                all_episodes = []
                for e in glob.glob(f'{args.save_path}/*.pickle'):
                    all_episodes += pickle.load(open(e, 'rb'))
                all_episodes.sort(key=lambda x: x['start_step'])
                if len(all_episodes) < args.num_per_noop:
                    return False
                n_to_consider = args.num_per_noop
                while n_to_consider < len(all_episodes) - 1 and all_episodes[n_to_consider - 1]['start_step'] == all_episodes[n_to_consider]['start_step']:
                    n_to_consider += 1
                return all([('score' in e) for e in all_episodes[:n_to_consider]])
                # for i in range(0, 31):
                #     json_path = args.save_path + '/' + str(i) + '.pickle'
                #     n_data = 0
                #     try:
                #         n_data = len(pickle.load(open(json_path, 'rb')))
                #     except FileNotFoundError:
                #         pass
                #     if n_data < args.num_per_noop:
                #         return False
                # return True

            if check_done():
                break

        tstart = time.time()
        update += 1

        if update == 1:
            print('Doing first run')

        obs, increase_ent, advantages, masks, actions, values, neglogpacs, valids, returns, sil_actions, sil_rew, sil_valid, states, epinfos = runner.run()

        if update == 1:
            print('Done first run')

        if hvd.size()>1:
            epinfos = flatten_lists(MPI.COMM_WORLD.allgather(epinfos))

        for epinfo in epinfos:
            if 'write_to_pickle' in epinfo:
                score_list = []
                json_path = epinfo['pickle_path']
                import pickle, pickletools
                try:
                    score_list = pickle.load(open(json_path, 'rb'))
                except FileNotFoundError:
                    pass
                score_list.append(epinfo['write_to_pickle'])
                open(json_path, 'wb').write(pickletools.optimize(pickle.dumps(score_list)))

        if update == 1:
            print('Doing training')
        if not test_mode:
            mblossvals = []
            for _ in range(args.noptepochs):
                mblossvals.append(
                    model.train(
                        args.learning_rate, obs, returns, advantages,
                        masks, actions, values, neglogpacs, valids,
                        increase_ent, sil_actions, sil_rew, sil_valid, states
                    )
                )
        if update == 1:
            print('Done training')

        if hvd.rank() == 0:
            tnow = time.time()
            tps = int(nbatch*hvd.size() / (tnow - tstart))
            if update % args.log_interval == 0 or update == 1:
                epinfobuf.extend(epinfos)
                if len(epinfos) >= 100:
                    epinfos_to_report = epinfos
                else:
                    epinfos_to_report = epinfobuf
                ev = explained_variance(values, returns)
                logger.logkv("serial_timesteps", update*args.nsteps)
                logger.logkv("nupdates", update)
                logger.logkv("total_timesteps", update*nbatch*hvd.size())
                logger.logkv("tps", tps)
                logger.logkv("explained_variance", float(ev))
                logger.logkv('eprewmean', safemean([epinfo['r'] for epinfo in epinfos_to_report]))
                logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfos_to_report]))
                logger.logkv('perc_using_cache', safemean([epinfo['using_cache'] for epinfo in epinfos_to_report if epinfo.get('is_extra_sil')]))
                logger.logkv('sil_from_start_episodes_prop', safemean([epinfo['from_start'] for epinfo in runner.recent_sil_episodes]))
                total_sil_steps = sum([epinfo['nsteps'] for epinfo in runner.recent_sil_episodes])
                from_start_sil_steps = sum([epinfo['nsteps'] * epinfo['from_start'] for epinfo in runner.recent_sil_episodes])
                logger.logkv('sil_from_start_steps_prop', from_start_sil_steps / max(total_sil_steps, 1))

                # Update the per-episode information buffers
                per_ep_rep = None
                if hasattr(env.venv, 'n_demos'):
                    per_ep_rep = [[]] * env.venv.n_demos
                    for idx in range(env.venv.n_demos):
                        new_ep_infos = [epinfo for epinfo in epinfos if epinfo['idx'] == idx]
                        per_ep_info_buffers[idx].extend(new_ep_infos)
                        if len(new_ep_infos) >= 100:
                            per_ep_rep[idx] = new_ep_infos
                        else:
                            per_ep_rep[idx] = list(per_ep_info_buffers[idx])

                if not test_mode:
                    if args.intrinsic_reward_weight > 0:
                        logger.logkv('im_avg_episode_reward', np.mean(runner.recent_intrinsic_rewards))
                        logger.logkv('im_n_cells', len(runner.intrinsic_reward_counts))
                        for dimension in runner.grid_resolution:
                            logger.logkv(f'im_{dimension.attr}_n_cells', len(set(getattr(e, dimension.attr) for e in runner.intrinsic_reward_counts)))

                    lossvals = np.mean(mblossvals, axis=0)
                    for (lossval, lossname) in zip(lossvals, model.loss_names):
                        logger.logkv(lossname, lossval)
                    if hasattr(env, 'max_starting_point'):
                        logger.logkv('max_starting_point', env.max_starting_point)
                        logger.logkv('min_starting_point', env.min_starting_point)
                        logger.logkv('as_good_as_demo_start', safemean(
                            [epinfo['as_good_as_demo'] for epinfo in epinfos_to_report if
                             epinfo['starting_point'] <= env.max_starting_point]))
                        logger.logkv('as_good_as_demo_all', safemean(
                            [epinfo['as_good_as_demo'] for epinfo in epinfos_to_report]))
                        logger.logkv('perc_started_below_max_sp', safemean(
                            [epinfo['starting_point'] <= env.max_starting_point for epinfo in epinfos_to_report]))
                    elif hasattr(env.venv, 'n_demos'):
                        expected_steps = np.mean(env.venv.recursive_getattr('expected_steps'), axis=0)
                        times_demos_chosen = np.sum(env.venv.recursive_getattr('times_demos_chosen'), axis=0)
                        assert len(expected_steps) == env.venv.n_demos
                        assert len(times_demos_chosen) == env.venv.n_demos
                        for idx in range(env.venv.n_demos):
                            max_sp = env.venv.demos[idx].max_starting_point
                            min_sp = env.venv.demos[idx].min_starting_point
                            logger.logkv(f'target_{idx}', env.venv.demos[idx].target)
                            logger.logkv(f'expected_steps_{idx}', expected_steps[idx])
                            logger.logkv(f'times_demos_chosen_{idx}', times_demos_chosen[idx])
                            logger.logkv(f'max_starting_point_{idx}', max_sp)
                            logger.logkv(f'min_starting_point_{idx}', min_sp)
                            as_good_s = [e['as_good_as_demo'] for e in per_ep_rep[idx] if e['starting_point'] <= max_sp]
                            logger.logkv(f'as_good_as_demo_start_{idx}', safemean(as_good_s))
                            as_good_a = [e['as_good_as_demo'] for e in per_ep_rep[idx]]
                            logger.logkv(f'as_good_as_demo_all_{idx}', safemean(as_good_a))
                            below_max = [e['starting_point'] <= max_sp for e in per_ep_rep[idx]]
                            logger.logkv(f'perc_below_max_sp_{idx}', safemean(below_max))
                            avg_score = [e['r'] for e in per_ep_rep[idx]]
                            logger.logkv(f'avg_score_{idx}', safemean(avg_score))
                            intrinsic_score = [e['intrinsic_score'] for e in per_ep_rep[idx]]
                            logger.logkv(f'intrinsic_score_{idx}', safemean(intrinsic_score))
                            avg_score_start = [e['r'] for e in per_ep_rep[idx] if e['starting_point'] == 0]
                            logger.logkv(f'avg_score_from_start_{idx}', safemean(avg_score_start))

                done_reason_counter = {}
                for inf in epinfobuf:
                    cur_reason = '-'.join(inf.get('done_reasons', []))
                    done_reason_counter[cur_reason] = done_reason_counter.get(cur_reason, 0) + 1

                for reason in sorted(done_reason_counter):
                    logger.logkv(f'done_{reason}', done_reason_counter[reason] / len(epinfobuf))

                logger.logkv('time_elapsed', tnow - tfirststart)
                logger.logkv('perc_valid', np.mean(valids))
                logger.logkv('tcount', update*nbatch*hvd.size())
                logger.dumpkvs()
            if args.save_interval and (update % args.save_interval == 0 or update == 1) and not test_mode:
                savepath = osp.join(osp.join(args.save_path, args.game), '%.6i' % update)
                print('Saving to', savepath)
                runner.save_intrinsic_reward_counts(update)
                model.save(savepath)

            if osp.exists(args.save_path + '/must_die'):
                sys.exit('Received the kill signal')

        if osp.exists(args.save_path + '/must_die_soft'):
            break

    env.close()

def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
