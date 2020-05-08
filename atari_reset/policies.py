import numpy as np
import tensorflow as tf
from baselines.common.distributions import make_pdtype

def to2d(x):
    size = 1
    for shapel in x.get_shape()[1:]: size *= shapel.value
    return tf.reshape(x, (-1, size))

def normc_init(std=1.0, axis=0):
    """
    Initialize with normalized columns
    """
    def _initializer(shape, dtype=None, partition_info=None):  # pylint: disable=W0613
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=axis, keepdims=True))
        return tf.constant(out)
    return _initializer

def ortho_init(scale=1.0):
    def _ortho_init(shape, dtype, partition_info=None): #pylint: disable=W0613
        #lasagne ortho init for tf
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4: # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v # pick the one with the correct shape
        q = q.reshape(shape)
        return (scale * q[:shape[0], :shape[1]]).astype(np.float32)
    return _ortho_init

def fc(x, scope, nout, init_scale=1.0, init_bias=0.0):
    with tf.variable_scope(scope): #pylint: disable=E1129
        nin = x.get_shape()[1].value
        w = tf.get_variable("w", [nin, nout], initializer=normc_init(init_scale))
        b = tf.get_variable("b", [nout], initializer=tf.constant_initializer(init_bias))
        return tf.matmul(x, w) + b

def conv(x, scope, noutchannels, filtsize, stride, pad='VALID', init_scale=1.0, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        nin = x.get_shape()[3].value
        w = tf.get_variable("w", [filtsize, filtsize, nin, noutchannels], initializer=ortho_init(init_scale))
        b = tf.get_variable("b", [noutchannels], initializer=tf.constant_initializer(0.0))
        z = tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding=pad, name="conv")+b
        return z

class GRUCell(tf.nn.rnn_cell.RNNCell):
    """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078)."""
    def __init__(self, num_units, name, nin, rec_gate_init=0.):
        tf.nn.rnn_cell.RNNCell.__init__(self)
        self._num_units = num_units
        self.rec_gate_init = rec_gate_init
        self.w1 = tf.get_variable(name + "w1", [nin+num_units, 2*num_units], initializer=normc_init(1.))
        self.b1 = tf.get_variable(name + "b1", [2*num_units], initializer=tf.constant_initializer(rec_gate_init))
        self.w2 = tf.get_variable(name + "w2", [nin+num_units, num_units], initializer=normc_init(1.))
        self.b2 = tf.get_variable(name + "b2", [num_units], initializer=tf.constant_initializer(0.))

    @property
    def state_size(self):
        return self._num_units
    @property
    def output_size(self):
        return self._num_units
    def call(self, inputs, state):
        """Gated recurrent unit (GRU) with nunits cells."""
        x, new = inputs
        while len(state.get_shape().as_list()) > len(new.get_shape().as_list()):
            new = tf.expand_dims(new,len(new.get_shape().as_list()))
        h = state * (1.0 - new)
        hx = tf.concat([h, x], axis=1)
        mr = tf.sigmoid(tf.matmul(hx, self.w1) + self.b1)
        # r: read strength. m: 'member strength
        m, r = tf.split(mr, 2, axis=1)
        rh_x = tf.concat([r * h, x], axis=1)
        htil = tf.tanh(tf.matmul(rh_x, self.w2) + self.b2)
        h = m * h + (1.0 - m) * htil
        return h, h

class CnnPolicy(object):

    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, test_mode=False, reuse=False):
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc)
        nact = ac_space.n
        X = tf.placeholder(tf.uint8, ob_shape) #obs
        with tf.variable_scope("model", reuse=reuse):
            h = tf.nn.relu(conv(tf.cast(X, tf.float32)/255., 'c1', noutchannels=64, filtsize=8, stride=4))
            h2 = tf.nn.relu(conv(h, 'c2', noutchannels=128, filtsize=4, stride=2))
            h3 = tf.nn.relu(conv(h2, 'c3', noutchannels=128, filtsize=3, stride=1))
            h3 = to2d(h3)
            h4 = tf.nn.relu(fc(h3, 'fc1', nout=1024))
            pi = fc(h4, 'pi', nact, init_scale=0.01)
            vf = fc(h4, 'v', 1, init_scale=0.01)[:,0]

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pi)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X:ob})
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X:ob})

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value

class GRUPolicy(object):

    def __init__(self, sess, ob_space, ac_space, nenv, nsteps, memsize=800, test_mode=False, reuse=False):
        nh, nw, nc = ob_space.shape
        nbatch = nenv*nsteps
        ob_shape = (nbatch, nh, nw, nc)
        nact = ac_space.n

        # use variables instead of placeholder to keep data on GPU if we're training
        X = tf.placeholder(tf.uint8, ob_shape)  # obs
        M = tf.placeholder(tf.float32, [nbatch])  # mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, memsize])  # states
        E = tf.placeholder(tf.uint8, [nbatch])

        with tf.variable_scope("model", reuse=reuse):
            h = tf.nn.relu(conv(tf.cast(X, tf.float32)/255., 'c1', noutchannels=64, filtsize=8, stride=4))
            h2 = tf.nn.relu(conv(h, 'c2', noutchannels=128, filtsize=4, stride=2))
            h3 = tf.nn.relu(conv(h2, 'c3', noutchannels=128, filtsize=3, stride=1))
            h3 = to2d(h3)
            h4 = tf.contrib.layers.layer_norm(fc(h3, 'fc1', nout=memsize), center=False, scale=False, activation_fn=tf.nn.relu)
            h5 = tf.reshape(h4, [nenv, nsteps, memsize])

            m = tf.reshape(M, [nenv, nsteps, 1])
            cell = GRUCell(memsize, 'gru1', nin=memsize)
            h6, snew = tf.nn.dynamic_rnn(cell, (h5, m), dtype=tf.float32, time_major=False, initial_state=S, swap_memory=True)

            h7 = tf.concat([tf.reshape(h6, [nbatch, memsize]), h4], axis=1)
            pi = fc(h7, 'pi', nact, init_scale=0.01)
            if test_mode:
                pi *= 2.
            else:
                pi = tf.where(E>0, pi/2., pi)
            vf = tf.squeeze(fc(h7, 'v', 1, init_scale=0.01))

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pi)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = np.zeros((nenv, memsize), dtype=np.float32)

        def step(ob, state, mask, increase_ent):
            return sess.run([a0, vf, snew, neglogp0], {X:ob, S:state, M:mask, E:increase_ent})

        def value(ob, state, mask):
            return sess.run(vf, {X:ob, S:state, M:mask})

        self.X = X
        self.M = M
        self.S = S
        self.E = E
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value


FFSHAPE = '1x1024'
GRU_POLICY = True
SD_MULTIPLY_EXPLORE = 2
MEMSIZE = 800

class FFPolicy(object):

    def __init__(self, sess, ob_space, ac_space, nenv, nsteps, memsize=800, test_mode=False, reuse=False):
        memsize = MEMSIZE
        if GRU_POLICY:
            # nh, nw, nc = ob_space.shape
            n_obs, = ob_space.shape
            nbatch = nenv*nsteps
            ob_shape = (nbatch, n_obs)
            assert len(ac_space.shape) == 1
            nact = ac_space.shape[0]
            ffdepth, ffwidth = [int(e) for e in FFSHAPE.split('x')]

            # use variables instead of placeholder to keep data on GPU if we're training
            X = tf.placeholder(tf.float32, ob_shape, name='X')  # obs
            M = tf.placeholder(tf.float32, [nbatch], name='Mask')  # mask (done t-1)
            S = tf.placeholder(tf.float32, [nenv, memsize * 2], name='States')  # states
            E = tf.placeholder(tf.uint8, [nbatch], name='EntBoost')

            with tf.variable_scope("model", reuse=reuse):
                last_layers = {}
                snews = []
                for cur_type, prefix in enumerate(['pi', 'v']):
                    hidden = X
                    for i in range(ffdepth):
                        hidden = tf.nn.relu(fc(hidden, f'{prefix}_fc{i}', nout=ffwidth))
                    h4 = tf.contrib.layers.layer_norm(fc(hidden, f'{prefix}_fcend', nout=memsize), center=False, scale=False, activation_fn=tf.nn.relu)
                    h5 = tf.reshape(h4, [nenv, nsteps, memsize], name=f'{prefix}_rnnsreshape')

                    m = tf.reshape(M, [nenv, nsteps, 1], name=f'{prefix}_reshapemask')
                    cell = GRUCell(memsize, f'{prefix}_gru1', nin=memsize)
                    cur_S = tf.split(S, 2, axis=1, name=f'{prefix}_split')[cur_type]
                    h6, snew_cur = tf.nn.dynamic_rnn(cell, (h5, m), dtype=tf.float32, time_major=False, initial_state=cur_S, swap_memory=True)
                    snews.append(snew_cur)

                    h7 = tf.concat([tf.reshape(h6, [nbatch, memsize], name=f'{prefix}_reshapefinal'), h4], axis=1, name=f'{prefix}_concat_h4')
                    last_layers[prefix] = h7

                snew = tf.concat(snews, axis=1)
                pi_mean = fc(last_layers['pi'], 'pi_mean', nact, init_scale=0.01)
                pi_sd = fc(last_layers['pi'], 'pi_sd', nact, init_scale=0.01)
                if test_mode:
                    pass
                    # pi_sd /= 2.
                else:
                    pi_sd = tf.where(E>0, pi_sd*SD_MULTIPLY_EXPLORE, pi_sd)
                pi = tf.concat([pi_mean, pi_sd], axis=len(pi_sd.shape)-1, name='pi')
                vf = tf.squeeze(fc(last_layers['v'], 'v', 1, init_scale=0.01))

            self.pdtype = make_pdtype(ac_space)
            self.pd = self.pdtype.pdfromflat(pi)
            import horovod.tensorflow as hvd
            # if hvd.rank() == 0:
            #     import ipdb;
            #     ipdb.set_trace()

            a0 = self.pd.sample()
            neglogp0 = self.pd.neglogp(a0)
            self.initial_state = np.zeros((nenv, memsize * 2), dtype=np.float32)

            def step(ob, state, mask, increase_ent):
                # with tf.Session() as sess2:
                #     writer = tf.summary.FileWriter("output", sess2.graph)
                #     print(sess2.run([a0, vf, snew, neglogp0, pi_mean, pi_sd], {X:ob, S:state, M:mask, E:increase_ent}))
                #     writer.close()
                import random
                import horovod.tensorflow as hvd
                maybe_write = random.random() < -1
                if hvd.rank() == 0 and maybe_write and False:
                    writer = tf.summary.FileWriter("output", sess.graph)
                a0_, vf_, snew_, neglogp0_, mean_, sd_ = sess.run([a0, vf, snew, neglogp0, pi_mean, pi_sd], {X:ob, S:state, M:mask, E:increase_ent})
                if hvd.rank() == 0 and maybe_write and False:
                    writer.close()
                if maybe_write:
                    print('mean', list(mean_[0]))
                    print('sd', list(np.exp(sd_)[0]))
                    print('action', list(np.tanh(a0_)[0]))
                    print('prob', np.exp(-neglogp0_)[0])
                    print('\n\n')
                return a0_, vf_, snew_, neglogp0_

            def value(ob, state, mask):
                return sess.run(vf, {X:ob, S:state, M:mask})

            self.X = X
            self.M = M
            self.S = S
            self.E = E
            self.pi = pi
            self.vf = vf
            self.step = step
            self.value = value

        else:
            ffdepth, ffwidth = [int(e) for e in FFSHAPE.split('x')]

            n_obs, = ob_space.shape
            nbatch = nenv * nsteps
            ob_shape = (nbatch, n_obs)
            assert len(ac_space.shape) == 1
            nact = ac_space.shape[0]
            X = tf.placeholder(tf.float32, ob_shape) #obs
            with tf.variable_scope("model", reuse=reuse):
                # h = tf.nn.relu(conv(tf.cast(X, tf.float32)/255., 'c1', noutchannels=64, filtsize=8, stride=4))
                # h2 = tf.nn.relu(conv(h, 'c2', noutchannels=128, filtsize=4, stride=2))
                # h3 = tf.nn.relu(conv(h2, 'c3', noutchannels=128, filtsize=3, stride=1))
                # h3 = to2d(h3)
                hidden = X
                for i in range(ffdepth):
                    hidden = tf.nn.relu(fc(hidden, f'fc{i}', nout=ffwidth))
                # Note: we multiply the number of outputs by 2 because we are outputing means and stds
                pi_mean = fc(hidden, 'pi_mean', nact, init_scale=0.01)
                # Note: we init the bias to log(2) so that the starting SD is 2 instead of 1
                pi_sd = fc(hidden, 'pi_sd', nact, init_scale=0.01, init_bias=0.6931471805599453)
                pi = tf.concat([pi_mean, pi_sd], axis=len(pi_sd.shape)-1, name='pi')
                vf = fc(hidden, 'v', 1, init_scale=0.01)[:,0]

            self.pdtype = make_pdtype(ac_space)
            self.pd = self.pdtype.pdfromflat(pi)

            a0 = self.pd.sample()
            neglogp0 = self.pd.neglogp(a0)
            self.initial_state = None

            def step(ob, *_args, **_kwargs):
                a, v, neglogp, mean, sd = sess.run([a0, vf, neglogp0, pi_mean, pi_sd], {X:ob})
                import random
                if random.random() < 0.01:
                    print('mean', list(mean[0]))
                    print('sd', list(np.exp(sd)[0]))
                    print('action', list(np.tanh(a)[0]))
                    print('prob', np.exp(-neglogp)[0])
                    print('\n\n')
                return np.tanh(a), v, self.initial_state, neglogp

            def value(ob, *_args, **_kwargs):
                return sess.run(vf, {X:ob})

            self.X = X
            # TODO: figure out what E is.
            self.E = tf.placeholder(tf.uint8, [nbatch])  # ignored
            self.pi = pi
            self.vf = vf
            self.step = step
            self.value = value


class FetchCNNPolicy(object):

    def add_X(self, state, ob_orig):
        Xs = self.Xs
        ob = ob_orig
        if self.proprioception_n > 0:
            state[self.Xs[-1]] = ob_orig[:, -self.proprioception_n:]
            ob = ob_orig[:, :-self.proprioception_n]
            ob = ob.reshape((len(ob), self.nh, self.nw, self.nc))
        for i, X_ in zip(range(0, self.nc, 3), Xs):
            state[X_] = ob[:, :, :, i:i + 3]
        return state

    def __init__(self, sess, ob_space, ac_space, nenv, nsteps, memsize=800, test_mode=False, reuse=False):
        memsize = MEMSIZE
        proprioception_n = 0
        if len(ob_space.shape) == 3:
            nh, nw, nc = ob_space.shape
        else:
            proprioception_n = 156
            nh, nw, nc = ob_space.pixel_space.shape
        self.proprioception_n = proprioception_n

        self.nc = nc
        self.nh = nh
        self.nw = nw
        nbatch = nenv*nsteps
        ob_shape = (nbatch, nh, nw, 3)
        nact = ac_space.shape[0]

        # use variables instead of placeholder to keep data on GPU if we're training
        Xs = []
        self.Xs = Xs
        for _ in range(0, nc, 3):
            Xs.append(tf.placeholder(tf.float32, ob_shape))  # obs
        if proprioception_n:
            Xs.append(tf.placeholder(tf.float32, (nbatch, proprioception_n)))
        M = tf.placeholder(tf.float32, [nbatch])  # mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, memsize * 2])  # states
        E = tf.placeholder(tf.uint8, [nbatch])

        with tf.variable_scope("model", reuse=reuse):
            last_layers = {}
            snews = []
            for cur_type, prefix in enumerate(['pi', 'v']):
                h3s = []
                reuse_conv = False
                for X_ in (Xs[:-1] if proprioception_n else Xs):
                    h = tf.nn.relu(conv(tf.cast(X_, tf.float32) / 255., f'{prefix}_c1', noutchannels=64, filtsize=8, stride=4, reuse=reuse_conv))
                    h2 = tf.nn.relu(conv(h, f'{prefix}_c2', noutchannels=128, filtsize=4, stride=2, reuse=reuse_conv))
                    h3_ = tf.nn.relu(conv(h2, f'{prefix}_c3', noutchannels=128, filtsize=3, stride=1, reuse=reuse_conv))
                    h3s.append(to2d(h3_))
                    reuse_conv = True
                if proprioception_n:
                    h3s.append(Xs[-1])
                h3 = tf.concat(h3s, axis=-1)
                h4 = tf.contrib.layers.layer_norm(fc(h3, f'{prefix}_fc1', nout=memsize), center=False, scale=False,
                                                  activation_fn=tf.nn.relu)
                h5 = tf.reshape(h4, [nenv, nsteps, memsize])

                m = tf.reshape(M, [nenv, nsteps, 1])
                cell = GRUCell(memsize, f'{prefix}_gru1', nin=memsize)
                cur_S = tf.split(S, 2, axis=1, name=f'{prefix}_split')[cur_type]
                h6, snew_cur = tf.nn.dynamic_rnn(cell, (h5, m), dtype=tf.float32, time_major=False, initial_state=cur_S,
                                             swap_memory=True)
                snews.append(snew_cur)

                h7 = tf.concat([tf.reshape(h6, [nbatch, memsize], name=f'{prefix}_reshapefinal'), h4], axis=1, name=f'{prefix}_concat_h4')
                last_layers[prefix] = h7

            snew = tf.concat(snews, axis=1)
            pi_mean = fc(last_layers['pi'], 'pi_mean', nact, init_scale=0.01)
            pi_sd = fc(last_layers['pi'], 'pi_sd', nact, init_scale=0.01)
            if test_mode:
                pass
                # pi_sd /= 2.
            else:
                pi_sd = tf.where(E>0, pi_sd*SD_MULTIPLY_EXPLORE, pi_sd)
            pi = tf.concat([pi_mean, pi_sd], axis=len(pi_sd.shape)-1, name='pi')
            vf = tf.squeeze(fc(last_layers['v'], 'v', 1, init_scale=0.01))

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pi)
        import horovod.tensorflow as hvd
        # if hvd.rank() == 0:
        #     import ipdb;
        #     ipdb.set_trace()

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = np.zeros((nenv, memsize * 2), dtype=np.float32)

        def step(ob, state, mask, increase_ent):
            # with tf.Session() as sess2:
            #     writer = tf.summary.FileWriter("output", sess2.graph)
            #     print(sess2.run([a0, vf, snew, neglogp0, pi_mean, pi_sd], {X:ob, S:state, M:mask, E:increase_ent}))
            #     writer.close()
            import random
            import horovod.tensorflow as hvd
            maybe_write = random.random() < -1
            if hvd.rank() == 0 and maybe_write and False:
                writer = tf.summary.FileWriter("output", sess.graph)
            a0_, vf_, snew_, neglogp0_, mean_, sd_ = sess.run([a0, vf, snew, neglogp0, pi_mean, pi_sd], self.add_X({S:state, M:mask, E:increase_ent}, ob))
            if hvd.rank() == 0 and maybe_write and False:
                writer.close()
            if maybe_write:
                print('mean', list(mean_[0]))
                print('sd', list(np.exp(sd_)[0]))
                print('action', list(np.tanh(a0_)[0]))
                print('prob', np.exp(-neglogp0_)[0])
                print('\n\n')
            return a0_, vf_, snew_, neglogp0_

        def value(ob, state, mask):
            return sess.run(vf, self.add_X({S: state, M: mask}, ob))

        # self.X = X
        self.M = M
        self.S = S
        self.E = E
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value
