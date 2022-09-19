from urllib.request import urlretrieve
import matplotlib.pyplot as plt
import nengo
from nengo.utils.filter_design import cont2discrete
import numpy as np
import tensorflow as tf
import nengo_dl
import os

# set seed to ensure this example is reproducible
seed = 0
# tf.random.set_seed(seed)
# np.random.seed(seed)
rng = np.random.RandomState(seed)

n_samp = 2400
n_pts = 500


unroll_factor = 10
n_samp = 2400
n_pts = 500
n_stream = 1
n_step = int(np.ceil(n_pts/n_stream))
n_step = int(np.ceil(n_step/unroll_factor))*unroll_factor
print(n_step)

n_chan = 6
n_label = 12

n_train = int(0.8*n_samp)
n_test = n_samp - n_train

x_data = np.zeros((n_samp,n_pts*n_chan))
y_data = np.zeros((n_samp,1))

# process the raw data
with open("gesture_real_12_2400.txt", "r")  as in_file:
    data_lines = in_file.readlines()

for i in range(n_samp):
    line = data_lines[i]
    items = np.array(line.split(" "))
    items = items.astype(int)
    
    x_data[i,:] = items[:-1]
    y_data[i] = items[-1]

# first random permutation
perm = rng.permutation(x_data.shape[0])
x_data = x_data[perm]
y_data = y_data[perm]

# divide the dataset into training and testing group
x_train = x_data[:n_train]
y_train = y_data[:n_train]
x_test = x_data[n_train:]
y_test = y_data[n_train:]

# generating more training data by timeshifting original samples
# x_train_shift_1 = np.roll(x_train, -50, axis=1)
# x_train_shift_1[:, -50:] = 0
# x_train_shift_2 = np.roll(x_train, 20, axis=1)
# x_train_shift_2[:, :20] = 0
# x_train = np.concatenate((x_train, x_train_shift_1, x_train_shift_2), axis=0)
# y_train = np.tile(y_train, (3, 1))

# Second permutation
# perm = rng.permutation(x_train.shape[0])
# x_train = x_train[perm,:]
# y_train = y_train[perm]

# x_train = x_train.reshape((x_train.shape[0],n_pts,n_chan))
x_train = x_train.reshape((x_train.shape[0],n_chan,n_pts)).swapaxes(1,2)
y_train = y_train.reshape((y_train.shape[0],1,1))
# x_test = x_test.reshape((x_test.shape[0],n_pts,n_chan))
x_test = x_test.reshape((x_test.shape[0],n_chan,n_pts)).swapaxes(1,2)
y_test = y_test.reshape((y_test.shape[0],1,1))

class LMUCell(nengo.Network):
    def __init__(self, hd, md, theta, xd, **kwargs):
        super().__init__(**kwargs)

        # compute the A and B matrices according to the LMU's mathematical derivation
        # (see the paper for details)
        Q = np.arange(md, dtype=np.float64)
        R = (2 * Q + 1)[:, None] / theta
        j, i = np.meshgrid(Q, Q)

        A = np.where(i < j, -1, (-1.0) ** (i - j + 1)) * R
        B = (-1.0) ** Q[:, None] * R
        C = np.ones((1, md))
        D = np.zeros((1,))

        A, B, _, _, _ = cont2discrete((A, B, C, D), dt=1.0, method="zoh")

        with self:
            nengo_dl.configure_settings(trainable=None)

            # create objects corresponding to the x/u/m/h variables in the above diagram
            self.x = nengo.Node(size_in=xd)
            self.u = nengo.Node(size_in=1)
            self.m = nengo.Node(size_in=md)
            self.h = nengo_dl.TensorNode(tf.nn.tanh, shape_in=(hd,), pass_time=False)

            # compute u_t from the above diagram. we have removed e_h and e_m as they
            # are not needed in this task.
            nengo.Connection(
                self.x, self.u, transform=np.ones((1,xd)), synapse=None
            )

            # compute m_t
            # in this implementation we'll make A and B non-trainable, but they
            # could also be optimized in the same way as the other parameters.
            # note that setting synapse=0 (versus synapse=None) adds a one-timestep
            # delay, so we can think of any connections with synapse=0 as representing
            # value_{t-1}.
            conn_A = nengo.Connection(self.m, self.m, transform=A, synapse=0)
            self.config[conn_A].trainable = False
            conn_B = nengo.Connection(self.u, self.m, transform=B, synapse=None)
            self.config[conn_B].trainable = False

            # compute h_t
            nengo.Connection(
                self.x, self.h, transform=nengo_dl.dists.Glorot(), synapse=None
            )
            nengo.Connection(
                self.h, self.h, transform=nengo_dl.dists.Glorot(), synapse=0
            )
            nengo.Connection(
                self.m,
                self.h,
                transform=nengo_dl.dists.Glorot(),
                synapse=None,
            )

with nengo.Network() as net:
    # remove some unnecessary features to speed up the training
    nengo_dl.configure_settings(
        trainable=None,
        stateful=True,
        keep_history=False,
    )

    # input node
    inp = nengo.Node(np.zeros(n_stream*n_chan))

    # lmu cell
    lmu = LMUCell(
        hd=100,
        md=50,
        theta=n_step,
        xd=n_chan*n_stream,
    )
    conn = nengo.Connection(inp, lmu.x, synapse=None)
    net.config[conn].trainable = False

    # dense linear readout
    out = nengo.Node(size_in=n_label)
    nengo.Connection(lmu.h, out, transform=nengo_dl.dists.Glorot(), synapse=None)

    # record output. note that we set keep_history=False above, so this will
    # only record the output on the last timestep (which is all we need
    # on this task)
    p = nengo.Probe(out)

do_training = True
n_epoch = 80
model_name = 'model_woaug_single#1'
if not os.path.exists('./weights/%s'%model_name):
    os.mkdir('./weights/%s'%model_name)

with nengo_dl.Simulator(net, minibatch_size=120, unroll_simulation=10) as sim:
    sim.compile(
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.optimizers.Adam(),
        metrics=["accuracy"],
    )

    x_train = x_train.reshape((n_train,n_step,-1))
    x_test = x_test.reshape((n_test,n_step,-1))

    test_acc = sim.evaluate(x_test, y_test, verbose=0)["probe_accuracy"]
    print(f"Initial test accuracy: {test_acc * 100:.2f}%")

    if do_training:
        for i_epoch in range(n_epoch):
            print("Epoch #%03d:"%i_epoch)
            sim.fit(x_train, y_train, epochs=1, stateful=True)
            sim.save_params("./weights/%s/params_epoch#%03d"%(model_name,i_epoch))
            test_acc = sim.evaluate(x_test, y_test, verbose=0)["probe_accuracy"]
            print(f"Test accuracy: {test_acc * 100:.2f}%")
        sim.save_params("./weights/%s/params_final"%model_name)
    else:
        sim.load_params("./weights/%s/params_final"%model_name)

    test_acc = sim.evaluate(x_test, y_test, verbose=0)["probe_accuracy"]
    print(f"Final test accuracy: {test_acc * 100:.2f}%")
