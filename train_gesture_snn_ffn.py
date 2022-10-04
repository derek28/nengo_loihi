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
tf.random.set_seed(seed)
np.random.seed(seed)

dt = 0.003
unroll_factor = 10
n_samp = 2400
n_pts = 500

n_chan = 6
n_label = 12
n_delay = 10
tau = n_delay*dt

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
x_data = x_data.reshape((n_samp,n_chan,n_pts)).swapaxes(1,2)
y_data = y_data.reshape((n_samp,1,1))

# first random permutation
perm = np.random.permutation(x_data.shape[0])
x_data = x_data[perm]
y_data = y_data[perm]

# divide the dataset into training and testing group
x_train = x_data[:n_train]
y_train = y_data[:n_train]
x_test = x_data[n_train:]
y_test = y_data[n_train:]

# data augmentation
aug_ratio = 4
smax = 50
x_train_aug = np.zeros((n_train*aug_ratio,n_pts,n_chan))
for i in range(n_train):
    ls = -(n_pts-1)
    rs = n_pts-1
    for j in range(n_chan):
        pidx = np.where(x_train[i,:,j]>0)[0]
        if pidx.shape[0] > 0:
            if ls < -pidx[0]:
                ls = -pidx[0]
            if rs > n_pts-1-pidx[-1]:
                rs = n_pts-1-pidx[-1]
    if ls < -smax:
        ls = -smax
    if rs > smax:
        rs = smax
    sran = np.concatenate((np.array(range(ls,0)), np.array(range(1,rs+1))), axis=0)
    sval = np.random.choice(sran, aug_ratio, replace=False)
    for j in range(aug_ratio):
        x_train_aug[(j-1)*n_train+i:(j-1)*n_train+i+1,:,:] = np.roll(x_train[i:i+1,:,:],sval[j],axis=1)

x_train = np.concatenate((x_train, x_train_aug), axis=0)
y_train = np.tile(y_train, (aug_ratio+1,1,1))
n_train = x_train.shape[0]

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
            self.x = [nengo.Node(np.zeros(1), label='Node_x%d'%i) for i in range(xd)]
            self.u = [nengo.Ensemble(1, 1, neuron_type=nengo.Tanh(tau_ref=1), gain=np.ones(1), bias=np.zeros(1), label='Ensemble_u%d'%i).neurons for i in range(xd)]
            self.m = [nengo.Ensemble(md, 1, neuron_type=nengo.Tanh(tau_ref=1), gain=np.ones(md), bias=np.zeros(md), label='Ensemble_m%d'%i).neurons for i in range(xd)]
            self.h = nengo.Ensemble(hd, 1, neuron_type=nengo.Tanh(tau_ref=1), gain=np.ones(hd), bias=np.zeros(hd), label='Node_h').neurons

            # compute u_t from the above diagram. we have removed e_h and e_m as they
            # are not needed in this task.
            conn_inp = [nengo.Connection(self.x[i], self.u[i], transform=np.ones((1,1)), synapse=tau) for i in range(xd)]

            # compute m_t
            # in this implementation we'll make A and B non-trainable, but they
            # could also be optimized in the same way as the other parameters.
            # note that setting synapse=0 (versus synapse=None) adds a one-timestep
            # delay, so we can think of any connections with synapse=0 as representing
            # value_{t-1}.
            conn_A = [nengo.Connection(self.m[i], self.m[i], transform=A, synapse=0) for i in range(xd)]
            for i in range(xd):
                self.config[conn_A[i]].trainable = False
            conn_B = [nengo.Connection(self.u[i], self.m[i], transform=B, synapse=None) for i in range(xd)]
            for i in range(xd):
                self.config[conn_B[i]].trainable = False

            for i in range(xd):
                nengo.Connection(
                    self.x[i], self.h, transform=nengo.Dense((hd,1), init=nengo_dl.dists.Glorot()), synapse=tau
                )
            nengo.Connection(
                self.h, self.h, transform=nengo.Dense((hd,hd), init=nengo_dl.dists.Glorot()), synapse=0
            )
            for i in range(xd):
                nengo.Connection(self.m[i], self.h, transform=nengo.Dense((hd,md), init=nengo_dl.dists.Glorot()), synapse=None)

max_rate = 250
amp = 1.0/max_rate
rate_reg = 1e-3
rate_target = max_rate*amp
relu = nengo.SpikingRectifiedLinear(amplitude=amp)
tau = 0.1

with nengo.Network() as net:
    net.config[nengo.Ensemble].max_rates = nengo.dists.Choice([max_rate])
    net.config[nengo.Ensemble].intercepts = nengo.dists.Choice([0])
    net.config[nengo.Connection].synapse = None

    nengo_dl.configure_settings(
        # learning_phase=True,
        trainable=None,
        stateful=True,
        keep_history=False,
    )

    inp = nengo.Node(np.zeros(n_chan), label='Input')
    fc1 = nengo.Ensemble(100, 1, neuron_type=nengo.RectifiedLinear(amplitude=amp), label='FC1').neurons
    fc2 = nengo.Ensemble(100, 1, neuron_type=nengo.RectifiedLinear(amplitude=amp), label='FC2').neurons
    fc3 = nengo.Ensemble(n_label, 1, neuron_type=nengo.RectifiedLinear(amplitude=amp), label='FC3').neurons
    out = nengo.Node(size_in=n_label, label='Output')

    conn1 = nengo.Connection(inp, fc1, transform=nengo_dl.dists.Glorot(), synapse=tau)
    conn2 = nengo.Connection(fc1, fc2, transform=nengo_dl.dists.Glorot(), synapse=tau)
    conn3 = nengo.Connection(fc2, fc3, transform=nengo_dl.dists.Glorot(), synapse=tau)
    conn4 = nengo.Connection(fc3, out, transform=nengo_dl.dists.Glorot(), synapse=tau)

    p = nengo.Probe(out)

do_training = True
n_epoch = 80
model_name = 'model_wiaug#4_simple_tau#0.1'
if not os.path.exists('./weights/%s'%model_name):
    os.mkdir('./weights/%s'%model_name)

with nengo_dl.Simulator(net, dt=dt, minibatch_size=120, unroll_simulation=10) as sim:
    sim.compile(
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.optimizers.Adam(),
        metrics=["accuracy"],
    )
    sim.keras_model.summary()

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
        sim.load_params("./weights/%s/params_epoch#079"%model_name)

    test_acc = sim.evaluate(x_test, y_test, verbose=0)["probe_accuracy"]
    print(f"Final test accuracy: {test_acc * 100:.2f}%")
    # test_result = sim.predict(x_test_dict)
    # print(test_result[p][0])
