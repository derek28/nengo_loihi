from urllib.request import urlretrieve
import matplotlib.pyplot as plt
import nengo
from nengo.utils.filter_design import cont2discrete
import numpy as np
import tensorflow as tf
import nengo_dl
import os
from nengo.utils.matplotlib import rasterplot

class Linear(nengo.neurons.NeuronType):
    def step(self, dt, J, output):
        """Implement the tanh nonlinearity."""
        output[...] = J

# seed = 0
# tf.random.set_seed(seed)
# np.random.seed(seed)

# n_samp = 2400
# unroll_factor = 10
# n_samp = 2400
# n_pts = 1500
# n_stream = 5
# n_step = int(np.ceil(n_pts/n_stream))
# n_step = int(np.ceil(n_step/unroll_factor))*unroll_factor

# n_chan = 6
# n_label = 12

# n_train = int(0.8*n_samp)
# n_test = n_samp - n_train

# x_data = np.zeros((n_samp,n_pts*n_chan))
# y_data = np.zeros((n_samp,1))

# with open("gesture_2400_1ms.txt", "r")  as in_file:
#     data_lines = in_file.readlines()

# for i in range(n_samp):
#     line = data_lines[i]
#     items = np.array(line.split(" "))
#     items = items.astype(int)
    
#     x_data[i,:] = items[:-1]
#     y_data[i] = items[-1]
# x_data = x_data.reshape((n_samp,n_chan,n_pts)).swapaxes(1,2)
# y_data = y_data.reshape((n_samp,1,1))

# # first random permutation
# perm = np.random.permutation(x_data.shape[0])
# x_data = x_data[perm]
# y_data = y_data[perm]

# # divide the dataset into training and testing group
# x_train = x_data[:n_train]
# y_train = y_data[:n_train]
# x_test = x_data[n_train:]
# y_test = y_data[n_train:]
# data = x_test[:1,:,:1]

dt=0.001
max_rate = 250
n_lim = 1
x = np.arange(-n_lim,n_lim,dt).reshape((1,-1,1))
# y = np.flip(np.arange(-n_lim,n_lim,dt),axis=0).reshape((1,-1,1))
# data = np.concatenate((x,y),axis=2)
# x = np.random.randint(0,2,(1,int(2*n_lim/dt),1))
# x = np.zeros((1,2000,1))
# x[0,1000,0] = 1
data = x
T = 2*n_lim
d1 = 1
d2 = 10
d3 = 1
tau = 0.1
with nengo.Network() as net:

    inp = nengo.Node(np.zeros(d1))
    # inp = nengo.Node(lambda t: t - n_lim)

    # activation = nengo_dl.TensorNode(tf.nn.tanh, shape_in=(1,), pass_time=False)
    # activation = nengo.Ensemble(1, 1, neuron_type=nengo.Tanh(tau_ref=1.0), gain=np.ones(1), bias=np.zeros(1)).neurons
    # activation = nengo.Ensemble(d2, 1, neuron_type=nengo.RegularSpiking(nengo.Tanh(tau_ref=1.0/max_rate),amplitude=1.0/max_rate), gain=np.ones(d2), bias=np.zeros(d2)).neurons
    # activation = nengo.Ensemble(d2, 1, neuron_type=nengo.SpikingRectifiedLinear(amplitude=1/max_rate), gain=max_rate*np.ones(d2), bias=np.zeros(d2))
    activation = nengo.Ensemble(d2, d1, neuron_type=nengo.SpikingRectifiedLinear(amplitude=1/max_rate), gain=max_rate*np.ones(d2), bias=np.zeros(d2))
    # activation = nengo.Ensemble(d2, d1, max_rates=nengo.dists.Choice([max_rate]), neuron_type=nengo.SpikingRectifiedLinear(), gain=np.ones(d2), bias=np.zeros(d2))
    # activation = nengo.Ensemble(d2, d1, max_rates = nengo.dists.Choice([max_rate]), neuron_type=nengo.LIF(tau_rc=0.02, tau_ref=0.002, amplitude=1))
    # activation = nengo.Ensemble(d2, 1, neuron_type=nengo.RegularSpiking(nengo.Tanh(tau_ref=1/max_rate),amplitude=1/max_rate), gain=np.ones(d2), bias=np.zeros(d2)).neurons
    # activation = nengo.Ensemble(1, 1, neuron_type=nengo.Tanh(tau_ref=1.0/max_rate), gain=np.ones(1), bias=np.zeros(1))
    # activation = nengo.Ensemble(d2, 1, max_rates=nengo.dists.Choice([max_rate]), neuron_type=nengo.SpikingRectifiedLinear(), gain=np.ones(d2), bias=np.zeros(d2))
    # activation = nengo.Ensemble(1, 1, max_rates=nengo.dists.Choice([max_rate]), neuron_type=nengo.RectifiedLinear(), gain=np.ones(1), bias=np.zeros(1)).neurons
    out = nengo.Node(size_in=d3)

    # transform = nengo.Dense((d2,d1), init=nengo_dl.dists.Glorot())
    # conn_it = nengo.Connection(inp, activation, transform=np.ones((1,d2)), synapse=None) 
    # conn_it = nengo.Connection(inp, activation.neurons, transform=transform, synapse=None) 
    # conn_it = nengo.Connection(inp, activation, transform=np.ones((d2,d1)), synapse=None) 
    # conn_it = nengo.Connection(inp, activation, transform=np.ones((5,5)), synapse=None)
    # conn_to = nengo.Connection(activation, out, transform=np.ones((d3,d2)), synapse=None)
    conn_it = nengo.Connection(inp, activation, function=lambda x: x, synapse=tau)
    # conn_it = nengo.Connection(inp, activation, transform=np.eye(1), synapse=0.1)
    conn_to = nengo.Connection(activation, out, transform=np.eye(1), synapse=tau)
    # conn_it = nengo.Connection(inp, out, synapse = 0.1)
    
    p_in = nengo.Probe(inp, synapse=None)
    p_act = nengo.Probe(activation.neurons)
    p_out = nengo.Probe(out, synapse=None)

    nengo_dl.configure_settings(
        trainable=None,
        stateful=True,
        keep_history=True,
        learning_phase=False
    )

with nengo_dl.Simulator(net, dt=dt) as sim:   
    sim.keras_model.summary()
    sim.run(T, data={inp:data})
    # sim.run(n_lim*2)
    # x = np.arange(-n_lim,n_lim,dt)
    # fr = nengo.Tanh(tau_ref=1).rates(x=x, gain=[1], bias=[0])

# print(sim.data[p_act].shape)
plt.figure()
plt.plot(sim.data[p_in])
plt.figure()
plt.plot(sim.data[p_out])
print(sim.trange().shape)
print(sim.data[p_act].shape)
plt.figure()
rasterplot(sim.trange(), sim.data[p_act])
plt.show()