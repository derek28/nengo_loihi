from urllib.request import urlretrieve
import matplotlib.pyplot as plt
import nengo
from nengo.utils.filter_design import cont2discrete
import numpy as np
import tensorflow as tf
import nengo_dl
import os

n_step = 2001
n_lim = 10
data = np.linspace(-n_lim,n_lim,n_step).reshape((1,-1,1))
with nengo.Network() as net:

    inp = nengo.Node(np.zeros(1))

    # tanh_tensor = nengo_dl.TensorNode(tf.nn.tanh, shape_in=(1,), pass_time=False)
    tanh_tensor = nengo.Ensemble(1, 1, neuron_type=nengo.Tanh(tau_ref=1), gain=np.ones(1), bias=np.zeros(1)).neurons

    out = nengo.Node(size_in=1)

    conn_it = nengo.Connection(inp, tanh_tensor, transform=1, synapse=None)
    conn_to = nengo.Connection(tanh_tensor, out, transform=1, synapse=None)
    
    p_in = nengo.Probe(inp)
    p_out = nengo.Probe(out)

with nengo_dl.Simulator(net) as sim:
    sim.run_steps(n_step, {inp:data})

plt.figure()
plt.plot(data[0], sim.data[p_in])
plt.plot(data[0], sim.data[p_out])
plt.show()