from urllib.request import urlretrieve
import matplotlib.pyplot as plt
import nengo
from nengo.utils.filter_design import cont2discrete
import numpy as np
import tensorflow as tf
import nengo_dl
import os

n_chan = 6
with nengo.Network() as net:
    inp1 = nengo.Node(np.zeros(n_chan))
    inp2 = nengo.Node(np.zeros(n_chan))
    out = nengo.Node(size_in=n_chan)

    conn1 = nengo.Connection(inp1, out, transform=np.eye(n_chan), synapse=None)
    conn2 = nengo.Connection(inp2, out, transform=np.eye(n_chan)*2, synapse=None)
    
    p_in = nengo.Probe(inp1)
    p_out = nengo.Probe(out)

n_step = 100
data1 = np.zeros((1,n_step,n_chan))
data2 = np.zeros((1,n_step,n_chan))
for i in range(n_chan):
    data1[:,:,i] = i+1
    data2[:,:,i] = 10*(i+1)

with nengo_dl.Simulator(net, minibatch_size=1, unroll_simulation=10) as sim:
    sim.run_steps(100, data={inp1:data1, inp2:data2})

plt.figure()
plt.plot(sim.data[p_in][0,:,0])
plt.plot(sim.data[p_out][0,:,0])
plt.show()