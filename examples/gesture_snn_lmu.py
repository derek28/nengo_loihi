import matplotlib.pyplot as plt
import numpy as np

import nengo
import nengo_dl

from nengo.transforms import Transform
from nengo.processes import PresentInput
from nengo.dists import Uniform
from nengo.utils.matplotlib import rasterplot

# parameters of LMU
theta = 1.5
order = 10

dt = 0.003
sim_t = int(theta/dt)
seed = 0

d_data = 6
n_data = 1200

with open("test.txt", "r")  as in_file:
    # data_lines = in_file.readlines()
    data_line = in_file.readline()

# data_line = data_lines[0]
data_vals = np.array(data_line.split(" ")).astype(int)
data = data_vals[:-1]
data = np.reshape(data,(1,sim_t,d_data))
label = data_vals[-1]

# compute the A and B matrices according to the LMU's mathematical derivation
# (see the paper for details)
Q = np.arange(order, dtype=np.float64)
R = (2 * Q + 1)[:, None] / theta
j, i = np.meshgrid(Q, Q)

A = np.where(i < j, -1, (-1.0) ** (i - j + 1)) * R
B = (-1.0) ** Q[:, None] * R

with nengo.Network(seed=seed) as net:
    nengo_dl.configure_settings(
        trainable=None,
        stateful=True,
        keep_history=True,
    )

    x = nengo.Node(np.zeros(d_data))
    u = nengo.Ensemble(n_neurons=100, dimensions=1, max_rates=1/dt)
    m = nengo.networks.EnsembleArray(
        n_neurons=100, n_ensembles=order, neuron_type=nengo.SpikingRectifiedLinear()
    )
    h = nengo.Ensemble(n_neurons=100, dimensions=10, max_rates=1/dt, neuron_type=nengo.RegularSpiking(nengo.Tanh()))
    exit()
    # tau = 0.1
    # nengo.Connection(inp, lmu.input, transform=B*tau, synapse=tau)
    # nengo.Connection(
    #     lmu.output, lmu.input, transform=A*tau+np.eye(order), synapse=tau
    # )

with nengo_dl.Simulator(net, dt=dt, minibatch_size=1) as sim:
    sim.run(theta,data={inp:data})

# print(sim.data[filtered].shape)
# # Plot the spiking output of the ensemble
# plt.figure(figsize=(10, 8))
# plt.subplot(221)
# rasterplot(sim.trange(),sim.data[spikes][0])
# plt.ylabel("Neurfon")
# plt.xlim(0, theta)

# # Plot the soma voltages of the neurons
# plt.subplot(222)
# plt.plot(sim.trange(),sim.data[filtered][0])
# plt.plot(sim.trange(),sim.data[input_probe][0])
# plt.show()