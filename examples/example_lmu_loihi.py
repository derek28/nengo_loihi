from collections import deque

import matplotlib.pyplot as plt
import numpy as np

import nengo
import nengo_dl

# parameters of LMU
theta = 1.0  # length of window (in seconds)
order = 6  # number of Legendre polynomials representing window

# parameters of input signal
freq = 2  # frequency limit
rms = 0.30  # amplitude of input (set to keep within [-1, 1])
delay = 0.5  # length of time delay network will learn

# simulation parameters
dt = 0.001  # simulation timestep
sim_t = 30  # length of simulation
seed = 10  # fixed for deterministic results

# compute the A and B matrices according to the LMU's mathematical derivation
# (see the paper for details)
Q = np.arange(order, dtype=np.float64)
R = (2 * Q + 1)[:, None] / theta
j, i = np.meshgrid(Q, Q)

A = np.where(i < j, -1, (-1.0) ** (i - j + 1)) * R
B = (-1.0) ** Q[:, None] * R

class IdealDelay(nengo.synapses.Synapse):
    def __init__(self, delay):
        super().__init__()
        self.delay = delay

    def make_state(self, *args, **kwargs):  # pylint: disable=signature-differs
        return {}

    def make_step(self, shape_in, shape_out, dt, rng, state):
        # buffer the input signal based on the delay length
        buffer = deque([0] * int(self.delay / dt))

        def delay_func(t, x):
            buffer.append(x.copy())
            return buffer.popleft()

        return delay_func


with nengo.Network(seed=seed) as net:
    # create the input signal
    stim = nengo.Node(
        output=nengo.processes.WhiteSignal(
            high=freq, period=sim_t, rms=rms, y0=0, seed=seed
        )
    )

    # probe input signal and an ideally delayed version of input signal
    p_stim = nengo.Probe(stim)
    p_ideal = nengo.Probe(stim, synapse=IdealDelay(delay))

# run the network (on the superhost) and display results
with nengo_dl.Simulator(net) as sim:
    sim.run(10)

    plt.figure(figsize=(16, 6))
    plt.plot(sim.trange(), sim.data[p_stim], label="input")
    plt.plot(sim.trange(), sim.data[p_ideal], label="ideal")
    plt.legend()
    plt.show()

exit()

with net:
    # nengo_loihi.set_defaults()

    lmu = nengo.networks.EnsembleArray(
        n_neurons=100, n_ensembles=order, neuron_type=nengo.SpikingRectifiedLinear()
    )
    tau = 0.1  # synaptic filter on LMU connections
    nengo.Connection(stim, lmu.input, transform=B * tau, synapse=tau)
    nengo.Connection(
        lmu.output, lmu.input, transform=A * tau + np.eye(order), synapse=tau
    )
    # nengo.Connection(stim, lmu.input, transform=B, synapse=tau)
    # nengo.Connection(
    #     lmu.output, lmu.input, transform=A, synapse=tau
    # )

with net:
    ens = nengo.Ensemble(1000, order, neuron_type=nengo.SpikingRectifiedLinear())
    nengo.Connection(lmu.output, ens)

    out = nengo.Node(size_in=1)

    # we'll use a Node to compute the error signal so that we can shut off
    # learning after a while (in order to assess the network's generalization)
    err_node = nengo.Node(lambda t, x: x if t < sim_t * 0.8 else 0, size_in=1)

    # the target signal is the ideally delayed version of the input signal,
    # which is subtracted from the ensemble's output in order to compute the
    # PES error
    nengo.Connection(stim, err_node, synapse=IdealDelay(delay), transform=-1)
    nengo.Connection(out, err_node, synapse=None)

    learn_conn = nengo.Connection(
        ens, out, function=lambda x: 0, learning_rule_type=nengo.PES(5e-4)
    )
    nengo.Connection(err_node, learn_conn.learning_rule, synapse=None)

    p_out = nengo.Probe(out)

# model = nengo_loihi.builder.Model(dt=dt)
model = nengo.builder.Model(dt=dt)
model.pes_error_scale = 200.0
# with nengo_loihi.Simulator(net, model=model) as sim:
with nengo.Simulator(net, model=model) as sim:
    sim.run(sim_t)

# we'll break up the output into multiple plots, just for
# display purposes
t_per_plot = 6
for i in range(sim_t // t_per_plot):
    plot_slice = (sim.trange() >= t_per_plot * i) & (
        sim.trange() < t_per_plot * (i + 1)
    )

    plt.figure(figsize=(16, 6))
    plt.plot(sim.trange()[plot_slice], sim.data[p_stim][plot_slice], label="input")
    plt.plot(sim.trange()[plot_slice], sim.data[p_ideal][plot_slice], label="ideal")
    plt.plot(sim.trange()[plot_slice], sim.data[p_out][plot_slice], label="output")
    if i * t_per_plot < sim_t * 0.8:
        plt.title("Learning ON")
    else:
        plt.title("Learning OFF")
    plt.ylim([-1, 1])
    plt.legend()
plt.show()