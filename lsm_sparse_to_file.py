#! /usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import rand
from numpy.random import randint
from scipy.linalg import eig

import nengo
import nengo_dl
from nengo.utils.matplotlib import rasterplot
from nengo.dists import Uniform
#from nengo.processes import Piecewise

dt = 0.001
t_sim = 2.0
total_pts = int(t_sim / dt)

ens_seed = 1
np_seed = 1
np.random.seed(np_seed)
rng = np.random.RandomState(np_seed)

class InputFunc:
    def __init__(self, filename):
        self.length = 1.5
        self.nchans = 6
        self.nsamps = 2400
        self.npts = 1500
        self.total_pts = total_pts

        # valid data points 1500, add 1500 zeros following that
        self.xdata = np.zeros((self.nsamps, self.total_pts, self.nchans))
        self.ydata = np.zeros((self.nsamps))

        self.samp_id = 0

        with open(filename, "r")  as in_file:
            data_lines = in_file.readlines()
        
        n_signal = self.nchans * self.npts
        for i in range(self.nsamps):
            line = data_lines[i]
            items = np.array(line.split(" "))
            items = items.astype(int) 
            self.ydata[i] = items[n_signal]    # 0-11 for sparse_catogorial_enthropy
            #print(str(int(self.ydata[i])))
            items = items * 1000                # scale by 1000
            self.xdata[i, :self.npts, :] = np.reshape(items[:n_signal], (self.nchans, self.npts)).T

        perm = rng.permutation(self.xdata.shape[0])
        self.xdata = self.xdata[perm, :]
        self.ydata = self.ydata[perm]
        return  

    def set_id(self, samp_id):
        self.samp_id = samp_id
        return 0

    def step(self, t):
        ii = int(t / dt - dt)
        if ii >= self.npts:
            return np.zeros(self.nchans)
        else:
            return self.xdata[self.samp_id, ii, :]

my_input = InputFunc("gesture_2400_1ms.txt")

model = nengo.Network(label="lsm")
with model:
    n_neurons = 256
    n_inputs = 6
    n_outputs = 12
    # The ensemble for the oscillator
    pool = nengo.Ensemble(
        n_neurons, 
        dimensions=1, 
        neuron_type = nengo.LIF(),
        intercepts=Uniform(0, 0),
        max_rates=Uniform(100, 100),
        seed=ens_seed
    )

    inp = nengo.Node(my_input.step)
    #inp = nengo.Node(np.zeros(6))
    #outp = nengo.Node(size_in=n_outputs)
    
    gain = 0.01 # normalize by average firing rate
    in_density = 0.3
    lsm_density = 0.7
    n_in_conn = int(in_density * n_neurons)
    n_lsm_conn = int(lsm_density * n_neurons) 
    # initialize input connection to the reservoir
    w_in = np.zeros((n_neurons, n_inputs))
    for i in range(int(n_inputs/2)):
        for j in range(n_in_conn):
            weight = rand() - 0.5
            neuron_sel = randint(0, n_neurons - 1)
            w_in[neuron_sel, 2 * i] = weight
            w_in[neuron_sel, 2 * i + 1] = -weight
    #w_in *= 0.1
    
    #w_bias = rand(n_neurons, 1) - 0.5
    w_bias = np.zeros((n_neurons, 1))
    
    w_pool = np.zeros((n_neurons, n_neurons))
    for i in range(n_neurons):
        for j in range(n_lsm_conn):
            w_pool[randint(0, n_neurons - 1), i] = rand() - 0.5

    #w_pool = rand(n_neurons, n_neurons) - 0.5
    coeff = gain / max(abs(eig(w_pool)[0]))
    print("coeff = %f" % coeff)
    w_pool *= coeff

    # initialize the readout layer connection
    #w_out = rand(n_outputs, n_neurons) - 0.5
    
    nengo.Connection(inp, pool.neurons, transform=w_in)
    nengo.Connection(pool.neurons, pool.neurons, transform=w_pool, synapse=0.05)
    #nengo.Connection(pool.neurons, outp, transform=w_out)

    # Indicate which values to record
    in_probe = nengo.Probe(inp, synapse=0)
    pool_neuron_probe = nengo.Probe(pool.neurons)
    #out_probe_filt = nengo.Probe(outp, synapse=0.03)
    #out_probe = nengo.Probe(outp, synapse=0)

write_file = False
batch_size = 40
for i in range(1):
    print("Batch #%d." % i)
    with nengo_dl.Simulator(model, minibatch_size=batch_size) as sim:
    # batch processing. shape: (minibatch_size, n_steps, node_size)
        sim.run(t_sim, data={inp:my_input.xdata[batch_size*i:batch_size*(i+1), :]})
        print(sim.data[pool_neuron_probe].shape)

        if write_file:
            for j in range(batch_size):
                samp = batch_size * i + j 
                print("Sample #%d" % samp)
                filename = "gesture_lsm_003/gesture_" + str(samp+1) + ".txt"
                with open(filename, "w") as out_file:
                    out_file.write(str(int(my_input.ydata[samp])) + "\n")   # line 1 for label. 
                    dat = sim.data[pool_neuron_probe][j]
                    print(dat.shape)
                    for k in range(total_pts): 
                        ts = k / 1000.0
                        for l in range(n_neurons): 
                            if dat[k, l] > 0.0:
                                out_file.write(str(l) + ' ' + str(ts) +"\n")

# plot sample output. label wrong for batch_size > 1
for i in range(5):
    plt.figure()
    rasterplot(sim.trange(), sim.data[in_probe][i])
    plt.title("%d" % int(my_input.ydata[i]))
    plt.xlabel("Time (s)")

    plt.figure()
    rasterplot(sim.trange(), sim.data[pool_neuron_probe][i])
    plt.title("%d" % int(my_input.ydata[i]))
    plt.xlabel("Time (s)")
plt.show()
