#! /usr/bin/env python3

'''

 Filename: to_spikes_loc.py
 Author: Kai Zheng
 Date: 09/03/2022
 Function: Turn the dataset into spike data. 

'''
 
import numpy as np
import matplotlib.pyplot as plt

import nengo
import nengo_dl
from nengo.dists import Uniform
from nengo.utils.matplotlib import rasterplot

n_samp = 10000
n_pts = 180
n_ant = 6

n_sig_pts = n_pts * n_ant

n_step = 500
dt = 0.003
t_sim = dt * n_step

x_data = np.zeros((n_samp, n_step, n_ant))   # S_n[t] 
y_data = np.zeros((n_samp, n_step, 4))      

# expand a frame with 180 points to 1500 points. a - normalize to -1 to 1. 
def expand_frame(frame, n_in=180, n_out=250, a=8):
    n_sel = np.linspace(0, n_in-1, n_out)
    n_sel = n_sel.astype('int')
    return (a * frame[n_sel])

# process the raw data
with open("local_6_10000.txt", "r") as in_file:
    data_lines = in_file.readlines()

for i in range(n_samp):
    line = data_lines[i]
    items = np.array(line.split(","))
    items = items.astype(np.float64)

    for j in range(n_ant):
        frame_ex = expand_frame(items[j*n_pts : (j+1)*n_pts], n_pts, n_step)
        x_data[i, :, j] = frame_ex
    y_data[i, :, 0] = items[n_sig_pts]    # x1 
    y_data[i, :, 1] = items[n_sig_pts+1]  # y1 
    y_data[i, :, 2] = items[n_sig_pts+2]  # x2 
    y_data[i, :, 3] = items[n_sig_pts+3]  # y2 

# Build a neural network
net = nengo.Network(label="Localization Neurons")
with net:
    nengo_dl.configure_settings(trainable=True)
    net.config[nengo.Ensemble].max_rates = nengo.dists.Choice([100])
    net.config[nengo.Ensemble].intercepts = nengo.dists.Choice([0.05])
    net.config[nengo.Connection].synapse = None
    neuron_type = nengo.LIF(amplitude=0.01)
    #neuron_type = nengo.SpikingRectifiedLinear(amplitude=0.01)
    
    nengo_dl.configure_settings(stateful=False)
    
    # input shape n_ant * n_step * 1
    #inp = nengo.Node(np.zeros(n_ant))
    inp = nengo.Node(lambda t: np.tile(np.sin(8*t), (6)))

    # add a to-spike layer 
    to_spike = nengo.Ensemble(
        n_neurons=12,
        dimensions=6,
        encoders=[
            [1,  0,  0,  0,  0,  0],
            [-1, 0,  0,  0,  0,  0],
            [0,  1,  0,  0,  0,  0],
            [0, -1,  0,  0,  0,  0],
            [0,  0,  1,  0,  0,  0],
            [0,  0, -1,  0,  0,  0],
            [0,  0,  0,  1,  0,  0],
            [0,  0,  0, -1,  0,  0],
            [0,  0,  0,  0,  1,  0],
            [0,  0,  0,  0, -1,  0],
            [0,  0,  0,  0,  0,  1],
            [0,  0,  0,  0,  0, -1],
        ]) 

    conn_in_spk = nengo.Connection(inp, to_spike, synapse=None)
    
#    inp_probe = nengo.Probe(inp, label="inp_p")
    to_spike_probe = nengo.Probe(to_spike.neurons, label="to_spike_p")
#    to_spike_probe_filt = nengo.Probe(to_spike.neurons, synapse=0.05, label="to_spike_p_filt")

minibatch_size = n_samp
sim = nengo_dl.Simulator(net, dt=dt, minibatch_size=minibatch_size)

sim.run(t_sim, data={inp: x_data[:minibatch_size]})
data = sim.data

print(data[to_spike_probe].shape)
#print(data[inp_probe].shape)

t_range = np.linspace(0, t_sim, n_step)

for fr in range(3):
    plt.figure()
    rasterplot(t_range, data[to_spike_probe][fr])
    plt.title("To-Spike "+str(fr))

    #plt.figure()
    #for i in range(n_ant):
#        plt.plot(t_range, data[inp_probe][fr][:, i] - i)
    #plt.title("Original "+str(fr))
    

with open("loc_spike_12_10000.txt", "w")  as out_file:
    for i in range (n_samp):
        spikes = (data[to_spike_probe][i] > 0.1).astype(int)
        for j in range(n_ant*2):
            for k in range(n_step):
                out_file.write("%d " % spikes[k, j])
        out_file.write("%.4f " % (y_data[i, -1, 0])) 
        out_file.write("%.4f " % (y_data[i, -1, 1])) 
        out_file.write("%.4f " % (y_data[i, -1, 2])) 
        out_file.write("%.4f\n" % (y_data[i, -1, 3])) 
        # write spikes to file.
#plt.show()
