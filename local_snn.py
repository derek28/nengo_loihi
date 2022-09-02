#! /usr/bin/env python3

'''

 Filename: local.py
 Author: Kai Zheng
 Date: 08/28/2022
 Function: localization for SIL Radar sensor array. Using Nengo_loihi.

'''
 
import numpy as np
import matplotlib.pyplot as plt

import nengo
import nengo_dl
import nengo_loihi
from nengo.dists import Uniform
from nengo.utils.matplotlib import rasterplot

import tensorflow as tf
from tensorflow.keras import layers, models

n_samp = 10000
n_pts = 180
n_ant = 6

n_sig_pts = n_pts * n_ant

n_step = 1500

n_train = int(0.8 * n_samp)
n_test = n_samp - n_train

x_data = np.zeros((n_samp, n_step, n_ant))   # S_n[t] 
y_data = np.zeros((n_samp, n_step, 2))   #  coordinate <x[t], y[t]>

# expand a frame with 180 points to 1500 points. a - normalize to -1 to 1. 
def expand_frame(frame, n_in=180, n_out=1500, a=10):
    n_sel = np.linspace(0, n_in-1, n_out)
    n_sel = n_sel.astype('int')
    return (a * frame[n_sel])

# process the raw data
with open("local_6_10000.txt", "r")  as in_file:
    data_lines = in_file.readlines()

ax = 6 # location normalization factor -x
ay = 6 # y 

for i in range(n_samp):
    line = data_lines[i]
    items = np.array(line.split(","))
    items = items.astype(np.float64)

    for j in range(n_ant):
        frame_ex = expand_frame(items[j*n_pts : (j+1)*n_pts], n_pts, n_step)
        x_data[i, :, j] = frame_ex

    loc_start_x = items[n_sig_pts]
    loc_start_y = items[n_sig_pts + 1]
    loc_end_x = items[n_sig_pts + 2]
    loc_end_y = items[n_sig_pts + 3]

    y_data[i, :, 0] = (np.linspace(loc_start_x, loc_end_x, n_step) + 3) / ax
    y_data[i, :, 1] = np.linspace(loc_start_y, loc_end_y, n_step) / ay

# divide the dataset into training and testing group
x_train = x_data[:n_train, :, :]
y_train = y_data[:n_train, :, :]
x_test = x_data[n_train:, :, :]
y_test = y_data[n_train:, :, :]

# Build a neural network
net = nengo.Network(label="Localization Neurons")
with net:
    net.config[nengo.Ensemble].max_rates = nengo.dists.Choice([100])
    net.config[nengo.Ensemble].intercepts = nengo.dists.Choice([0])
    net.config[nengo.Connection].synapse = 0.001
    neuron_type = nengo.LIF(amplitude=0.01)
    #neuron_type = nengo.SpikingRectifiedLinear(amplitude=0.01)
    
    nengo_dl.configure_settings(stateful=False)
    
    # input shape n_ant * n_step * 1
    inp = nengo.Node(np.zeros(n_ant))

    # add to-spike layer using ensemble!
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

    nengo.Connection(inp, to_spike)

    x = nengo_dl.Layer(tf.keras.layers.Dense(80))(to_spike)
    x = nengo_dl.Layer(neuron_type)(x)

    x = nengo_dl.Layer(tf.keras.layers.Dense(80))(x)
    x = nengo_dl.Layer(neuron_type)(x)

    x = nengo_dl.Layer(tf.keras.layers.Dense(20))(x)
    x = nengo_dl.Layer(neuron_type)(x)

    outp = nengo_dl.Layer(tf.keras.layers.Dense(2))(x)

    inp_probe = nengo.Probe(inp, label="inp_p")
    to_spike_probe = nengo.Probe(to_spike.neurons, label="to_spike_p")
    out_probe = nengo.Probe(outp, label="out_p")
    out_probe_filt = nengo.Probe(outp, synapse=0.3, label="out_p_filt")

minibatch_size = 200
sim = nengo_dl.Simulator(net, minibatch_size=minibatch_size)

do_training = True
if do_training:
    sim.compile(
        optimizer=tf.optimizers.RMSprop(0.001),
        loss = {out_probe: tf.losses.MeanSquaredError()}
    )
    sim.fit(x_train, {out_probe: y_train}, epochs=5)

    # save the parameters
    sim.save_params("./local_snn_params")
else:
    sim.load_params("./local_snn_params")

sim.compile(loss={out_probe_filt: tf.metrics.mean_squared_error})
print(
    "MSE after training:",
    36 * sim.evaluate(x_test, {out_probe_filt: y_test})["loss"], 
    )


data = sim.predict(x_test[:minibatch_size, :, :])

print(data[to_spike_probe].shape)
print(data[inp_probe].shape)
print(data[out_probe].shape)
print(data[out_probe_filt].shape)

t_range = np.linspace(0, 1.5, 1500)
fr = 1

plt.figure()
rasterplot(t_range, data[to_spike_probe][fr])

plt.figure()
for i in range(n_ant):
    dat = data[inp_probe][fr][:, i] + (n_ant - i)
    plt.plot(t_range, dat)

plt.figure()
rasterplot(t_range, data[out_probe][fr])

for fr in range(4):
    plt.figure()
    x_pred = data[out_probe_filt][fr][:, 0] * ax - 3
    y_pred = data[out_probe_filt][fr][:, 1] * ay
    x_true = y_test[fr][:, 0] * ax - 3
    y_true = y_test[fr][:, 1] * ay
    plt.plot(x_pred, y_pred)
    plt.plot(x_true, y_true)
    plt.xlim((-3, 3))
    plt.ylim((0, 6))

plt.show()

