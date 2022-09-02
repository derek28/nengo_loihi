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
n_pts = 250
n_ant = 3

n_sig_pts = n_pts * n_ant

#n_step = 1000
t_sim = 1.0
dt = 0.001
n_step = int(t_sim / dt)

n_train = int(0.8 * n_samp)
n_test = n_samp - n_train

x_data = np.zeros((n_samp, n_step, n_ant))   # S_n[t] 
y_data = np.zeros((n_samp, n_step, 1))   #  coordinate <x[t], y[t]>

# expand a frame with 180 points to 1500 points. a - normalize to -1 to 1. 
def expand_frame(frame, n_in=250, n_out=1000, a=1.8):
    n_sel = np.linspace(0, n_in-1, n_out)
    n_sel = n_sel.astype('int')
    return (a * frame[n_sel])

# process the raw data
with open("gesture_8_10000.txt", "r")  as in_file:
    data_lines = in_file.readlines()

for i in range(n_samp):
    line = data_lines[i]
    items = np.array(line.split(","))
    items = items.astype(np.float64)

    for j in range(n_ant):
        frame_ex = expand_frame(items[j*n_pts : (j+1)*n_pts], n_pts, n_step)
        x_data[i, :, j] = frame_ex
    y_data[i, :, 0] = items[n_sig_pts]-1    # 0-7 for sparse_catogorial_enthropy

# divide the dataset into training and testing group
x_train = x_data[:n_train, :, :]
y_train = y_data[:n_train, :, :]
x_test = x_data[n_train:, :, :]
y_test = y_data[n_train:, :, :]

# Build a neural network
net = nengo.Network(label="Gesture Recognition Neurons")
with net:
    nengo_dl.configure_settings(trainable=True)
    net.config[nengo.Ensemble].max_rates = nengo.dists.Choice([100])
    net.config[nengo.Ensemble].intercepts = nengo.dists.Choice([0])
    net.config[nengo.Connection].synapse = None
    neuron_type = nengo.LIF(amplitude=0.01)
    #neuron_type = nengo.SpikingRectifiedLinear(amplitude=0.01)
    
    nengo_dl.configure_settings(stateful=False)
    
    # input shape n_ant * n_step * 1
    inp = nengo.Node(np.zeros(n_ant))

    # add a to-spike layer 
    encoders=[
        [1,  0,  0],
        [-1, 0,  0],
        [0,  1,  0],
        [0, -1,  0],
        [0,  0,  1],
        [0,  0, -1],
    ] 
    
    #to_spike = nengo_dl.TensorNode(tf.keras.layers.Dense(6), pass_time=False, shape_in=(3, ))    
    to_spike = nengo.Ensemble(
        n_neurons=6,
        dimensions=3,
        encoders=encoders
    ) 
    conn_in_spk = nengo.Connection(inp, to_spike, synapse=None)
    net.config[conn_in_spk].trainable = False

    delay0 = nengo_dl.TensorNode(tf.keras.layers.Dense(80), pass_time=False, shape_in=(6, ))
    conn_in_0 = nengo.Connection(to_spike.neurons, delay0, synapse=None)
    conn_in_1 = nengo.Connection(to_spike.neurons, delay0, synapse=0.04)
    delay0 = nengo_dl.Layer(neuron_type)(delay0)

    delay1 = nengo_dl.TensorNode(tf.keras.layers.Dense(80), pass_time=False, shape_in=(80, ))
    conn_d01_0 = nengo.Connection(delay0, delay1, synapse=None)
    conn_d01_1 = nengo.Connection(delay0, delay1, synapse=0.08)
    delay1 = nengo_dl.Layer(neuron_type)(delay1)

    delay2 = nengo_dl.TensorNode(tf.keras.layers.Dense(80), pass_time=False, shape_in=(80, ))
    conn_d12_0 = nengo.Connection(delay1, delay2, synapse=None)
    conn_d12_1 = nengo.Connection(delay1, delay2, synapse=0.16)
    delay2 = nengo_dl.Layer(neuron_type)(delay2)

    delay3 = nengo_dl.TensorNode(tf.keras.layers.Dense(80), pass_time=False, shape_in=(80, ))
    conn_d23_0 = nengo.Connection(delay2, delay3, synapse=None)
    conn_d23_1 = nengo.Connection(delay2, delay3, synapse=0.32)
    delay3 = nengo_dl.Layer(neuron_type)(delay3)

#    delay4 = nengo_dl.TensorNode(tf.keras.layers.Dense(80), pass_time=False, shape_in=(80, ))
#    conn_d34_0 = nengo.Connection(delay3, delay4, synapse=None)
#    conn_d34_1 = nengo.Connection(delay3, delay4, synapse=0.32)
#    delay4 = nengo_dl.Layer(neuron_type)(delay4)

    outp = nengo_dl.TensorNode(tf.keras.layers.Dense(8), pass_time=False, shape_in=(80, ))
    nengo.Connection(delay3, outp)

    inp_probe = nengo.Probe(inp, label="inp_p")
    to_spike_probe = nengo.Probe(to_spike.neurons, label="to_spike_p")
    delay0_probe = nengo.Probe(delay0[0:10], label='delay0')
    delay3_probe = nengo.Probe(delay3[0:10], label='delay4')
    out_probe = nengo.Probe(outp, label="out_p")
    out_probe_filt = nengo.Probe(outp, synapse=0.16, label="out_p_filt")

minibatch_size = 200
sim = nengo_dl.Simulator(net, dt=dt, minibatch_size=minibatch_size)

do_training = True
if do_training:
    sim.compile(
        optimizer=tf.optimizers.RMSprop(0.001),
        loss = {out_probe: tf.losses.SparseCategoricalCrossentropy(from_logits=True)}
    )
    sim.fit(x_train, {out_probe: y_train}, epochs=10)

    # save the parameters
    sim.save_params("./gesture_snn_params")
else:
    sim.load_params("./gesture_snn_params")


def classification_accuracy(y_true, y_pred):
    return tf.metrics.sparse_categorical_accuracy(y_true[:, -1], y_pred[:, -1])
    
sim.compile(loss={out_probe_filt: classification_accuracy})
print(
    "Accuracy after training:",
    sim.evaluate(x_test, {out_probe_filt: y_test})["loss"], 
    )


data = sim.predict(x_test[:minibatch_size])

print(data[to_spike_probe].shape)
print(data[inp_probe].shape)
print(data[out_probe].shape)
print(data[out_probe_filt].shape)

t_range = np.linspace(0, 1, n_step)

fr = 0

plt.figure()
rasterplot(t_range, data[to_spike_probe][fr])
plt.title("To-Spike Probe")

plt.figure()
rasterplot(t_range, data[delay0_probe][fr])
plt.title("Delay0 Probe")

plt.figure()
rasterplot(t_range, data[delay3_probe][fr])
plt.title("Delay3 Probe")

for fr in range(4):
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    for i in range(n_ant):
        dat = data[inp_probe][fr][:, i] + (n_ant - i)
        plt.plot(t_range, dat)
    
    plt.subplot(1, 2, 2)
    plt.plot(t_range, tf.nn.softmax(data[out_probe_filt][fr]))
    plt.legend([str(i) for i in range(8)], loc="upper left")
    plt.ylabel("probability")
    plt.title("Gesture"+str(y_test[fr, 0, 0]))
plt.show()

