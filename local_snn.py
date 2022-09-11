#! /usr/bin/env python3

'''

 Filename: local_cnn.py
 Author: Kai Zheng
 Date: 08/17/2022
 Function: localization for SIL Radar sensor array. Using Nengo_loihi.

'''
 
import numpy as np
import matplotlib.pyplot as plt

import nengo
import nengo_dl
import nengo_loihi

import tensorflow as tf
from tensorflow.keras import layers, models

n_samp = 10000
n_pts = 500
n_chan = 12

n_signal = n_pts * n_chan

n_traj_pts = 2

n_train = int(0.8*n_samp)
n_test = n_samp - n_train

x_data = np.zeros((n_samp, n_signal))   # nengo dl only accepts 1D data type
y_data = np.zeros((n_samp, n_traj_pts * 2))   # <x1, y1> and <x2, y2>

with open("loc_spike_12_10000.txt", "r")  as in_file:
    data_lines = in_file.readlines()

for i in range(n_samp):
    line = data_lines[i]
    items = np.array(line.split(" "))
    items = items.astype(np.float64)
    
    x_data[i, :] = items[0:n_signal].astype(int)

    loc_start_x = items[n_signal]
    loc_start_y = items[n_signal + 1]
    loc_end_x = items[n_signal + 2]
    loc_end_y = items[n_signal + 3]

    y_data[i, :] = np.concatenate((np.linspace(loc_start_x, loc_end_x, n_traj_pts),
            np.linspace(loc_start_y, loc_end_y, n_traj_pts)))
    

# Divide the data into training group and test group
x_train = x_data[0:n_train, :] 
y_train = y_data[0:n_train, :]
x_test = x_data[n_train:, :]
y_test = y_data[n_train:, :]

plt.figure()
plt.plot(y_train[0, 0:2], y_train[0, 2:4])

#plt.show()

# Add time dimension
x_train = x_train[:, None, :] 
y_train = y_train[:, None, :]
x_test = x_test[:, None, :]
y_test = y_test[:, None, :]

print(x_train.shape)
print(y_train.shape)

# Building a neural network
# Input
inp = tf.keras.Input(shape=(12, 500, 1), name="input")

to_spikes = layers.Activation(tf.nn.relu)(inp)

# Convolutional Layer 0 
conv0 = layers.Conv2D(
    filters=64, 
    kernel_size=(2,16),
    #strides=(2, 4),
    strides=(2, 1),
    activation=tf.nn.relu,
)(to_spikes)

pool0 = layers.AveragePooling2D((1, 4))(conv0)

# Convolutional Layer 1 
conv1 = layers.Conv2D(
    filters=64,
    kernel_size=(3, 8),
    #strides=(1, 2),
    activation=tf.nn.relu,
)(pool0)

# Average Pooling Layer 0
pool1 = layers.AveragePooling2D((1, 2))(conv1)

# Convolutional Layer 2 
conv2 = layers.Conv2D(
    filters=96,
    kernel_size=(4, 8),
    #strides=(1, 2),
    activation=tf.nn.relu,
)(pool1)

# Max Pooling Layer 2
pool2 = layers.AveragePooling2D((1, 2))(conv2)

# Flatten
flatten = layers.Flatten()(pool2)

# Dense layer 0
dense = layers.Dense(180, 
    activation=tf.nn.relu,
)(flatten)

# Output Layer (Dense)
outp = layers.Dense(4)(dense)

model = models.Model(inputs=inp, outputs=outp)
model.summary()

# NengoDL Converter. Keras --> Nengo
converter = nengo_dl.Converter(model)

# It's important to note that we are using standard (non-spiking) ReLU neurons at this point.
do_training = False
if do_training:
    with nengo_dl.Simulator(converter.net, minibatch_size=40) as sim:
        # run training
        sim.compile(
            optimizer=tf.optimizers.Adam(0.001),
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=['mse'],
        )
        sim.fit(
            {converter.inputs[inp]: x_train},
            {converter.outputs[outp]: y_train},
            validation_data=(
                {converter.inputs[inp]: x_test},
                {converter.outputs[outp]: y_test},
            ),
            epochs=20,
        )

        # save the parameters to file
        sim.save_params("loc_params")

def run_network(
    activation,
    params_file="loc_params",
    n_steps=30,
    scale_firing_rates=1,
    synapse=None,
    n_test=400,
):
    # convert the keras model to a nengo network
    nengo_converter = nengo_dl.Converter(
        model,
        swap_activations={tf.nn.relu: activation},
        scale_firing_rates=scale_firing_rates,
        synapse=synapse,
    )

    # get input/output objects
    nengo_input = nengo_converter.inputs[inp]
    nengo_output = nengo_converter.outputs[outp]

    # add a probe to the first convolutional layer to record activity.
    # we'll only record from a subset of neurons, to save memory.
    conv0_neurons = np.linspace(
        0,
        np.prod(conv0.shape[1:]),
        100,
        endpoint=False,
        dtype=np.int32,
    )
    conv1_neurons = np.linspace(
        0,
        np.prod(conv1.shape[1:]),
        100,
        endpoint=False,
        dtype=np.int32,
    )
    conv2_neurons = np.linspace(
        0,
        np.prod(conv2.shape[1:]),
        100,
        endpoint=False,
        dtype=np.int32,
    )
    to_spikes_neurons = np.linspace(
        0,
        np.prod(to_spikes.shape[1:]),
        100,
        endpoint=False,
        dtype=np.int32,
    )
    with nengo_converter.net:
        conv_probe = nengo.Probe(nengo_converter.layers[conv0][conv0_neurons], label="conv_p")
        #conv_probe_1 = nengo.Probe(nengo_converter.layers[conv2][conv2_neurons], label="conv_p1")
        #to_spikes_probe = nengo.Probe(nengo_converter.layers[to_spikes][to_spikes_neurons])
        #out_probe = nengo.Probe(nengo_converter.outputs[outp], label="outp")
        #out_probe_filt = nengo.Probe(nengo_converter.outputs[outp], synapse=0.1, label="outp_filt")

    # repeat inputs for some number of timesteps
    tiled_x_test = np.tile(x_test[:n_test], (1, n_steps, 1))

    # set some options to speed up simulation
    with nengo_converter.net:
        nengo_dl.configure_settings(stateful=False)

    # build network, load in trained weights, run inference on test images
    with nengo_dl.Simulator(
        nengo_converter.net, minibatch_size=20, progress_bar=False
    ) as nengo_sim:
        nengo_sim.load_params(params_file)
        data = nengo_sim.predict({nengo_input: tiled_x_test})

    # compute mse on test data, using output of network on
    # last timestep
    #print(data[nengo_output].shape)
    #print(y_test.shape)
    loc_pred = data[nengo_output][:, -1]
    loc_true = y_test[:n_test, -1]
    #print(loc_pred.shape)
    #print(loc_true.shape)
    mse = tf.metrics.mean_squared_error(loc_true, loc_pred)
    print(np.mean(mse))

    # plot the results
    for ii in range(3):
        plt.figure(figsize=(15, 4))
	
        loc_x_true = y_test[ii][-1][0:2]
        loc_y_true = y_test[ii][-1][2:4]
        loc_x_pred = data[nengo_output][ii][-1][0:2]
        loc_y_pred = data[nengo_output][ii][-1][2:4]
        
        plt.subplot(1, 3, 1)
        plt.title("Object Trajectory")
        plt.plot(loc_x_true, loc_y_true, label='True')
        plt.plot(loc_x_pred, loc_y_pred, '--', label='Pred')
        plt.legend(loc='lower right')
        plt.xlim([-3, 3])
        plt.ylim([0, 6])

        loc_x1_pred_t = data[nengo_output][ii][:, 0]
        loc_x2_pred_t = data[nengo_output][ii][:, 1]
        loc_y1_pred_t = data[nengo_output][ii][:, 2]
        loc_y2_pred_t = data[nengo_output][ii][:, 3]

        plt.subplot(1, 3, 2)
        plt.plot(loc_x1_pred_t, loc_y1_pred_t)
        plt.plot(loc_x2_pred_t, loc_y2_pred_t)
        plt.scatter(loc_x_true[0], loc_y_true[0], label='Start')
        plt.scatter(loc_x_true[1], loc_y_true[1], label='End')
        plt.title("Localization Result vs. Timestep")
        plt.xlim([-3, 3])
        plt.ylim([0, 6])
        plt.legend()

        plt.subplot(1, 3, 3)
        scaled_conv_probe_data = data[conv_probe][ii] * scale_firing_rates
        if isinstance(activation, nengo.SpikingRectifiedLinear):
            scaled_conv_probe_data *= 0.001
            rates = np.sum(scaled_conv_probe_data, axis=0) / (n_steps * nengo_sim.dt)
            plt.ylabel("Number of spikes")
        else:
            rates = scaled_conv_probe_data
            plt.ylabel("Firing rates (Hz)")
        plt.xlabel("Timestep")
        plt.title(
            f"conv2 mean={rates.mean():.1f} Hz, "
            f"max={rates.max():.1f} Hz"
        )
        plt.plot(scaled_conv_probe_data)
        plt.tight_layout()

#run_network(activation=nengo.RectifiedLinear(), n_steps=30) 
run_network(activation=nengo.SpikingRectifiedLinear(), n_steps=250, scale_firing_rates=100, synapse=0.05) 

plt.show()
