#! /usr/bin/env python3

'''

 Filename: local.py
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
n_pts = 180
n_ant = 6

n_sig_pts = n_pts * n_ant

n_traj_pts = 2

n_train = int(0.8*n_samp)
n_test = n_samp - n_train

x_data = np.zeros((n_samp, n_ant * n_pts))   # nengo dl only accepts 1D data type
y_data = np.zeros((n_samp, n_traj_pts * 2))   # <x1, y1> and <x2, y2>

with open("local_6_10000.txt", "r")  as in_file:
    data_lines = in_file.readlines()

for i in range(n_samp):
    line = data_lines[i]
    items = np.array(line.split(","))
    items = items.astype(np.float64)
    
    x_data[i, :] = items[0 : n_sig_pts]

    loc_start_x = items[n_sig_pts]
    loc_start_y = items[n_sig_pts + 1]
    loc_end_x = items[n_sig_pts + 2]
    loc_end_y = items[n_sig_pts + 3]

    y_data[i, :] = np.concatenate((np.linspace(loc_start_x, loc_end_x, n_traj_pts),
            np.linspace(loc_start_y, loc_end_y, n_traj_pts)))
    

x_data = np.concatenate((x_data, (-1) * x_data), axis=1)

# Divide the data into training group and test group
x_train = x_data[0:n_train, :] 
y_train = y_data[0:n_train, :]
x_test = x_data[n_train:, :]
y_test = y_data[n_train:, :]

# Make a negative copy of the input!!! --> (6, 180, 2)

#plt.figure()
#plt.plot(y_train[0, 0:2], y_train[0, 2:4])

#plt.figure()
#plt.plot(x_train[0, 0, :])
#plt.plot(x_train[0, 1, :])
#plt.plot(x_train[0, 2, :])
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
inp = tf.keras.Input(shape=(6, 180, 2), name="input")

to_spikes = layers.Activation(tf.nn.relu)(inp)

#to_spikes = layers.Conv2D(
#    filters=3,
#    kernel_size=1,
#    activation=tf.nn.relu,
#    use_bias=False,
#    name="to-spikes",
#)(inp)

# Convolutional Layer 0 
conv0 = layers.Conv2D(
    filters=64, 
    kernel_size=(1,16),
    activation=tf.nn.relu,
)(to_spikes)

# Max Pooling Layer 0
pool0 = layers.AveragePooling2D((1, 4))(conv0)

# Convolutional Layer 1 
conv1 = layers.Conv2D(
    filters=64,
    kernel_size=(3, 8),
    activation=tf.nn.relu,
)(pool0)

# Convolutional Layer 2 
conv2 = layers.Conv2D(
    filters=96,
    kernel_size=(4, 8),
    activation=tf.nn.relu,
)(conv1)

# Max Pooling Layer 1
pool1 = layers.AveragePooling2D((1, 2))(conv2)

# Flatten
flatten = layers.Flatten()(pool1)

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
do_training = True
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
    sample_neurons = np.linspace(
        0,
        np.prod(conv2.shape[1:]),
        200,
        endpoint=False,
        dtype=np.int32,
    )
    to_spikes_neurons = np.linspace(
        0,
        np.prod(to_spikes.shape[1:]),
        200,
        endpoint=False,
        dtype=np.int32,
    )
    with nengo_converter.net:
        conv2_probe = nengo.Probe(nengo_converter.layers[conv2][sample_neurons])
        to_spikes_probe = nengo.Probe(nengo_converter.layers[to_spikes][to_spikes_neurons])
        #out_probe = nengo.Probe(nengo_converter.outputs[outp], synapse=0.1, label="outp_filt")

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
    #loc_pred = data[outp_filt][:, -1]
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

        plt.subplot(1, 3, 2)
        scaled_to_spikes_data = data[to_spikes_probe][ii] * scale_firing_rates
        if isinstance(activation, nengo.SpikingRectifiedLinear):
            scaled_to_spikes_data *= 0.001
            rates = np.sum(scaled_to_spikes_data, axis=0) / (n_steps * nengo_sim.dt)
            plt.ylabel("Number of spikes")
        else:
            rates = scaled_to_spikes_data
            plt.ylabel("Firing rates (Hz)")
        plt.xlabel("Timestep")
        plt.title(
            f"to_spk mean={rates.mean():.1f} Hz, "
            f"max={rates.max():.1f} Hz"
        )
        plt.plot(scaled_to_spikes_data)

        plt.subplot(1, 3, 3)
        scaled_conv_data = data[conv2_probe][ii] * scale_firing_rates
        if isinstance(activation, nengo.SpikingRectifiedLinear):
            scaled_conv_data *= 0.001
            rates = np.sum(scaled_conv_data, axis=0) / (n_steps * nengo_sim.dt)
            plt.ylabel("Number of spikes")
        else:
            rates = scaled_conv_data
            plt.ylabel("Firing rates (Hz)")
        plt.xlabel("Timestep")
        plt.title(
            f"conv2 mean={rates.mean():.1f} Hz, "
            f"max={rates.max():.1f} Hz"
        )
        plt.plot(scaled_conv_data)
        plt.tight_layout()

#run_network(activation=nengo.RectifiedLinear(), n_steps=30) 
run_network(activation=nengo.SpikingRectifiedLinear(), n_steps=300, scale_firing_rates=300, synapse=0.05) 

plt.show()
