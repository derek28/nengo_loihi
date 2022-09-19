import matplotlib.pyplot as plt
import nengo
from nengo.utils.filter_design import cont2discrete
from nengo.utils.matplotlib import rasterplot
import numpy as np
import tensorflow as tf
import nengo_dl
import os

# set seed to ensure this example is reproducible
seed = 10
# tf.random.set_seed(seed)
# np.random.seed(seed)
rng = np.random.RandomState(seed)

n_samp = 1200
n_pts = 500
n_chan = 6
n_label = 12
t_sim = 1.5
t_range = np.linspace(0, t_sim, n_pts)

n_signal = n_pts * n_chan

n_train = int(0.8*n_samp)
n_test = n_samp - n_train

x_data = np.zeros((n_samp,n_signal))
y_data = np.zeros((n_samp))

# process the raw data
with open("gesture_real_12_1200.txt", "r")  as in_file:
    data_lines = in_file.readlines()

for i in range(n_samp):
    line = data_lines[i]
    items = np.array(line.split(" "))
    items = items.astype(int)
    
    x_data[i,:] = items[:n_signal]
    y_data[i] = items[n_signal]

# first random permutation
perm = rng.permutation(x_data.shape[0])
x_data = x_data[perm,:]
y_data = y_data[perm]

# divide the dataset into training and testing group
x_train = x_data[:n_train,:]
y_train = y_data[:n_train]
x_test = x_data[n_train:,:]
y_test = y_data[n_train:]

# x_train = x_train.reshape((x_train.shape[0],n_chan,n_pts)).swapaxes(1,2)
x_train = x_train.reshape((x_train.shape[0],n_pts,n_chan))
plt.figure()
rasterplot(t_range, x_train[0])
plt.figure()
rasterplot(t_range, x_train[1])
plt.show()