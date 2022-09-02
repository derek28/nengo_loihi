# nengo_loihi
Using Nengo framework for SNN simulation.

Dataset (from simulation):
gesture_8_10000.txt  8 gestures, with signals from 3 antennas, totally 10000 samples. Each sample 6x180. 
local_6_10000.txt  Localization with signals from 6 antennas. 10000 smaples. Eacg sample 3x250.

local_cnn.py  Using CNN for localization, then convert into SNN. Performance degrade. Not neuromorphic. 
local_snn.py  Directly building SNN model. Data input 6 x n_step. 
gesture_snn.py  Directly building SNN model. Data input 3 x n_step. 
mnist_loihi.py  Copied from nengo_loihi website for reference. 
