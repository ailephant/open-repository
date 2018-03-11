# Updated code of Milo Spencer-Harper for Python 3.6
# https://medium.com/technology-invention-and-more/how-to-build-a-simple-neural-network-in-9-lines-of-python-code-cc8f23647ca1

import numpy as np

training_set_inputs = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])

training_set_outputs = np.array([[0, 1, 1, 0]]).T

np.random.seed(1)

synaptic_weights = 2 * np.random.random((3, 1)) - 1

for iteration in range(10000):

    output = 1 / (1 + np.exp(-(np.dot(training_set_inputs, synaptic_weights))))

    synaptic_weights += np.dot(training_set_inputs.T, (training_set_outputs - output) * output * (1 - output))

print ( 1 / (1 + np.exp(-(np.dot(np.array([1, 0, 0]), synaptic_weights)))))