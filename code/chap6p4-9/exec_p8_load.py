import network3p8
from network3p8 import Network

retrieved = Network.load("p8_saved.save")

print("weight between the first neuron in the max-pooling layer "
      "and the first neuron in the fully-connected layer: {}".format(
              retrieved.params[2][0][0].eval()))