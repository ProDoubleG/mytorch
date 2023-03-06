import numpy
import mytorch.nn as nn

class flatten(nn.Model.Layer):
    def __init__(self, input, output_dim=1):
        
        self.backward_output = None
        self.backward_input = None

        self.output_dim = output_dim

        self.__call__(input)

    def forward(self):

        if nn.Model.APPEND_LAYER:
            nn.Model.LAYER_TEST.append(self)

        self.forward_output = self.forward_input.reshape(self.output_dim,-1)

    def backpropagate(self):
        # print("flatten_backprop_input :",self.backward_input.shape)
        self.backward_output = self.backward_input.reshape(self.forward_input.shape)
        # print("flatten_backprop_output :",self.backward_output.shape)
        # assert False