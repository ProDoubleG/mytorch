import numpy
import mytorch.nn as nn

class Softmax(nn.Model.Activation):

    def __init__(self, dim=None):

        self.output_dim = dim
        self.__call__(input)

    def __init__(self, input, dim=None):

        self.output_dim = dim
        self.forward_input = input.forward_output
        self.__call__(input)

    def forward(self):

        if nn.Model.APPEND_LAYER:
            nn.Model.LAYER_TEST.append(self)
        
        h = numpy.max(self.forward_input)
        exp = numpy.exp(h - self.forward_input)
        sum_exp = numpy.sum(exp)
        self.forward_output = exp/sum_exp

        # print("softmax_sum :", numpy.sum(self.forward_output))
        
        if self.output_dim != None:
            assert len(numpy.array(self.forward_output)) == self.output_dim

    def backpropagate(self):
        
        self.backward_output = (1 + self.backward_input)*self.forward_output
        # print("softmax :",self.backward_output)
        
    # self.forward_output = nn.Model.MODEL_OUTPUT
        
    # nn.Model.stream_tape['now'] = self.forward_output
    # def __call__(self):
    #     return self.forward_output