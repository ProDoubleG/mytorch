import numpy

class Model:
    APPEND_LAYER = False
    LAYER_TEST =  list()
    STEAM_TAPE = dict()
    GRADIENT_TAPE = dict()
    NUMBER_OF_STREAM = 1
    MODEL_INPUT = None

    """Super Model Class"""
    def __init__(self):
        # self.model_output = None
        self.model_input = None
        self.model_depth = 0
        # self.stream_tape = dict()
        # self.gradient_tape = dict()

    def forward(self):
        ...

    def set_layer_list(self):
        
        Model.APPEND_LAYER = True
        self.forward(Model.MODEL_INPUT[0])
        Model.APPEND_LAYER = False

    def __call__(self,model_input):

        Model.MODEL_INPUT = model_input
        
        self.set_layer_list()

        for idx in range(len(model_input)):
            
            # print(self.forward(model_input[idx]).forward_output.shape)
            try:
                self.model_output = numpy.concatenate((self.model_output, self.forward(model_input[idx]).forward_output),axis=0)
            except AttributeError as e:
                self.model_output = self.forward(model_input[idx]).forward_output
        
        print("model output shape : ",self.model_output.shape)
        return self.model_output

    class Layer:
        """Super Layer Class"""
        def __init__(self):
            self.name = 'layer'
            self.layer_weight = None

            self.forward_input = None
            self.forward_output = None

            self.backward_input = None
            self.backward_output = None
            
            self.gradient = None

        def forward(self):
            ...

        def append(self):

            Model.LAYER_TEST.append(self)

        def __call__(self, input):

            try:
                self.forward_input=input.forward_output
            except AttributeError:
                self.forward_input = input
            
            self.forward()


            return self

    class Activation:
        """Super Layer Class"""
        def __init__(self):
            self.forward_input = None
            self.forward_output = None

            self.backward_input = None
            self.backward_output = None

        def forward(self):
            ...

        def append(self):

            Model.LAYER_TEST.append(self)

        def __call__(self, input):
            try:
                self.forward_input=input.forward_output
            except AttributeError:
                self.forward_input = input
            
            self.forward()

            return self

    class Loss():
        
        def __init__(self):
            self.prediction = None
            self.label = None

            self.backward_input = None
            self.backward_output = None

            self.loss_output = None

        def __call__(self, prediction, label):
            self.prediction = prediction
            self.label = label
            self.forward()

            return self

        def backward(self):
            print("forward_done",Model.MODEL_INPUT.shape)
            #TODO: LAYER 목록 있으니까
            # LAYER 목록 따라서 forward 내려가고
            # prediction 나오면 LAYER 목록 역순으로 backprop 따고
            # Trainable Layer들에 대해서 gradient 를 리스트로 박자

            for layer in Model.LAYER_TEST:
                layer.forward
            for idx in range(len(Model.MODEL_INPUT)):
                
                forward_out = Model.forward(Model.MODEL_INPUT[idx]).forward_output
                print(forward_out.shape)
                assert False
                # backward_in = self.backward_output
                backward_in = forward_out
                layer_class_list = Model.LAYER_TEST[::-1]
                for layer_class in layer_class_list:
                    layer_class.backward_input = backward_in
                    layer_class.backpropagate()
                    backward_in = layer_class.backward_output

class Linear(Model.Layer):

    def __init__(self,input_size, output_size):
        super().__init__()
        self.layer_weight = numpy.random.randn(input_size, output_size)
        self.name = 'dense_layer'
        
    def forward(self):

        if Model.APPEND_LAYER:
            Model.LAYER_TEST.append(self)

        self.forward_output = numpy.matmul(self.forward_input, self.layer_weight)
        return self.forward_output

    def backpropagate(self):
        self.gradient = numpy.matmul(self.forward_input.T,self.backward_input)
        self.backward_output = numpy.matmul(self.backward_input,self.layer_weight.T)
        
        print("linear backprop: ",self.backward_output.shape)

class Conv2d(Model.Layer):

    def __init__(self, output_channels, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.output_channels = output_channels
        self.stride = stride
        self.layer_weight = None

    def forward(self):

        # try:
        #     assert len(self.forward_input.shape) != 4
        # except AssertionError as e:
        #     print(f"input shape for conv2d layer must be in 4 dimension, (n, w, h, c)")

        if Model.APPEND_LAYER:
            Model.LAYER_TEST.append(self)

        try:
            if (self.forward_input.shape[0]-self.kernel_size)%self.stride != 0 or (self.forward_input.shape[1]-self.kernel_size)%self.stride != 0:
                raise Exception(f"Cannot clean-cut input {self.forward_input.shape} with given Kernel size and stride ")
        except Exception as e:
            print("Error in Convolution Layer!")
            print(e)

        input_x = self.forward_input.shape[0] # input x
        input_y = self.forward_input.shape[1] # input y
        input_z = self.forward_input.shape[2] # input channel

        if self.layer_weight is None:
            self.layer_weight = numpy.random.randn(self.output_channels, input_z, self.kernel_size, self.kernel_size)

        # (number_of)filter, channel, x, y) # 5 1 4 4

        self.transposed_forward_input = self.forward_input.transpose(2,0,1)

        receptive_field_num_x = (input_x - self.kernel_size)//self.stride+1
        receptive_field_num_y = (input_y - self.kernel_size)//self.stride+1

        self.forward_output = numpy.zeros((self.output_channels,receptive_field_num_x, receptive_field_num_y))

        for nth_filter in range(self.output_channels): # 필터 개수로 iter
            for nth_channel in range(input_z): # 채널별로 iter
                for _x in range(receptive_field_num_x):
                    for _y in range(receptive_field_num_y):
                        convolution_kernel = self.layer_weight[nth_filter][nth_channel]*self.transposed_forward_input[nth_channel][_x*self.stride:_x*self.stride+self.kernel_size,_y*self.stride:_y*self.stride+self.kernel_size]
                        convolution_sum = numpy.sum(convolution_kernel)
                        self.forward_output[nth_filter][_x][_y] = convolution_sum
        
        return self.forward_output

class CrossEntropyLoss(Model.Loss):

        def forward(self):

            self.loss_output = -1*numpy.sum(self.label*numpy.log(self.prediction))
            self.backward_output = -1*self.label/(self.prediction)