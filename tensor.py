import matplotlib.pyplot as plt
import numpy as np
X = np.array([2.1, 2.2, -1.8])
X = np.random.randn(3, 4)
class dense_layer:
    def __init__(self, inputs, neurons):
        self.weight = 0.2 * np.random.randn(inputs, neurons)
        self.bias = 0.3 * np.random.randn(1, neurons)

    def forward(self, inputdata):
        self.output = np.dot(inputdata, self.weight) + self.bias
        return self.output

Layer1 = dense_layer(4, 5)
Layer1.forward(X)
Layer2 = dense_layer(5, 2)
Layer2.forward(Layer1.output)
print('The dense_layer_output:\n', Layer1.output)
print('ANN_output:\n', Layer2.output)
X = np.load('D:/X.npy')
# print('The signs digt data set:\n', X)
print('X250:\n',X[250])
'''plt.imshow(X[250])
plt.show()
print('X250 shape:\n', X[250].shape)
print('X250 dimension:\n', X[250].ndim)'''
class Activation_Relu:
    def forward(self,inputs):
        self.output=np.maximum(0,inputs)
Activation1 = Activation_Relu()
Activation1.forward(Layer2.output)
# print('The out_Activation is:\n', Activation1.output)
class stepup_activation:
    def forward(self,inputs):
        self.output = np.heaviside(inputs, 0)
Activation2 = stepup_activation()
Activation2.forward(Layer2.output)
print('The stepup_activation_out is:\n', Activation2.output)
print('Out_activation:\n', Activation1.output)
class softmax_activation:
    def forward(self,inputs):
        exp_values = np.exp(inputs)
        # self.exp_values_aftermv = exp_values-max(exp_values)
        exp_values_aftermv = exp_values-np.max(exp_values)
        self.output_overflormvd = exp_values_aftermv/np.sum(exp_values_aftermv)
        exp_values_total = np.sum(exp_values)
        # self.probabilities = exp_values/exp_values_total
        # probabilities = exp_values/exp_values_total
        probabilities = exp_values_aftermv/np.sum(exp_values_aftermv)
        self.output = probabilities
        # return probabilities
Activation3 = softmax_activation()
Activation3.forward(Layer2.output)
# print('The softmax_Act3_out is:\n',Activation3.output)
print('Overflow:\n', Activation3.output_overflormvd)
