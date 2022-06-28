import numpy as np
import pandas as pd
import ActivationFunction as AF


class HiddenLayer():
    def __init__(self, n_in, n_out,activate='tanh' ):

        Activation = AF.ActivationFunction(mode = activate)
        self.f = Activation.f
        self.f_deriv = Activation.f_deriv

        #初始化权重和偏置
        self.W = np.random.uniform(low = np.sqrt(3. / (n_in + n_out)),
                                    high = np.sqrt(9. / (n_in + n_out)),
                                    size = (n_in,n_out))

        self.b = np.random.uniform(low = np.sqrt(0. / (n_in + n_out)),
                                    high = np.sqrt(6. / (n_in + n_out)),
                                    size = (n_out,))
        
        #将当前层W的导数和b的导数设置为0
        # self.grad_W = np.zeors(n_in,n_out)
        # self.grad_b = np.zeors(n_in,n_out)
        self.grad_W = np.zeors(self.W.shape)
        self.grad_b = np.zeors(self.b.shape)


    def forward(self, input):
        output = self.W@input
        output = self.f(output)
        return output

    def backward(self,output):
        pass



class MLP():
    def __init__(self, n_layers, n_h, hidden_activate = 'tanh'):
        self.hidden_activate = hidden_activate
        self.mode = 'train'

        self.layers = []
        for i in range(n_layers):
            self.layers.append(HiddenLayer(n_h[i], n_h[i+1], activate=self.hidden_activate))
        


    def train(self):
        self.mode = 'train'
    

    def forward_onelayer(self,input, layer):
        y = layer(input)
        return y


    def forward(self,input):
        for layer in self.layers:
            output = layer.forward(input)
            input = output
        return output

    def backward(self):
        pass