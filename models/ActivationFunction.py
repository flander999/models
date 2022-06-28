from math import exp
import numpy as np


class ActivationFunction():
    def __init__(self, mode = 'tanh'):
        if mode == 'tanh':
            self.f = self._tanh
            self.f_deriv = self._deriv_tanh
        elif mode == 'sigmoid':
            self.f = self._sigmoid
            self.f_deriv = self._deriv_sigmoid
        elif mode == 'relu':
            self.f = self._relu
            self.f_deriv = self._deriv_relu
        elif mode == 'relu':
            self.f = self._Leaky_relu
            self.f_deriv = self._deriv_leaky_relu 


    def _tanh(x):
        above = exp(x) - exp(-x)
        bottom = exp(x) + exp(-x)
        return above/bottom


    def _sigmoid(x):
        return 1/(1-exp(-x))


    def _relu(x):
        return x if x > 0 else 0


    # dealing with dead relu problem, alpha should be really small and positive
    def _Leaky_relu(x):
        return x if x > 0 else 0.001


    def _deriv_tanh(x):
        return 1-x^2


    def _deriv_sigmoid(x):
        return x(1-x)

    def _deriv_relu(x):
        return 1 if x != 0 else 0

    def _deriv_leaky_relu(x):
        return 1 if x != 0.001 else 0
