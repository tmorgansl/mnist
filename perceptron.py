# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 14:31:10 2019

@author: tamor
"""

import numpy as np

class Perceptron(object):
    def __init__(self, input_size, epochs=100):
        self.weight = np.zeros(input_size)
        self.bias = np.zeros(1)
        self.epochs = epochs
    
    @staticmethod
    def activation_fn(z):
        if z >= 0:
            return 1.
        return 0.
    
    def predict(self, x):
        # forward propagation
        z = self.weight.T.dot(x) + self.bias
        return self.activation_fn(z)
    
    def fit(self, x, y):
        for _ in range(self.epochs):
            for i in range(y.shape[0]):
                # full forward pass
                a = self.predict(x[i])
                error = y[i] - a
                self.weight = self.weight + (error * x[i])
                self.bias += error
                # compute error
                # update weight and bias
                
    
if __name__ == '__main__':
    x = np.array([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]])
    
    y = np.array([0, 0, 0, 1])
    
    perceptron = Perceptron(input_size=2)
    
    perceptron.fit(x, y)
    
    accuracy = 0.
    for i in range(y.shape[0]):
        accuracy += perceptron.predict(x[i]) == y[i]
    accuracy /= y.shape[0]
    
    print("{:.4}".format(accuracy))
    
    print(perceptron.weight)
    print(perceptron.bias)