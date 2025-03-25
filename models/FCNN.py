# Full Connected Neural Network

import numpy as np

from collections import OrderedDict
from utils.layers import *
from utils.math_func import numerical_gradient


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01, reg_lambda=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['ReLU'] = ReLU()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        self.lastLayer = SoftmaxWithLoss()

        self.reg_lambda = reg_lambda


    def predict(self, x): # x: image data
        for layer in self.layers.values():
            x = layer.forward(x)

        return x


    def loss(self, x, t): # x: image data; t: labels
        y = self.predict(x)
        data_loss = self.lastLayer.forward(y, t)
        reg_loss = (0.5 * self.reg_lambda *
                    (np.sum(self.params['W1'] ** 2) + np.sum(self.params['W2'] ** 2)))
        return data_loss + reg_loss


    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = t.argmax(axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy


    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        grads = {'W1': numerical_gradient(loss_W, self.params['W1']),
                 'W2': numerical_gradient(loss_W, self.params['W2']),
                 'b1': numerical_gradient(loss_W, self.params['b1']),
                 'b2': numerical_gradient(loss_W, self.params['b2'])}
        return grads


    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        #backward
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        # Introduce regularization
        grads = {'W1': self.layers['Affine1'].dW + self.reg_lambda * self.params['W1'],
                 'W2': self.layers['Affine2'].dW + self.reg_lambda * self.params['W2'],
                 'b1': self.layers['Affine1'].db,
                 'b2': self.layers['Affine2'].db}
        return grads
'''
将layers保存为OrderedDict这一点非常重要：
正向传播只需要按照添加元素的顺序调用各层的forward()方法即可完成处理，反向传播只需要按照相反顺序调用各层。
'''

class TwoHiddenLayersNet:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, weight_init_std = 0.01, reg_lambda=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size1)
        self.params['b1'] = np.zeros(hidden_size1)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size1, hidden_size2)
        self.params['b2'] = np.zeros(hidden_size2)
        self.params['W3'] = weight_init_std * np.random.randn(hidden_size2, output_size)
        self.params['b3'] = np.zeros(output_size)

        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['ReLU1'] = ReLU()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['ReLU2'] = ReLU()
        self.layers['Affine3'] = Affine(self.params['W3'], self.params['b3'])

        self.lastLayer = SoftmaxWithLoss()

        self.reg_lambda = reg_lambda


    def predict(self, x): # x: image data
        for layer in self.layers.values():
            x = layer.forward(x)

        return x


    def loss(self, x, t): # x: image data; t: labels
        y = self.predict(x)
        data_loss = self.lastLayer.forward(y, t)
        reg_loss = (0.5 * self.reg_lambda *
                    (np.sum(self.params['W1'] ** 2) + np.sum(self.params['W2'] ** 2) + np.sum(self.params['W3'] ** 2)))
        return data_loss + reg_loss


    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = t.argmax(axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy


    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        grads = {'W1': numerical_gradient(loss_W, self.params['W1']),
                 'W2': numerical_gradient(loss_W, self.params['W2']),
                 'b1': numerical_gradient(loss_W, self.params['b1']),
                 'b2': numerical_gradient(loss_W, self.params['b2']),
                 'W3': numerical_gradient(loss_W, self.params['W3']),
                 'b3': numerical_gradient(loss_W, self.params['b3'])}
        return grads


    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        #backward
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        # Introduce regularization
        grads = {'W1': self.layers['Affine1'].dW + self.reg_lambda * self.params['W1'],
                 'b1': self.layers['Affine1'].db,
                 'W2': self.layers['Affine2'].dW + self.reg_lambda * self.params['W2'],
                 'b2': self.layers['Affine2'].db,
                 'W3': self.layers['Affine3'].dW + self.reg_lambda * self.params['W3'],
                 'b3': self.layers['Affine3'].db}
        return grads