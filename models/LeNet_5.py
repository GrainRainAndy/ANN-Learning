
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from collections import OrderedDict
from utils.layers import *


class LeNet_5: # Simple but Classic LeNet-5 model
    def __init__(self, input_dim=(1, 28, 28), output_size=10, weight_init_std=0.01):
        # C1: Convolutional Layer 1
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(6, input_dim[0], 5, 5)
        self.params['b1'] = np.zeros(6)

        # S2: Subsampling Layer 2
        # No parameters to initialize for pooling layers
        # Activated by Sigmoid

        # C3: Convolutional Layer 3
        self.params['W3'] = weight_init_std * np.random.randn(16, 6, 5, 5)
        self.params['b3'] = np.zeros(16)

        # S4: Subsampling Layer 4
        # No parameters to initialize for pooling layers
        # Activated by Sigmoid

        # C5: Convolutional Layer 5
        self.params['W5'] = weight_init_std * np.random.randn(120, 16, 5, 5)
        self.params['b5'] = np.zeros(120)

        # F6: Fully Connected Layer 6
        self.params['W6'] = weight_init_std * np.random.randn(120, 84)
        self.params['b6'] = np.zeros(84)
        # Activated by Sigmoid

        # Output Layer
        self.params['W7'] = weight_init_std * np.random.randn(84, output_size)
        self.params['b7'] = np.zeros(output_size)


        # Create layers
        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'], stride=1, pad=2)
        self.layers['BatchNorm1'] = BatchNormalization(gamma=1, beta=0)
        self.layers['Pool2'] = MeanPooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Sigmoid2'] = Sigmoid()
        self.layers['Conv3'] = Convolution(self.params['W3'], self.params['b3'], stride=1, pad=0)
        self.layers['BatchNorm3'] = BatchNormalization(gamma=1, beta=0)
        self.layers['Pool4'] = MeanPooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Sigmoid4'] = Sigmoid()
        self.layers['Conv5'] = Convolution(self.params['W5'], self.params['b5'], stride=1, pad=0)
        self.layers['BatchNorm5'] = BatchNormalization(gamma=1, beta=0)
        self.layers['Flatten'] = Flatten()
        self.layers['Affine6'] = Affine(self.params['W6'], self.params['b6'])
        self.layers['Sigmoid6'] = Sigmoid()
        self.layers['Affine7'] = Affine(self.params['W7'], self.params['b7'])

        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        date_loss = self.last_layer.forward(y, t)
        return date_loss

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['W1'], grads['b1'] = self.layers['Conv1'].dW, self.layers['Conv1'].db
        grads['W3'], grads['b3'] = self.layers['Conv3'].dW, self.layers['Conv3'].db
        grads['W5'], grads['b5'] = self.layers['Conv5'].dW, self.layers['Conv5'].db
        grads['W6'], grads['b6'] = self.layers['Affine6'].dW, self.layers['Affine6'].db
        grads['W7'], grads['b7'] = self.layers['Affine7'].dW, self.layers['Affine7'].db

        return grads


class LeNet_5_M: # Modern LeNet-5 model
    def __init__(self, input_dim=(1, 28, 28), output_size=10, weight_init_std=0.01, train_flg = True):
        # C1: Convolutional Layer 1
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(6, input_dim[0], 5, 5)
        self.params['b1'] = np.zeros(6)

        # S2: Subsampling Layer 2
        # No parameters to initialize for pooling layers
        # Activated by ReLU

        # C3: Convolutional Layer 3
        self.params['W3'] = weight_init_std * np.random.randn(16, 6, 5, 5)
        self.params['b3'] = np.zeros(16)

        # S4: Subsampling Layer 4
        # No parameters to initialize for pooling layers
        # Activated by ReLU

        # C5: Convolutional Layer 5
        self.params['W5'] = weight_init_std * np.random.randn(120, 16, 5, 5)
        self.params['b5'] = np.zeros(120)

        # F6: Fully Connected Layer 6
        self.params['W6'] = weight_init_std * np.random.randn(120, 84)
        self.params['b6'] = np.zeros(84)
        # Activated by Sigmoid

        # Output Layer
        self.params['W7'] = weight_init_std * np.random.randn(84, output_size)
        self.params['b7'] = np.zeros(output_size)


        # Create layers
        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'], stride=1, pad=2)
        self.layers['BatchNorm1'] = BatchNormalization(gamma=1, beta=0, train_flg=train_flg)
        self.layers['Pool2'] = MeanPooling(pool_h=2, pool_w=2, stride=2)
        self.layers['ReLU2'] = ReLU()
        self.layers['Conv3'] = Convolution(self.params['W3'], self.params['b3'], stride=1, pad=0)
        self.layers['BatchNorm3'] = BatchNormalization(gamma=1, beta=0, train_flg=train_flg)
        self.layers['Pool4'] = MeanPooling(pool_h=2, pool_w=2, stride=2)
        self.layers['ReLU4'] = ReLU()
        self.layers['Conv5'] = Convolution(self.params['W5'], self.params['b5'], stride=1, pad=0)
        self.layers['BatchNorm5'] = BatchNormalization(gamma=1, beta=0, train_flg=train_flg)
        self.layers['Flatten'] = Flatten()
        self.layers['Affine6'] = Affine(self.params['W6'], self.params['b6'])
        self.layers['Sigmoid6'] = Sigmoid()
        self.layers['Affine7'] = Affine(self.params['W7'], self.params['b7'])

        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        date_loss = self.last_layer.forward(y, t)
        return date_loss

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['W1'], grads['b1'] = self.layers['Conv1'].dW, self.layers['Conv1'].db
        grads['W3'], grads['b3'] = self.layers['Conv3'].dW, self.layers['Conv3'].db
        grads['W5'], grads['b5'] = self.layers['Conv5'].dW, self.layers['Conv5'].db
        grads['W6'], grads['b6'] = self.layers['Affine6'].dW, self.layers['Affine6'].db
        grads['W7'], grads['b7'] = self.layers['Affine7'].dW, self.layers['Affine7'].db

        return grads


class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x