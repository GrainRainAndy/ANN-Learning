import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))

def relu(x):
    return np.maximum(0, x)

def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    for idx in range(x.size):
        temp_x = x[idx]
        x[idx] = temp_x + h
        fxh1 = f(x) # get f(x+h)
        x[idx] = temp_x - h
        fxh2 = f(x) # get f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = temp_x
    return grad

def gradient_descent(f, init_x, lr = 0.01, step_num = 1000):
    x = init_x
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    return x

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(-1, t.size)
        y = y.reshape(-1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size