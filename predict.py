import sys
import pickle
import argparse

import numpy as np
from torch import newaxis

from tools.pic2mnist import img2mnist
from models.LeNet_5 import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict number from image')
    parser.add_argument('--m', type=str, help='Path to the model', default=".//saved_params//autosave//best_model_params.pkl")
    parser.add_argument('--i', type=str, help='Path to the image', default=".//dataset//testPic//4.jpg")
    args = parser.parse_args()

    modelsSavePath = args.m

    with open(modelsSavePath, 'rb') as f:
        network_params = pickle.load(f)

    network = LeNet_5_M(input_dim=(1, 28, 28), output_size=10, weight_init_std=0.01, train_flg=False)
    network.params = network_params

    img_path = args.i
    img = img2mnist(img_path, flatten=False)
    img = img[newaxis, newaxis, :, :]

    y = network.predict(img)
    print(f"Predicted number is: {np.argmax(y)}")
