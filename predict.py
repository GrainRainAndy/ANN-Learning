import sys
import pickle
import argparse

import numpy as np

from tools.pic2mnist import img2mnist

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict number from image')
    parser.add_argument('--m', type=str, help='Path to the model', default=".//saved_params//autosave//best_model.pkl")
    parser.add_argument('--i', type=str, help='Path to the image', default=".//dataset//testPic//4.jpg")
    args = parser.parse_args()

    modelsSavePath = args.m

    (network, loss) = pickle.load(open(modelsSavePath, 'rb'))

    img_path = args.i
    img = img2mnist(img_path)

    y = network.predict(img)
    print(f"Predicted number is: {np.argmax(y)}")
