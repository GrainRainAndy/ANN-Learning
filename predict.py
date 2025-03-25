import numpy as np
import pickle
from tools.pic2mnist import img2mnist

if __name__ == '__main__':
    modelsSavePath = ".//saved_params//autosave//best_model.pkl"

    (network, loss) = pickle.load(open(modelsSavePath, 'rb'))

    img_path = ".//dataset//testPic//4.jpg"
    img = img2mnist(img_path)

    y = network.predict(img)
    print(f"Predicted number is: {np.argmax(y)}")
