from PIL import Image
import numpy as np

def img2mnist(img_path, flatten=True):
    img = Image.open(img_path).convert('L')
    img = img.resize((28, 28))
    img = np.array(img)
    img = img.astype(np.float32)
    img = img / 255.0
    if flatten:
        img = img.reshape(1, 784)
        img = img.flatten()
    return img

if __name__ == '__main__':
    img_path = "../dataset/testPic"
    img_name = "7(1).jpg"
    full_path = img_path + "//" + img_name

    img = Image.open(full_path).convert('L')
    img = img.resize((28, 28))
    arr = np.array(img)
    min_val = np.percentile(arr, 5)
    max_val = np.percentile(arr, 95)

    arr_rescaled = np.clip((arr - min_val) / (max_val - min_val) * 255, 0, 255).astype(np.uint8)

    enhanced_img = Image.fromarray(arr_rescaled, mode='L')
    enhanced_img.save(full_path)
