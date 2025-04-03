import argparse
import torch
from torchvision import transforms
from PIL import Image

# 从模型文件中导入 LeNet5
from models.LeNet_5 import LeNet5

def load_model(model_path, device):
    """
    加载保存的模型参数，并返回设置为评估模式的模型
    """
    model = LeNet5(num_classes=10)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def predict_image(model, image_path, device):
    """
    对指定图像进行处理，并使用模型预测
    """
    # 定义预处理流程：转为灰度图、调整为28×28、转换为Tensor并归一化
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 打开图像并应用预处理
    image = Image.open(image_path)
    image = transform(image)
    # 增加 batch 维度
    image = image.unsqueeze(0).to(device)

    # 进行预测
    with torch.no_grad():
        output = model(image)
        predicted = torch.argmax(output, dim=1)
    return predicted.item()

def main():
    parser = argparse.ArgumentParser(description='Predict number from image using PyTorch LeNet5')
    parser.add_argument('--m', type=str, default="./saved_params/autosave/best_model_params.pth",
                        help='Path to the saved model parameters')
    parser.add_argument('--i', type=str, default="./dataset/testPic/1.jpg",
                        help='Path to the image file')
    args = parser.parse_args()

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = load_model(args.m, device)

    # 进行预测
    pred = predict_image(model, args.i, device)
    print(f"Predicted number is: {pred}")

if __name__ == '__main__':
    main()
