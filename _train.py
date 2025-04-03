import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# 从指定的模型文件中导入 LeNet5 模型
from models.LeNet_5 import LeNet5

def evaluate(model, data_loader, device):
    """计算模型在数据集上的准确率"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    return correct / total

def main():
    parser = argparse.ArgumentParser(description='Train LeNet-5 on MNIST using PyTorch')
    parser.add_argument('--continue_training', type=bool, default=False, help='Continue training from saved model')
    args = parser.parse_args()

    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 超参数设置
    batch_size = 100
    num_epochs = 10
    learning_rate = 0.01

    # 数据存储路径（确保已创建 ./dataset/_mnist 目录）
    data_root = "./dataset/_mnist"

    # 定义数据预处理流程（使用 MNIST 的均值和标准差）
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 加载 MNIST 数据集（如果数据不存在，会自动下载到 data_root 目录）
    train_dataset = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 创建模型并放置到设备上
    model = LeNet5(num_classes=10)
    model.to(device)

    # 模型保存目录设置
    saved_params_dir = "./saved_params"
    autosave_dir = os.path.join(saved_params_dir, "autosave")
    os.makedirs(autosave_dir, exist_ok=True)
    best_model_path = os.path.join(autosave_dir, "best_model_params.pth")

    # 若设置继续训练并且存在已保存模型，则加载模型参数
    if args.continue_training and os.path.exists(best_model_path):
        print("Loading saved model from:", best_model_path)
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    # 定义损失函数和优化器（此处使用 Adagrad）
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adagrad(model.parameters(), lr=learning_rate)

    # 用于记录训练过程中的指标
    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    best_loss = float('inf')

    # 训练循环（按 epoch 迭代）
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        num_batches = len(train_loader)

        # tqdm 进度条显示当前 epoch 的训练进度
        pbar = tqdm(enumerate(train_loader), total=num_batches, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (images, labels) in pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = running_loss / num_batches
        train_loss_list.append(avg_loss)

        # 计算训练和测试集准确率
        train_acc = evaluate(model, train_loader, device)
        test_acc = evaluate(model, test_loader, device)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

        # 保存每个 epoch 的模型参数
        epoch_model_path = os.path.join(autosave_dir, f"autosave_params_epoch{epoch+1}.pth")
        torch.save(model.state_dict(), epoch_model_path)

        # 保留最近 3 个 autosave 检查点
        autosave_files = sorted([f for f in os.listdir(autosave_dir) if f.startswith("autosave_params_")],
                                  key=lambda x: os.path.getctime(os.path.join(autosave_dir, x)))
        if len(autosave_files) > 3:
            for f in autosave_files[:-3]:
                os.remove(os.path.join(autosave_dir, f))

        # 保存表现最好的模型（基于平均损失）
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), best_model_path)
            print("Best model updated.")

    print("Training finished.")

    # 绘制训练过程中的准确率曲线
    epochs_range = range(1, num_epochs + 1)
    plt.figure()
    plt.plot(epochs_range, train_acc_list, label="Train Accuracy", marker='o')
    plt.plot(epochs_range, test_acc_list, label="Test Accuracy", marker='s')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    diagram_path = "./diagrams/diagram.png"
    os.makedirs(os.path.dirname(diagram_path), exist_ok=True)
    plt.savefig(diagram_path)
    plt.show()

if __name__ == '__main__':
    main()
