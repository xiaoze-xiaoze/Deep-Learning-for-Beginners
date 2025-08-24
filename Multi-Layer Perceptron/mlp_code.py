import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

# 下载训练集
training_data = datasets.MNIST(
    root="Data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# 下载测试集
test_data = datasets.MNIST(
    root="Data",
    train=False,
    download=True,
    transform=ToTensor(),
)

# 将数据集转换为可迭代的对象
train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 定义神经网络
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

train_loss_history = []

def train(model, dataloader, loss_function, optimizer):
    model.train()    # 设置为训练模式
    epoch_loss = 0
    batch_size_num = 1
    
    for x, y in dataloader:
        pred = model.forward(x)          # 前向传播
        loss = loss_function(pred, y)    # 计算损失
        optimizer.zero_grad()            # 梯度清零
        loss.backward()                  # 反向传播
        optimizer.step()                 # 更新参数
        
        loss_value = loss.item()         # 获取损失值
        epoch_loss += loss_value         # 累加损失值
        if batch_size_num % 100 == 0:
            print(f"Batch {batch_size_num}  Loss: {loss_value:.4f}")
        batch_size_num += 1
    
    avg_loss = epoch_loss / len(dataloader)
    train_loss_history.append(avg_loss)

test_loss_history = []

def test(model, dataloader, loss_function):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()    # 设置为评估模式
    
    test_loss, correct = 0,0
    with torch.no_grad():    # 关闭梯度计算
        for x,y in dataloader:
            pred = model.forward(x)                # 前向传播
            test_loss += loss_function(pred,y).item()    # 累加损失值
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()    # 累加正确率

    test_loss /= num_batches
    correct /= size
    test_loss_history.append(test_loss)
    print(f"\nTest:")
    print(f"Accuracy: {(100*correct)}%")
    print(f"Avg loss: {test_loss}")

if __name__ == '__main__':
    model = MLP()
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    epochs = 30
    for i in range(epochs):
        print(f"\nEpoch {i+1}")
        train(model, train_dataloader, loss_function, optimizer)
        torch.save(model.state_dict(), "D:\Deep Learning for Beginners\Multi-Layer Perceptron\model\model.pth")

    # 绘制损失函数
    plt.figure(figsize=(12,5))
    plt.plot(train_loss_history, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()

    test(model, test_dataloader, loss_function)