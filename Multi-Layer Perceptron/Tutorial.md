## 深度学习领域的 "Hello World" —— 基于多层感知机(MLP)的MNIST手写数字识别

我做这个**Deep Learning for Beginners**项目的初衷是给一位学管理学的好友一些深度学习上的参考，加上我在学习的时候也踩了一些坑，所以我准备做一个系列大概5-6篇文章，带一些代码水平比较差的甚至是管理学这种偏文科的专业的同学入门深度学习，作者本人水平其实也很有限，写这篇文章的时候只是一个大一的学生，我觉得可能从我的角度去讲可能更好理解一点，我认为既然是新手入门我们就不要过度关注原理，并不是说原理不重要，但是一开始就从数学的角度看原理，一个是很枯燥，另一个是太拖进度，所以我更倾向于这一轮结束我们再去补原理。不过在未来的大概半年到一年的时间里我会推出一个系列，通过Numpy从零手搓一个类似Pytorch的玩具级的深度学习框架，细致到每一步的数学推导的，从底层的数学公式去理解神经网络的内部推演运算，不过我还在补很多底层的东西，敬请期待吧……

**下面开始我们的第一个项目吧！基于MLP的MNIST手写数字识别**

首先先来简单介绍一下**MNIST**数据集，全称是**Modified National Institute of Standards and Technology**，是机器学习领域最经典的入门级图像数据集了，包含了60000张训练图像和10000张测试图像，每张图像都是28x28的灰度图像，像素值在0到255之间，标签是0到9的数字，代表了图像中手写的数字。

对于初学者我们可以把整个过程拆分为以下几个步骤：
```mermaid
flowchart LR
    A(数据工程) --> B(模型设计)
    B --> C(训练与调参)
    C --> D(模型评估)
```

**那么就不多废话了直接进入代码部分！**

首先来导入必要的包：
```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
```

然后就是数据工程了，其实对于这个项目来说很简单，就是下载数据集，等后续到我们自己的数据集了我们可能就需要进行一些处理，然后再划分训练集和测试集。
```python
# 下载训练集
training_data = datasets.MNIST(
    root="Data",             # 数据集存放的路径
    train=True,              # 是否为训练集
    download=True,           # 是否下载
    transform=ToTensor(),    # 数据转换
)

# 下载测试集
test_data = datasets.MNIST(
    root="Data",
    train=False,
    download=True, 
    transform=ToTensor(),
)
```
这段代码会通过`datasets.MNIST()`函数下载我们的数据集，然后在当前的文件夹下创建一个Data文件夹，里面有一个MNIST文件夹，这里就是我们的数据集了。

然后再通过`DataLoader()`函数将数据集转换为可迭代的对象，方便后续的训练和测试。
```python
train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)
```

数据集准备好了下面我们就可以开始搭建神经网络了!
```python
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
```
基于**Pytorch**框架的神经网络都是继承自`nn.Module`类，所以我们在定义神经网络的时候需要继承这个类，然后实现`__init__()`和`forward()`方法，`__init__()`方法是初始化神经网络的结构，`forward()`方法是实现前向传播的过程。

`nn.Flatten()`可以将输入的图像从`(batch_size, 1, 28, 28)`的形状转换为`(batch_size, 28*28)`的形状，因为我们的全连接层只能接受一维的输入。

`nn.Linear()`是全连接层，接受两个参数，第一个参数是输入的维度，第二个参数是输出的维度。

现在使用`784→512→256→128→10`的结构，有助于特征的逐步抽象和压缩。

`torch.relu()`是激活函数，这里我们使用的是ReLU激活函数，当然你也可以使用其他的激活函数，比如Sigmoid，Tanh等。

然后我们就可以开始写训练部分了，我一般喜欢写成一个函数然后在`if __name__ == '__main__':`中调用:
```python
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
```
训练的过程就是不断地迭代数据集，然后计算损失，反向传播，更新参数，直到模型收敛的过程，我们只需要记住这个过程即可，其他的细节我们可以根据需要进行调整。


接下来我们来写测试部分：
```python
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
```
在测试部分我们需要关闭梯度计算，因为测试的时候我们不需要更新参数，只需要计算损失和正确率即可。

最后我们来写主函数：
```python
if __name__ == '__main__':
    model = MLP()
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    epochs = 25
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
```
我们可以通过绘制损失函数来查看模型的训练情况，损失函数应该随着训练的进行而下降，说明模型的训练是正常的。

使用不同的优化器效果可能也有差异，SGD优化器的效果可能会比Adam优化器差一些，因为Adam优化器是基于动量和自适应学习率的优化器，而SGD优化器是基于梯度下降的优化器，所以在一些情况下，Adam优化器的效果会更好，就这个项目而言SGD的争取率在88%左右，Adam优化器的正确率在97%以上。


以上就是基于多层感知机的简单的分类任务，下面我会附上完整的代码，只需要修改"D:\Deep Learning for Beginners\Multi-Layer Perceptron\model\model.pth"这个模型保存路径即可直接运行。

完整代码：
```python
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
            print(f"Batch {batch_size_num} Loss: {loss_value:.4f}")
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
    
    epochs = 25
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
```

如果你能看到这里，相信你对我的纯代码向的讲解是可以接受的，我预计会继续写4-5篇文章，包基于**CNN**的MNIST手写数字识别，当前多层感知机的准确率大概在97%左右，CNN可以达到99%以上；**RNN**现在 的使用场景已经很少了我目前也没有找到比较合适的项目应该就不做了；我最熟悉的**LSTM/GRU**我会找一个股票预测的数据集带着各位同学做一个简单的时序预测项目，本质上LSTM/GRU就是RNN的变体，解决RNN的梯度消失/梯度爆炸的问题；**Transformer**这块我还在学习推导这一块，目前也没想好；**GCN**的项目我已经写完了，Cora数据集是一个包含2708个节点的论文引用网络，我们通过GCN仅靠图结构或特征把节点分成若干簇（GCN其实才是我的好友需要学习的内容，但是我想了一下图卷积神经网络，又是图又是卷积的，还是需要MLP和CNN打一个基础的）；最后就是**GAN生成对抗网络**，这个网络比较难我学的也比较浅，所以没想好做不做，但是用的也比较少，基本上常见的会用到的深度学习算法就这么多，这个系列大概是周更的样子，我尽量在一到两个月内更完。

大家也可以给我反馈一下，想不想看mlp的一个简单的公式推导，我也可以简单写一篇，我保证我讲的非常的通俗，基本上只要学过高数和线代都可以听懂，非常非常的通俗易懂，想要对深度学习有更进一步研究的我觉得这是必修课，一定要学好微积分和线代，对于深度学习来说是很重要的，所有的深度学习的参数更新都是通过微积分来实现的，所有的优化算法都是通过线代来实现的，所以学好这两门课对于深度学习来说是非常重要的。

大家想看一些什么教程也可以留言，我除了深度学习以外，传统的机器学习和强化学习也是我的方向之一，传统的机器学习的可解释性会更强，但是现在的使用场景也在慢慢变少，强化学习今年也开始火起来了，不过如果你连深度学习都没搞利索就不要想着涉猎强化学习了，会更加的复杂。

通过这个项目你可以在半个小时之内跑出自己的第一个深度学习模型，深度学习是很神奇的，尤其是从背后的原理出发，你会发现很多东西都是很简单的，但是你却可以通过这些简单的东西来实现很多复杂的事情，这也是深度学习的魅力所在。