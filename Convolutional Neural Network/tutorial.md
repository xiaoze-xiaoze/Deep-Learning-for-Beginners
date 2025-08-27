## 基于卷积神经网络(CNN)的MNIST手写数字识别

相信大家在看完前两篇文章之后已经对神经网络是怎么一回事有些了解了，但是我们发现多层感知机虽然什么都能做一点吧但是效果并没有很理想，MNIST数据集的识别率也只有97%而到不了99%甚至100%，引用 **《动手学深度学习》** 上的一段话就是 **"多层感知机十分适合处理表格数据，其中行对应样本，列对应特征，对于表格数据我们寻找的模式可能涉及特征之间的交互，但是我们不能预先假设任何与特征交互相关的先验结构，此时多层感知机可能是最好的选择，然而对于高维感知数据，这种缺少结构的网络可能会变得不实用。"** 所以为了图像识别任务而设计了卷积神经网络，当然现在卷积神经网络已经不只应用于图像识别任务了。

试想一下，一张照片具有百万级的像素，这就意味着网络有一百万个输入，即便隐藏层不断降低维度，参数量也会直接爆炸，这时候再用多层感知机显然是不合适的，卷积神经网络就是为了解决这个问题而设计的。
<br>
在正式开始学习CNN之前，我们应该先了解一下图像。图像是由像素组成的，每个像素都有自己的颜色，以MNIST的灰度图为例，每个像素的颜色值范围是0-255，0表示黑色，255表示白色，中间的灰度值表示不同的灰色，可以用一个矩阵表示，换成彩色图片也是同理，由3个RGB数值来描述一种颜色，可以用三个矩阵表示，其中每一个矩阵又叫一个channel(通道)，**这里一定一定要好好理解，我当时学的时候就是没有理解灰度图导致在卷积部分卡了好久！！！** 
<br>
在学习卷积层之前我要先引入两个概念，分别是 **平移不变性和局部性**。
**平移不变性**就像你训练了一只小狗认球，你在客厅左边扔一个球，它知道这是球，你在客厅右边甚至卧室扔出这个球，它还是会认出来。反映到图像上就是，你想要识别的内容不会因为位置的改变而改变。
**局部性**则可以用生活常识来解释，我们在识别物品的时候往往不需要一次性看完整张图片就可以识别出来，比如我们要分辨是猫是狗，我们只需要看到鼻子或者耳朵就能识别出来，这就体现了局部性。
到这里大家不妨停下来好好理解一下，理解的透彻一点，因为卷积这个操作就是通过上面两个概念设计出来的。

现在我们开始学习**卷积层(Convolutional Layer)**，卷积这个概念最早其实源于数学的卷积积分，公式如下：
$$ (f * g)(x) = \int_{-\infty}^{\infty} f(z) g(x - z) dz $$
但是严格意义上，卷积层做的并非数学上的卷积积分，而是互相关计算，当然这对初学者来说其实并不重要，有兴趣的可以去查一下。

刚才我们提到了平移不变性和局部性两个概念，引导我们把图片进行分割成一个个小区域再进行处理，这就是卷积的基本思想。这个时候我们会引入一个新的概念叫卷积核，一般是一个$3 \times 3$或者$ 5 \times 5 $的矩阵，我们会在图片中提取与卷积核一样大的部分，然后用卷积核与这部分进行点积运算，得到一个标量，这个过程我们可以用下面的公式表示：
$$ (f * g)(x) = \sum_{i=0}^{n} \sum_{j=0}^{n} f(i, j) g(x - i, y - j) $$
然后我们不断的滑动卷积核，直到卷积核覆盖完整个图片，我们就会得到一个全新的矩阵，这个矩阵就是我们卷积后的结果。
可以参考以下过程：
$$\begin{bmatrix} 0 & 1 & 2 \\ 3 & 4 & 5 \\ 6 & 7 & 8 \end{bmatrix} * \begin{bmatrix} 0 & 1 \\ 2 & 3 \end{bmatrix} = \begin{bmatrix} 19 & 25 \\ 37 & 45 \end{bmatrix}$$

我知道我这么讲肯定是有些抽象的，但是本人实力确实有限，不会做视频，所以大家可以去看b站上的视频，动画理解起来会比较容易，比如我当时学习的时候看的是这个[视频](【“卷积”到底“卷”了个啥？】https://www.bilibili.com/video/BV1SWUSYnE9f?vd_source=50ed0ee2d4962669e8b4cba297699e89)，这位up讲的很清晰。

这时候就要引出我在学习时遇到的一个问题了，我们知道了卷积是为了提取特征，卷积到底能提取什么特征？但是从结果导向来看，我们又可以分解成两个问题，一个是卷积层与MLP的关系，另一个是卷积结果矩阵与灰度图的关系。

我们先来回答一下第一个问题，卷积层与MLP的关系。虽然一般的卷积层会有多个卷积核，但是我们现在先不考虑这个问题，先假设卷积层只有一个卷积核，那么这一步其实可以等效成MLP中输入层 → 隐藏层的过程，图像被卷积层分割成一个个的小区域可以理解成MLP中的输入，MLP中使用的是线性变换即WA + b，权重矩阵W是一个$1 \times n $的矩阵，卷积层中的卷积核其实是一样的，我们甚至可以把卷积核抽象的展开去理解，卷积的过程其实就相当于MLP中的线性变换过程，所以我们不用刻意的去强调提取到了什么特征，跟多层感知机一样是如果存在特征会被这种结构提取出来。

现在来回答第二个问题，卷积结果矩阵与灰度图的关系。这里其实就涉及到会被提取出什么特征了，我们的卷积层经过不断的前向传播 → 反向传播 → 梯度下降之后会获得一个可以提取特征的卷积核，这里也可以引出卷积核的另一个名称**滤波器**了，顾名思义对输入进行过滤，这样我们的图像在卷积处理后可以进行一个可视化映射，高响应的部分数值会变得更大，表现为接近白色，低响应的部分数值会变得更小，表现为接近黑色，用颜色的强反差这样的方式区分特征和非特征。

但是图像中往往有很多组特征，我们需要通过不同的卷积核来识别不同的特征，所以一个卷积层里会有多个卷积核。

在卷积层处理图像时，我们可能还需要对图像进行填充和设置步幅。填充是指在图像的边界添加额外的像素，可以解决边缘信息丢失的问题；步幅是指卷积核在图像上滑动的步长，在图像很大的时候可以增大步幅，来减少输出的大小。填充和步幅都可以有效的调整维度，这里就先不展开讲了。
<br>
下面就要引入卷积神经网络的另一个重要组件——**池化层(Pooling Layer)**，也可以翻译成汇聚层。

90%的池化层都使用的是最大池化层和平均池化层，其实池化的操作是来自于统计学/信号处理的"降采样"思想，池化本质上就是一种局部最大化或者滑动平均。其目的是在对特征图降维的同时保留关键信息，同时根据平移不变性的原理，消除小幅的平移旋转，可以提高模型的鲁棒性。
池化层的计算过程如下：
$$ \begin{bmatrix} 0 & 1 & 2 \\ 3 & 4 & 5 \\ 6 & 7 & 8 \end{bmatrix} \xrightarrow{\text{2$\times$2 max pool}} \begin{bmatrix} 4 & 5 \\ 7 & 8 \end{bmatrix} $$

现在你已经充分理解了CNN卷积神经网络的基本原理，我们要开始写代码了，由于还是训练MNIST数据集，所以代码只有在模型设计部分会有一些变化，其他部分都是一样的，我们大可直接把上一篇的MLP的代码粘过来，直接修改网络架构即可。

网络架构如下：
```python
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32*7*7, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 32*7*7)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
```
```input_channels = 1```是输入通道为1，因为灰度图就只有一张图
```out_channels = 16```是输出通道为16，因为我们要用16个卷积核来提取特征
```kernel_size = 3```是卷积核的大小为3x3
```padding = 1```是填充为1
```stride = 2```是步幅为2
因为池化层的计算过程是对特征图进行降维，所以池化层的输出通道数与卷积层的输出通道数相同。

因为CNN比MLP强大，所以```epochs = 8```即可
<br>
完整代码：
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
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
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32*7*7, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 32*7*7)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

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
    model = CNN()
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    epochs = 8
    for i in range(epochs):
        print(f"\nEpoch {i+1}")
        train(model, train_dataloader, loss_function, optimizer)
        torch.save(model.state_dict(), "D:\Deep Learning for Beginners\Convolutional Neural Network\model\model.pth")

    # 绘制损失函数
    plt.figure(figsize=(12,5))
    plt.plot(train_loss_history, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()

    test(model, test_dataloader, loss_function)
```

卷积神经网络在传统视觉领域的应用是相当广泛的，比如图像分类、物体检测、图像分割等，可以说是人工智能/计算机视觉领域的基石之一，下一篇我应该会更新GNN/GCN或者LSTM/GRU的内容，相对来说LSTM的应用会更加广泛，RNN虽然也算是深度学习的基石之一，但是本身并不好用，所以我只会在LSTM篇简单介绍其中的递归思想。