## 卷积神经网络(CNN)训练MNIST数据集结果可视化

在上一篇的文章里我们训练了自己的卷积神经网络(CNN)模型，并且正确率能到达99.2%，但是正确率永远都是一个冷冰冰的数据，我们当然想实打实的看到我们的模型效果到底如何，所以我简单写了一个pygame的窗口，我们可以通过鼠标书写数字，然后模型识别并反馈给我们

完整代码如下:
```python
import pygame
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    
model = CNN()
model.load_state_dict(torch.load("D:\\Deep Learning for Beginners\\Convolutional Neural Network\\model\\model.pth"))
model.eval()

# pygame初始化
cell = 20
size = 28 * cell
pygame.init()
screen = pygame.display.set_mode((size, size))
pygame.display.set_caption("Draw a digit (0-9) - ESC to clear, SPACE to predict")
canvas = np.zeros((28, 28), dtype=np.uint8)

def draw():
    # 将28x28的canvas放大到560x560显示
    scaled_canvas = np.repeat(np.repeat(canvas * 255, cell, axis=0), cell, axis=1)
    surface = pygame.surfarray.make_surface(scaled_canvas.T)  # pygame需要转置
    screen.blit(surface, (0, 0))
    pygame.display.flip()

def clean():
    canvas[:] = 0
    pygame.display.set_caption('Draw a digit (0-9) - ESC to clear, SPACE to predict')

def main():
    clock = pygame.time.Clock()
    drawing = False
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    clean()

                elif event.key == pygame.K_SPACE:
                    # 推理
                    img = canvas.astype(np.float32)
                    img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)    # (1,1,28,28)
                    
                    with torch.no_grad():
                        logits = model(img)
                        prob = torch.softmax(logits, dim=1)
                        pred = int(torch.argmax(prob, dim=1))
                        conf = float(prob[0, pred])
                    pygame.display.set_caption(f'Predict: {pred}  (conf={conf:.2f})')

            elif event.type == pygame.MOUSEBUTTONDOWN:
                drawing = True

            elif event.type == pygame.MOUSEBUTTONUP:
                drawing = False

            elif event.type == pygame.MOUSEMOTION and drawing:
                x, y = pygame.mouse.get_pos()
                gx, gy = x // cell, y // cell
                if 0 <= gx < 28 and 0 <= gy < 28:
                    # 涂2x2区域
                    canvas[gy, gx] = 1
                    if gx + 1 < 28:
                        canvas[gy, gx + 1] = 1
                    if gy + 1 < 28:
                        canvas[gy + 1, gx] = 1
                    if gx + 1 < 28 and gy + 1 < 28:
                        canvas[gy + 1, gx + 1] = 1

        draw()
        clock.tick(120)

if __name__ == '__main__':
    main()
```
你只需要把```model.load_state_dict(torch.load("D:\\Deep Learning for Beginners\\Convolutional Neural Network\\model\\model.pth"))```这行代码中的路径改为你自己的模型路径即可，同时要注意我们保存模型的时候只保存了参数，所以一开始设计模型的时候要与训练时的模型一致。

这样你就可以实打实的测试你的模型效果了！