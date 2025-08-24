## 简单的多层感知机(MLP)推导

这一篇文章主要是想用大家能听懂的方式简单讲一下神经网络内部到底是如何运行推导的，当然我对神经网络做出了一定的简化，但其实原理很简单，只是需要大家有一定的数学基础，涉及到的数学知识只有一些矩阵和梯度的计算。
<br>

先带大家看一下简化后的神经网络，这里我简化了具体的全连接层：
```mermaid
flowchart LR
    A(输入层) --> B(隐藏层1)
    B --> C(隐藏层2)
    C --> D(输出层)
```

神经网络可以简单理解成一个**前向传播 → 反向传播 → 梯度下降**的过程。
<br>

首先是**前向传播**，我们可以简单的理解为**线性变换 + 激活函数**。

线性变换的公式很简单： $Z^{[L]} = W^{[L]} A^{[L-1]} + b^{[L]}$

| 符号 | 含义 | 维度|
|:---|:---|:---|
| $Z^{[L]}$ | 第 $L$ 层的线性变换结果 | $(n^{[L]}, batch)$ |
| $W^{[L]}$ | 第 $L$ 层的权重矩阵 | $(n^{[L]}, n^{[L-1]})$ |
| $A^{[L-1]}$ | 第 $L-1$ 层的激活结果 | $(n^{[L-1]}, batch)$ |
| $b^{[L]}$ | 第 $L$ 层的偏置向量 | $(n^{[L]}, 1)$ |


激活函数的公式也很简单： $A^{[L]} = g(Z^{[L]})$

下面我会列举几个常见的激活函数：

| 激活函数 | 公式 | 值域 |
|:---|:---|:---|
| Sigmoid | **$g(z) = \frac{1}{1 + e^{-z}}$** | $(0, 1)$ |
| Tanh | **$g(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$** | $(-1, 1)$ |
| ReLU | **$g(z) = \max(0, z)$** | $[0, +\infty)$ |

为了便于理解，我们假设一个具体的网络结构，这个神经网络有两个输入特征，隐藏层1有3个神经元，隐藏层2有2个神经元，输出层有1个神经元。

现在公式都知道了，那我们就开始推导吧！
1. 输入层 → 隐藏层1：
$Z^{[1]} = W^{[1]} A^{[0]} + b^{[1]} \quad\quad A^{[1]} = g(Z^{[1]})$
维度：$(3×1) = (3×2) \cdot (2×1) + (3×1)$
<br>
2. 隐藏层1 → 隐藏层2：
$Z^{[2]} = W^{[2]} A^{[1]} + b^{[2]} \quad\quad A^{[2]} = g(Z^{[2]})$
维度：$(2×1) = (2×3) \cdot (3×1) + (2×1)$
<br>
3. 隐藏层2 → 输出层：
$Z^{[3]} = W^{[3]} A^{[2]} + b^{[3]} \quad\quad A^{[3]} = Z^{[3]}$
维度：$(1×1) = (1×2) \cdot (2×1) + (1×1)$

前向传播就这么简单，我们可以得到输出层的结果 $A^{[3]}$。
<br>

然后是**反向传播**，我们可以简单的理解为**损失函数的梯度计算**，主要是告诉网络优化的方向，找出每个参数对最终误差的“贡献度”，从而精准地更新参数，让模型学得更快更好，因此梯度是反向传播的'灵魂'，它让模型从“盲目猜测”变为“精准优化”，将数学上的微分工具转化为工程上的高效学习算法。


**链式法则**可以帮我们计算复合函数的梯度，公式如下： **$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial x}$**
这样我们就可以求出前向传播中任何一个参数矩阵的梯度了。

现在开始推导吧！
1. 这里损失函数我们用**均方误差**： $L = \frac{1}{2} (Z^{[3]} - y)^2$
2. 输出层 → 隐藏层2：
对输出层线性输出$Z^{[3]}$的梯度： $ \frac{\partial L}{\partial Z^{[3]}} = Z^{[3]} - y $
<br>
对隐藏层2的权重矩阵$W^{[3]}$的梯度： $ \frac{\partial L}{\partial W^{[3]}} = \frac{\partial L}{\partial Z^{[3]}} \cdot \frac{\partial Z^{[3]}}{\partial W^{[3]}} = (Z^{[3]} - y) \cdot (A^{[2]})^T $
<br>
对隐藏层2的偏置向量$b^{[3]}$的梯度： $ \frac{\partial L}{\partial b^{[3]}} = \frac{\partial L}{\partial Z^{[3]}} \cdot \frac{\partial Z^{[3]}}{\partial b^{[3]}} = (Z^{[3]} - y) $
<br>
对隐藏层2的激活结果$A^{[2]}$的梯度： $ \frac{\partial L}{\partial A^{[2]}} = \frac{\partial L}{\partial Z^{[3]}} \cdot \frac{\partial Z^{[3]}}{\partial A^{[2]}} = (Z^{[3]} - y) \cdot W^{[3]} $
<br>
3. 隐藏层2 → 隐藏层1：
对隐藏层2线性输出$Z^{[2]}$的梯度： $ \frac{\partial L}{\partial Z^{[2]}} = \frac{\partial L}{\partial A^{[2]}} \odot \frac{\partial A^{[2]}}{\partial Z^{[2]}} = \frac{\partial L}{\partial A^{[2]}} \odot g'(Z^{[2]}) $
<br>
对隐藏层1的权重矩阵$W^{[2]}$的梯度： $ \frac{\partial L}{\partial W^{[2]}} = \frac{\partial L}{\partial Z^{[2]}} \cdot \frac{\partial Z^{[2]}}{\partial W^{[2]}} = \frac{\partial L}{\partial Z^{[2]}} \cdot (A^{[1]})^T $
维度：$(2×3) = (2×1) \cdot (1×3)$
<br>
对隐藏层1的偏置向量$b^{[2]}$的梯度： $ \frac{\partial L}{\partial b^{[2]}} = \frac{\partial L}{\partial Z^{[2]}} \cdot \frac{\partial Z^{[2]}}{\partial b^{[2]}} = \frac{\partial L}{\partial Z^{[2]}} $
<br>
对隐藏层1的激活结果$A^{[1]}$的梯度： $ \frac{\partial L}{\partial A^{[1]}} = \frac{\partial L}{\partial Z^{[2]}} \cdot \frac{\partial Z^{[2]}}{\partial A^{[1]}} = (W^{[2]})^T \cdot \frac{\partial L}{\partial Z^{[2]}} $
维度：$(3×1) = (3×2) \cdot (2×1)$
<br>
4. 隐藏层1 → 输入层：
对隐藏层1线性输出$Z^{[1]}$的梯度： $ \frac{\partial L}{\partial Z^{[1]}} = \frac{\partial L}{\partial A^{[1]}} \odot \frac{\partial A^{[1]}}{\partial Z^{[1]}} = \frac{\partial L}{\partial A^{[1]}} \odot g'(Z^{[1]}) $
<br>
对输入层的权重矩阵$W^{[1]}$的梯度： $ \frac{\partial L}{\partial W^{[1]}} = \frac{\partial L}{\partial Z^{[1]}} \cdot \frac{\partial Z^{[1]}}{\partial W^{[1]}} = \frac{\partial L}{\partial Z^{[1]}} \cdot X^{T} $
维度：$(3×2) = (3×1) \cdot (1×2)$
<br>
对输入层的偏置向量$b^{[1]}$的梯度： $ \frac{\partial L}{\partial b^{[1]}} = \frac{\partial L}{\partial Z^{[1]}} \cdot \frac{\partial Z^{[1]}}{\partial b^{[1]}} = \frac{\partial L}{\partial Z^{[1]}} $
<br>
对输入层的输入$X$的梯度： $ \frac{\partial L}{\partial X} = \frac{\partial L}{\partial Z^{[1]}} \cdot \frac{\partial Z^{[1]}}{\partial X} = (W^{[1]})^T \cdot \frac{\partial L}{\partial Z^{[1]}} $
维度：$(2×1) = (2×3) \cdot (3×1)$
<br>

最后是**梯度下降**，我们可以简单的理解为**参数更新**，主要是根据梯度来更新参数，让模型朝着梯度的反方向移动，从而减小损失函数的值，公式如下： **$W^{[L]} = W^{[L]} - \alpha \frac{\partial L}{\partial W^{[L]}}$**

我们直接开始推导：
1. 输入层 → 隐藏层1：
$W^{[1]} = W^{[1]} - \eta \frac{\partial L}{\partial W^{[1]}} \quad\quad b^{[1]} = b^{[1]} - \eta \frac{\partial L}{\partial b^{[1]}}$
<br>
2. 隐藏层1 → 隐藏层2：
$W^{[2]} = W^{[2]} - \eta \frac{\partial L}{\partial W^{[2]}} \quad\quad b^{[2]} = b^{[2]} - \eta \frac{\partial L}{\partial b^{[2]}}$
<br>
3. 隐藏层2 → 输出层：
$W^{[3]} = W^{[3]} - \eta \frac{\partial L}{\partial W^{[3]}} \quad\quad b^{[3]} = b^{[3]} - \eta \frac{\partial L}{\partial b^{[3]}}$
<br>

这样我们就完成了一次模型的迭代,训练的过程就是不断的迭代，直到损失函数的值趋于稳定。

下面是一些我们值得深入思考的问题，算是一个 Q & A 环节：
1. 偏置b存在的意义是什么？
2. 为什么反向传播中要使用权重矩阵的转置？
3. 学习率 $ \eta $ 为什么不能太大也不能太小？
4. 梯度消失和梯度爆炸为什么会发生？
<br>

答案如下：
1. 偏置b相当于线性函数 y = wx + b 中的截距项。没有偏置时，神经元只能学习过原点的变换；有了偏置，神经元可以"平移"激活函数，让模型在输入为0时仍能产生非零输出，大大增强了模型的表达能力。
2. 这是链式法则和矩阵维度匹配的要求。在前向传播中，我们用 W·A 得到下一层；在反向传播中，梯度要"逆向"传播回上一层，所以需要用 W^T·梯度。这样既保证了数学上的正确性，也确保了矩阵维度能够匹配。
3. 学习率太大的话，每次更新步长过大，可能直接"跳过"最优解，导致损失函数震荡甚至发散；而太小的话，每次更新步长过小，模型收敛极其缓慢，需要很多轮训练才能达到较好效果。
4. 当激活函数导数很小（如Sigmoid的饱和区）时，多层相乘后梯度变得极小，深层参数几乎不更新，就会出现梯度消失的问题；当权重过大时，多层相乘后梯度变得极大，导致参数更新过于剧烈，训练不稳定，就会出现梯度爆炸的问题。

有什么不理解的地方或者我推导有问题的地方也欢迎大家留言讨论。