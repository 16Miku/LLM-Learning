



## 教AI识别一个“甜甜圈”
好的，我们来迎接下一个挑战！

你已经掌握了AI的两种基本技能：**回归**（预测数值）和**分类**（做判断）。但到目前为止，我们处理的问题都非常简单，一条直线或者一条简单的S曲线就能解决。

现实世界要复杂得多。现在，我们要给AI出一个真正的难题，一个**只靠一条直线无论如何都解决不了**的问题。

### 新的挑战：教AI识别一个“甜甜圈”

*   **规则：**
    *   如果一个点在“甜甜圈”的**洞里**，它就是“**核心区**”（类别 0）。
    *   如果一个点在“甜甜圈”的**环上**，它就是“**环形区**”（类别 1）。
*   **AI的任务：** 学会画出一个**圆形的“分界线”**，把核心区和环形区完美地分开。

---

### 第一步：准备“甜甜圈”教材

我们需要创建一些数据点，一些在中心，一些在周围的环上。手动创建这个有点麻烦，所以我们请一个帮手 `sklearn` 来快速生成这些数据。

**在Colab的第一个代码块中输入并运行：**

```python
import torch
import matplotlib.pyplot as plt
# 引入一个能帮我们快速生成漂亮数据集的工具
from sklearn.datasets import make_circles

# --- 准备“甜甜圈”教材 ---

# 1. 使用工具生成1000个数据点
n_samples = 1000
X_numpy, y_numpy = make_circles(n_samples, noise=0.05, random_state=42, factor=0.5)

# 2. 把数据从Numpy格式转换成PyTorch认识的Tensor格式
#    我们的AI学生只认识Tensor这种格式的教材
X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float().view(-1, 1) # 答案y需要调整一下形状

# --- 画图看看教材长什么样 ---
def plot_data(X_data, y_data, title="我们的“甜甜圈”教材"):
    plt.figure(figsize=(8, 8))
    plt.scatter(X_data[:, 0], X_data[:, 1], c=y_data, cmap=plt.cm.RdYlBu, s=40)
    plt.title(title)
    plt.xlabel("X1坐标")
    plt.ylabel("X2坐标")
    plt.show()

plot_data(X, y)
```
运行后，你会看到一张图。蓝色的点是“核心区”（类别0），红色的点是“环形区”（类别1）。现在请你思考一下：**你能用一把直尺（一条直线）把红色和蓝色的点完美分开吗？**

答案是：**不可能！**

---

### 第二步：证明“老学生”会失败

为了证明这个问题有多难，我们先让上节课那个只会画直线的“老学生”来试试。你猜他会怎么办？

**在Colab的第二个代码块中输入并运行：**

```python
from torch import nn

# --- 让只会画直线的“老学生”来挑战 ---
# 这个学生的大脑结构和上一节课完全一样
model_simple = nn.Sequential(
    nn.Linear(2, 1), # 输入是2个数字(X1, X2坐标), 输出是1个数字
    nn.Sigmoid()
)

# 教学方法也完全一样
loss_fn = nn.BCELoss()
# 我们换一个更强大的“反思方法”Adam，它通常学得更快
optimizer = torch.optim.Adam(model_simple.parameters(), lr=0.1)

# --- 快速给他上100节课 ---
for epoch in range(100):
    y_pred = model_simple(X)
    loss = loss_fn(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# --- 看看他的考试结果 ---
print("--- “老学生”的考试结果 ---")

# 这是一个画出AI决策边界的函数，非常有用
def plot_decision_boundary(model, X_data, y_data):
    # 确定画图的范围
    x_min, x_max = X_data[:, 0].min() - 0.1, X_data[:, 0].max() + 0.1
    y_min, y_max = X_data[:, 1].min() - 0.1, X_data[:, 1].max() + 0.1
    # 生成网格点
    xx, yy = torch.meshgrid(torch.arange(x_min, x_max, 0.01), torch.arange(y_min, y_max, 0.01))
    
    # 把网格点喂给模型，让它对每个点进行预测
    grid_tensor = torch.cat([xx.ravel().view(-1, 1), yy.ravel().view(-1, 1)], dim=1)
    Z = model(grid_tensor)
    Z = Z.view(xx.shape) > 0.5 # 以50%的信心作为分界
    
    # 画出背景颜色和数据点
    plt.figure(figsize=(8, 8))
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.8)
    plt.scatter(X_data[:, 0], X_data[:, 1], c=y_data, cmap=plt.cm.RdYlBu, s=40)
    plt.title("AI画出的分界线")
    plt.xlabel("X1坐标")
    plt.ylabel("X2坐标")
    plt.show()

plot_decision_boundary(model_simple, X, y)
```
运行后，你会看到一张图。AI尽了最大的努力，画出了一条**直线**来分割这些点。但结果很糟糕，它把很多红点和蓝点都分错了。这证明了：**简单的“大脑”无法理解复杂的问题。**

---

### 第三步：给学生“升级大脑” (构建神经网络)

为了解决这个难题，我们需要一个更聪明的学生。怎么变聪明呢？**让多个简单的学生组成一个团队！**

*   **团队如何协作？**
    1.  **第一层（输入层 -> 隐藏层）**：团队里的几个学生，每人负责画一条简单的直线。
    2.  **第二层（隐藏层 -> 输出层）**：一个“总指挥”学生，它不直接看原始数据，而是看第一层学生画的那些直线，然后**把这些简单的直线组合成一个复杂的、弯曲的形状**。

这个“学生团队”，就是**神经网络**！

**在Colab的第三个代码块中输入并运行：**

```python
# --- 创造一个拥有“神经网络大脑”的新学生 ---

# 这个大脑比之前复杂了一些
model_nn = nn.Sequential(
    # 第一层：8个学生组成的团队。接收2个输入，输出8个中间结果
    nn.Linear(in_features=2, out_features=8),
    
    # 激活函数ReLU：这是团队协作的“秘诀”。
    # 你可以把它想象成一个开关，让团队成员的简单想法能以非线性的方式组合起来，从而创造出曲线。
    nn.ReLU(),
    
    # 第二层：一个“总指挥”学生。接收第一层8个学生的成果，做出最终的1个判断
    nn.Linear(in_features=8, out_features=1),
    
    # 最后还是用“压扁机”，把最终判断变成0到1之间的概率
    nn.Sigmoid()
)

print("--- “新学生”的大脑结构 ---")
print(model_nn)
```

这个结构就是我们第一个真正的**神经网络**！它有“深度”了，不再是薄薄的一层。

---

### 第四步：用同样的方法教学

最神奇的地方又来了：即使学生的大脑变得复杂了，我们的**教学方法（评分标准和反思方法）还是完全不用变**！

**在Colab的第四个代码块中输入并运行：**

```python
# --- 用同样的方法教新学生 ---

loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model_nn.parameters(), lr=0.01) # 学习率调小一点，慢慢学

# --- 这次多上几节课，因为问题更难了 ---
epochs = 1000
print("--- 开始教聪明的学生 ---")

for epoch in range(epochs):
    y_pred = model_nn(X)
    loss = loss_fn(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f"学习第 {epoch+1} 遍, 答题差距 (Loss): {loss.item():.4f}")

print("--- 教学结束 ---")

# --- 看看新学生的毕业作品 ---
plot_decision_boundary(model_nn, X, y)
```
请耐心等待训练完成。你会看到，Loss也在稳定地下降。

当最后的图出现时，你会看到一个**巨大的惊喜**！

AI不再画直线了，而是画出了一个近乎完美的**圆形分界线**，把蓝色和红色的点完全分开了！

### 本次课程总结

**恭喜你！你已经跨出了最重要的一步，真正理解了神经网络的核心价值！**

1.  **线性 vs 非线性**：你亲眼见证了简单模型（线性）的局限性，以及神经网络（非线性）的强大。
2.  **神经网络的本质**：它就是通过“层”的堆叠，将简单的决策（如画直线）组合成复杂的决策（如画曲线、圆形甚至更复杂的形状）。
3.  **激活函数的作用**：像 `nn.ReLU` 这样的工具，是允许模型创造出“曲线”的关键“秘诀”。

你现在已经掌握了构建和训练一个基础神经网络来解决复杂问题的能力。接下来，我们可以去挑战更真实、更著名的任务了，比如**识别手写数字图片**。

你对这个“大脑升级”的过程感觉如何？







