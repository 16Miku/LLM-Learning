


## 教AI学会判断一个数字是“大”还是“小”


非常好！我们继续前进。

你已经成功教会了AI如何**预测一个连续的数值**（找到最佳直线），这叫做“**回归**”。

现在，我们要教它一个新技能：**做出一个判断**，比如“是”或“不是”、“A类”或“B类”。这叫做“**分类**”。

### 新的挑战：教AI判断一个数字是“大”还是“小”

*   **规则很简单：**
    *   如果一个数字小于5，我们把它归为“**小数**”（类别 0）。
    *   如果一个数字大于等于5，我们把它归为“**大数**”（类别 1）。
*   **AI的任务：** 只看一堆带标签的数字，自己学会这个“以5为界”的规则。

---

### 第一步：准备“分类”教材

我们的教材包含两部分：问题（一些数字）和答案（0代表小数，1代表大数）。

**在Colab的第一个代码块中输入并运行：**

```python
import torch
import matplotlib.pyplot as plt

# --- 准备分类教材 ---

# 1. 创建问题 (X)
# 我们创建从0到10的一系列数字
X = torch.arange(0, 10, 0.25).view(-1, 1)

# 2. 创建标准答案 (y)
# y不再是一个连续的数字，而是类别标签：0 或 1
# X < 5 会返回一个布尔值 (True/False)，我们把它转成浮点数 (1.0/0.0)
y = (X >= 5).float()

# --- 画图看看教材长什么样 ---
def plot_data(X_data, y_data, title="我们的分类教材"):
    plt.figure(figsize=(8, 5))
    # 我们用不同的标记来画出两类点
    plt.scatter(X_data[y_data == 0], y_data[y_data == 0], c='blue', marker='o', label='小数 (类别 0)')
    plt.scatter(X_data[y_data == 1], y_data[y_data == 1], c='red', marker='x', label='大数 (类别 1)')
    plt.title(title)
    plt.xlabel("问题 (数字X)")
    plt.ylabel("答案 (类别y)")
    plt.legend()
    plt.grid(True)
    # y轴只显示0和1
    plt.yticks([0, 1])
    plt.show()

plot_data(X, y)
```

运行后，你会看到一张图。所有蓝色的圆圈（小数）都在y=0这条线上，所有红色的叉（大数）都在y=1这条线上。AI的目标就是学会画一条“分界线”，把这两类点分开。

---

### 第二步：给学生一个“新大脑” (修改模型)

上次学生的大脑只会画直线，但直线可以无限延伸。而这次我们的答案只有0和1，我们需要学生的回答被**压缩**在0到1之间。

所以，我们要给它加一个新工具：**Sigmoid函数**。

*   **Sigmoid是什么？** 你可以把它想象成一个“**压扁机**”。无论你给它多大或多小的数字，它都能把它压成一个0到1之间的小数。
*   **有什么用？** 这个0到1之间的小数，可以被看作是一个**概率**。比如，输出0.98就代表学生有98%的把握认为这个数是“大数”。

**在Colab的第二个代码块中输入并运行：**

```python
from torch import nn

# --- 创造一个会做判断的学生 ---

# 我们把“画直线”和“压扁机”打包在一起
model = nn.Sequential(
    nn.Linear(1, 1),   # 第一步：还是像上次一样，先在脑子里画条线
    nn.Sigmoid()       # 第二步：把线上的结果用“压扁机”压到0和1之间
)

print("--- 这是我们学生的新大脑结构 ---")
print(model)
```

---

### 第三步：更新“评分标准”

上次我们用“平均差距”（MSELoss）来评分，因为它适合衡量距离。但这次是判断题，我们得换个更聪明的评分标准：**BCELoss** (Binary Cross-Entropy Loss)。

*   **BCELoss是什么？** 你可以把它理解成一个专门给“是/否”判断题打分的老师。
    *   如果正确答案是“是”(1)，学生回答的概率越接近1，扣分越少。
    *   如果正确答案是“否”(0)，学生回答的概率越接近0，扣分越少。
    *   它特别擅长惩罚那些“**答错还特别自信**”的学生。

**在Colab的第三个代码块中输入并运行：**

```python
# --- 准备新的评分标准 ---

# 1. 评分标准：专门给判断题打分的老师
loss_fn = nn.BCELoss()

# 2. 反思方法：这个不用变！还是让学生慢慢调整
optimizer = torch.optim.SGD(model.parameters(), lr=0.1) # 学习率可以稍微调大一点

print("新的评分标准准备好了！")
```

---

### 第四步：开始上课！(训练分类模型)

教学流程和上次**一模一样**！这再次证明了这套流程的强大。

**在Colab的第四个代码块中输入并运行：**

```python
# --- 开始上课 ---

epochs = 200
print("--- 教学开始 ---")

for epoch in range(epochs):
    # 1. 学生做题 (给出判断概率)
    y_pred_prob = model(X)
    
    # 2. 老师评分
    loss = loss_fn(y_pred_prob, y)
    
    # 3. 学生反思与调整
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        print(f"学习第 {epoch+1} 遍, 答题差距 (Loss): {loss.item():.4f}")

print("--- 教学结束 ---")
```

你会看到，这次的“答题差距(Loss)”也一样在飞快地减小。

---

### 第五步：毕业考试 (看看AI画的“分界线”)

学生学完了，我们来看看它的学习成果。它学到的不再是一条简单的直线，而是一条**S形的曲线**。这条曲线代表了它判断一个数是“大数”的**信心**。

**在Colab的最后一个代码块中输入并运行：**

```python
# --- 毕业考试 ---

# 我们来画出学生脑子里的那条S形“信心曲线”
def plot_decision_boundary(X_data, y_data, model):
    model.eval()
    with torch.no_grad():
        y_pred_prob = model(X_data)
    
    plt.figure(figsize=(8, 5))
    plt.scatter(X_data[y_data == 0], y_data[y_data == 0], c='blue', marker='o', label='小数 (类别 0)')
    plt.scatter(X_data[y_data == 1], y_data[y_data == 1], c='red', marker='x', label='大数 (类别 1)')
    plt.plot(X_data, y_pred_prob, color='green', label='AI的“信心曲线”')
    plt.axhline(y=0.5, color='gray', linestyle='--', label='50%信心分界线') # 画一条50%信心的辅助线
    plt.title("学生的毕业作品")
    plt.xlabel("问题 (数字X)")
    plt.ylabel("判断为“大数”的概率")
    plt.legend()
    plt.grid(True)
    plt.yticks([0, 0.5, 1])
    plt.show()

plot_decision_boundary(X, y, model)


# --- 考几个新问题 ---
print("\n--- 老师开始提问 ---")
test_questions = torch.tensor([[2.0], [4.9], [5.1], [9.0]])
predictions_prob = model(test_questions)

for i in range(len(test_questions)):
    question = test_questions[i].item()
    prob = predictions_prob[i].item()
    # 我们以50%的信心作为判断标准
    answer = "大数" if prob > 0.5 else "小数"
    print(f"问题: {question:.1f} -> AI判断为“大数”的概率是 {prob*100:.1f}%, 所以我认为是: {answer}")
```

运行后，你会看到一张图，绿色的S形曲线就是AI的学习成果。
*   当X很小时，曲线接近0，表示AI非常有信心这是“小数”。
*   当X很大时，曲线接近1，表示AI非常有信心这是“大数”。
*   **曲线穿过0.5的地方，大约就在X=5附近！** 这说明AI真的学会了以5为界的规则！

最后，提问环节的结果也证明了这一点。AI能够准确地判断出4.9是小数，而5.1是大数。

### 本次课程总结

你又成功地教会了AI一项新技能！这次你学到了：

1.  **分类任务**：AI不仅能预测数值，还能做出判断。
2.  **Sigmoid**：一个能将任何数值“压扁”成概率的强大工具。
3.  **BCELoss**：一个更适合分类任务的“评分标准”。
4.  **决策边界**：AI学习分类，本质上是学习一条或一个面，来把不同类别的数据分开。

你已经掌握了PyTorch解决两大核心问题（回归和分类）的基本方法。你觉得这个过程有趣吗？




