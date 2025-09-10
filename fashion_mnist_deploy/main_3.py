# 导入所有需要的工具
import torch
from torch import nn
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torchvision.transforms as transforms
import io

# --- 1. 定义你的模型架构 ---
# 这里的代码必须和你训练时使用的模型架构一模一样！
# 所以我们直接从训练代码中复制过来。
# `Model_V2` (新的模型架构)
class FashionMNIST_Model_V2(nn.Module):
    def __init__(self):
        super().__init__()
        # --- 特征提取器 (变得更深了) ---
        self.feature_extractor = nn.Sequential(
            # 第一个卷积模块
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 第二个卷积模块
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # --- 分类器 (现在接收更小的特征图) ---
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # 经过两次2x2的池化，28x28的图片变成了 7x7
            # 最后的通道数是64
            nn.Linear(in_features=64 * 7 * 7, out_features=10)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x

# --- 2. 加载你训练好的模型 ---

# a. 自动检测可用设备
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"正在使用的设备: {device}")

# b. 实例化一个模型 (先不要送到设备)
model = FashionMNIST_Model_V2()

# c. 加载权重，并告诉它要映射到我们当前自动选择的设备上
model.load_state_dict(torch.load("fashion_mnist_cnn_model_v3.pth", map_location=device))

# d. 最后，把整个加载好的模型送到正确的设备上
model.to(device)

# e. 切换到评估模式
model.eval()

# --- 3. 定义类别名称 (和保存时一样) ---
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]




# 在 main.py 中修改

# --- 4. 定义图片预处理流程 ---
# 这个流程现在必须和我们重新训练时的流程一模一样！
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1), # 确保是灰度图
    transforms.Resize((28, 28)),                # 缩放到28x28
    transforms.ToTensor(),                      # 转换成Tensor
    transforms.Normalize(mean=(0.5,), std=(0.5,)) # <--- 添加了这一行！
])


# --- 5. 创建FastAPI应用 ---
app = FastAPI(title="Fashion MNIST Classifier API")

# --- 6. 创建API接口 (Endpoint) ---
# @app.post("/predict") 告诉FastAPI:
# 当有POST请求发送到 "/predict" 这个网址时，就执行下面的函数
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    接收一张图片，返回AI对这张图片的分类预测。
    """
    # a. 读取上传的文件内容
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    # b. 对图片进行预处理
    # .unsqueeze(0) 是为了增加一个“批次”维度，因为模型期望的输入是 [B, C, H, W]
    image_tensor = transform(image).unsqueeze(0).to(device) # <--- 添加 .to(device)

    # c. 用模型进行预测 (在torch.no_grad()下，不做梯度计算，速度更快)
    with torch.no_grad():
        output = model(image_tensor)
        # 获取最可能的类别索引
        pred_index = torch.argmax(output, dim=1).item()
        # 从类别名称列表中找到对应的名称
        pred_class_name = class_names[pred_index]

    # d. 返回JSON格式的预测结果
    return {"predicted_class": pred_class_name}

# --- 7. (可选) 创建一个根接口，用于测试服务是否启动 ---
@app.get("/")
def read_root():
    return {"message": "Welcome to the Fashion MNIST API!"}
