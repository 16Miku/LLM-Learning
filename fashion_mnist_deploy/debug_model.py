import torch
from main import MNIST_CNN_Model, class_names # 从你的main.py导入模型和类别
from PIL import Image
import torchvision.transforms as transforms

# --- 1. 加载模型 (和main.py完全一样) ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model = MNIST_CNN_Model().to(device)
model.load_state_dict(torch.load("fashion_mnist_cnn_model.pth", map_location=device))
model.eval()

# --- 2. 准备一张测试图片 ---
# 确保你有一张名为 test_image.jpg 的图片在文件夹里
IMAGE_PATH = "test_image.jpg" 

try:
    image = Image.open(IMAGE_PATH)
except FileNotFoundError:
    print(f"错误: 找不到图片 '{IMAGE_PATH}'。请确保图片存在。")
    exit()

# --- 3. 定义和应用预处理 ---
# 这里的 transform 必须和 main.py 里的一模一样！
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

image_tensor = transform(image).unsqueeze(0).to(device)

# --- 4. 进行预测 ---
with torch.no_grad():
    output = model(image_tensor)
    pred_index = torch.argmax(output, dim=1).item()
    pred_class_name = class_names[pred_index]

print("--- 本地调试结果 ---")
print(f"模型预测的类别是: {pred_class_name}")
