import requests
import random

# --- 准备一张测试图片 ---
# 找一张裤子(Trouser)的图片，或者其他衣物的图片，放到和脚本相同的文件夹下
# 并把文件名修改为 "test_image.jpg"
IMAGE_PATH = "test_image.jpg" 
# 替换成你的图片路径

# API服务器的地址
API_URL = "http://127.0.0.1:8000/predict"

# --- 发送请求 ---
try:
    # 'rb' 表示以二进制只读模式打开文件
    with open(IMAGE_PATH, "rb") as image_file:
        # 'files' 参数是requests库用来发送文件的特定格式
        files = {"file": (image_file.name, image_file, "image/jpeg")}
        
        print("正在向服务器发送图片，请稍候...")
        # 发送POST请求
        response = requests.post(API_URL, files=files)
        
        # 检查服务器是否成功响应
        response.raise_for_status() 
        
        # 解析服务器返回的JSON结果
        result = response.json()
        
        print("\n--- AI服务器返回的结果 ---")
        print(f"AI认为这张图片是: {result['predicted_class']}")

except FileNotFoundError:
    print(f"错误: 找不到图片文件 '{IMAGE_PATH}'。请确保图片和脚本在同一个文件夹下。")
except requests.exceptions.RequestException as e:
    print(f"错误: 请求失败。请确保FastAPI服务器正在运行。")
    print(f"详细错误: {e}")


