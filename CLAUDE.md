# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

这是一个 PyTorch 和大语言模型 (LLM) 学习教程仓库，采用渐进式学习路径，从 PyTorch 基础到 LLM 应用开发。主要开发环境为 Google Colab，部分代码可在本地运行。

## 技术栈

- **深度学习**: PyTorch, TorchVision
- **LLM 生态**: Hugging Face Transformers, PEFT (LoRA), Unsloth, BitsAndBytes
- **应用框架**: LangChain, CrewAI
- **部署**: FastAPI, Uvicorn, vLLM
- **其他**: OpenAI Whisper, LLaVA

## 常用命令

### FastAPI 部署服务 (fashion_mnist_deploy)

```bash
# 安装依赖
pip install fastapi "uvicorn[standard]" python-multipart Pillow torch torchvision

# 启动服务
cd fashion_mnist_deploy
uvicorn main_3:app --reload

# 测试 API
python test_api.py
```

### Jupyter Notebook

```bash
jupyter notebook
```

### 常用依赖安装

```bash
# PyTorch
pip install torch torchvision

# Hugging Face
pip install transformers datasets accelerate peft

# LangChain
pip install langchain langchain-google-genai

# CrewAI
pip install crewai
```

## 代码架构

### 学习路径

1. **PyTorch 基础** (PyTorch_0-4): 回归 → 分类 → 神经网络 → CNN
2. **模型部署** (PyTorch_5 + fashion_mnist_deploy): FastAPI Web API
3. **Transformer** (Transformer_Learning): 理论与实践
4. **LLM 应用** (LLM_Learning_1-14): 预训练模型调用 → 微调 → LoRA → RAG → 多智能体

### 关键目录

- `fashion_mnist_deploy/`: FastAPI + PyTorch CNN 部署示例
  - `main_3.py`: 最新版 API 服务代码
  - `test_api.py`: API 测试脚本
  - `*.pth`: 训练好的模型权重

### 实战项目: Chat-嬛嬛

基于 Llama3-8B 的 LoRA 微调项目，模型已发布至 Hugging Face:
- https://huggingface.co/16Miku/Chat-HuanHuan-Llama3-8B-merged

## 注意事项

- 大多数 LLM 相关笔记本需要 GPU 运行，建议使用 Google Colab
- 4 位量化相关代码可在 Colab 免费 T4 GPU 上运行
- 如遇 Notebook 渲染问题，参考 `Remove-Widgets.md`
