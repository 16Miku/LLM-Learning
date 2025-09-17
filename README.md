
# 从零逐步指导学习和掌握PyTorch

[PyTorch_Learning.md](PyTorch_Learning.md)

[PyTorch_Learning.ipynb](PyTorch_Learning.ipynb)



# PyTorch基础知识手册

[PyTorch_0.md](PyTorch_0.md)




# 教AI学会画一条直线（回归）

[PyTorch_1.md](PyTorch_1.md)

[PyTorch_1.ipynb](PyTorch_1.ipynb)



# 教AI学会判断一个数字是“大”还是“小”（分类）

[PyTorch_2.md](PyTorch_2.md)

[PyTorch_2.ipynb](PyTorch_2.ipynb)








# 教AI识别一个“甜甜圈”（基础神经网络）



[PyTorch_3.md](PyTorch_3.md)

[PyTorch_3.ipynb](PyTorch_3.ipynb)





# 教AI识别手写数字（卷积神经网络）



[PyTorch_4.md](PyTorch_4.md)

[PyTorch_4_1.ipynb](PyTorch_4_1.ipynb)

[PyTorch_4_2.ipynb](PyTorch_4_2.ipynb)






# 使用FastAPI将Fashion-MNIST模型部署成一个功能齐全的Web API服务






[PyTorch_5.md](PyTorch_5.md)

[PyTorch_5_1.ipynb](PyTorch_5_1.ipynb)

[PyTorch_5_2.ipynb](PyTorch_5_2.ipynb)

[PyTorch_5_3.ipynb](PyTorch_5_3.ipynb)

[fashion_mnist_deploy](fashion_mnist_deploy)






# Transformer学习

[Transformer_Learning.md](Transformer_Learning.md)

[Transformer_Learning.ipynb](Transformer_Learning.ipynb)




# 通过手动删除Jupyter Notebook的metadata.widgets，解决了在 GitHub 上无法渲染和查看 Notebook 的问题。

[Remove-Widgets.md](Remove-Widgets.md)

# LLM-Learning



[LLM_Learning_1.ipynb](LLM_Learning_1.ipynb)学会了如何手动调用预训练好的语言模型。

[LLM_Learning_2.ipynb](LLM_Learning_2.ipynb)学习了对预训练模型进行微调 (Fine-tuning) 。

[LLM_Learning_3.ipynb](LLM_Learning_3.ipynb)学习了保存、加载、分享模型。


[LLM_Learning_4.ipynb](LLM_Learning_4.ipynb)学习了生成式AI，三种解码策略，Decoder-only和Encoder-Decoder两种架构的区别，实践了对话和摘要式应用。



[LLM_Learning_5.ipynb](LLM_Learning_5.ipynb)学习了LoRA高效微调。


[Chat_嬛嬛.ipynb](Chat_嬛嬛.ipynb)学习了开源项目self-llm的Chat-嬛嬛项目，并在colab中进行实践微调，但是colab免费T4 GPU显存不够用。


[Chat_嬛嬛_4位量化.ipynb](Chat_嬛嬛_4位量化.ipynb)使用4位量化技术+LoRA进行了微调实践，成功在colab免费T4 GPU上运行。


[Chat_嬛嬛_微调并合并推送.ipynb](Chat_嬛嬛_微调并合并推送.ipynb)使用L4 GPU成功进行了微调，并将训练好的LoRA“插件”和Llama3基础模型进行合并为“Chat-嬛嬛”模型，推送至Hugging Face。

[Chat_嬛嬛_拉取检验.ipynb](Chat_嬛嬛_拉取检验.ipynb)从Hub仓库拉取微调模型进行检验。

微调模型地址：https://huggingface.co/16Miku/Chat-HuanHuan-Llama3-8B-merged


[LLM_Learning_6.ipynb](LLM_Learning_6.ipynb)学习了LoRA模型的合并、保存与加载，但还没有运行代码实践。


[LLM_Learning_6_1.ipynb](LLM_Learning_6_1.ipynb)使用T4 GPU 进行微调，训练了两个多小时，训练完放着没及时后续操作，断开了。



[LLM_Learning_7.ipynb](LLM_Learning_7.ipynb)学习了使用经典评估指标 ROUGE 进行模型评估。

[LLM_Learning_8.ipynb](LLM_Learning_8.ipynb)学习了搭建基于关键词匹配的简易RAG系统。


[LLM_Learning_9.ipynb](LLM_Learning_9.ipynb)学习了多模态大模型的原理，并实践运行LLaVA模型。


[LLM_Learning_10.ipynb](LLM_Learning_10.ipynb)学习了基于 LangChain + Gemini 的简易 ReAct Agent。







[LLM_Learning_11.ipynb](LLM_Learning_11.ipynb)学习了Flash Attention优化。



[LLM_Learning_12.ipynb](LLM_Learning_12.ipynb)学习了训练后量化PTQ，但是运行GPTQ代码一直不成功。学习了使用 unsloth 工具进行一键式优化，并将用 unsloth 量化后的模型以两种格式分别保存并推送到我的Huggingface仓库。


原生的Hugging Face格式：
https://huggingface.co/16Miku/llama-3-8b-instruct-unsloth-4bit

GGUF格式：
https://huggingface.co/16Miku/llama-3-8b-instruct-unsloth-gguf




[LLM_Learning_13.ipynb](LLM_Learning_13.ipynb)学习了使用 LangChain 框架构建 RAG 系统。









