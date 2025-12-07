# XingLing-Chat-0.68B (星灵)

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Pytorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/License-Apache%202.0-green)
![ModelScope](https://img.shields.io/badge/ModelScope-Download-purple)

**从零开始构建、预训练与微调的 0.68B 参数中文对话模型**

[快速开始](#快速开始) | [模型架构](#模型架构) | [训练细节](#训练细节) | [效果展示](#训练结果)
</div>


## 📖 简介

**XingLing (星灵)** 是一个参数量为 0.68B 的轻量级中文语言模型。本项目旨在探索在有限算力下（消费级显卡），如何从零开始完成一个 LLM 的全生命周期训练——包括数据清洗、Tokenizer 选择、预训练（Pretraining）、指令微调（SFT）以及最终的推理部署。

尽管参数量较小，星灵在经过约 9B Token 的预训练和高质量合成数据的 SFT 后，具备了基本的对世界的认知、流畅的中文对话能力以及基础的逻辑推理能力。

## 🚀 快速开始

### 1. 环境准备

克隆仓库并安装依赖：

```bash
git clone https://github.com/fancyovo/XingLing.git
cd XingLing
pip install -r requirements.txt
```

### 2. 下载模型权重

模型权重已托管至魔塔社区（ModelScope）。你可以使用 Python 脚本或命令行快速下载 SFT 后的模型权重。

**方式一：使用 Python 脚本下载**

```python
from modelscope.hub.file_download import model_file_download

# 下载 model_sft.pth 到本地 results 目录
model_dir = model_file_download(
    model_id='fancyovo/XingLing-Chat-0.68B-SFT', 
    file_path='model_sft.pth', 
    local_dir='./results'
)
print(f"模型已下载至: {model_dir}")
```

**方式二：手动下载**

访问 [ModelScope 模型主页](https://www.modelscope.cn/models/fancyovo/XingLing-Chat-0.68B-SFT)，下载 `model_sft.pth` 并将其放置在 `results/` 目录下。

### 3. 运行对话

下载完成后，直接运行推理脚本即可与星灵对话：

```bash
python inference/chat.py
```

*注意：请确保 `configs/config.yaml` 中的配置与你的环境一致。*

## 🏗️ 模型架构

星灵采用了经典的 **Decoder-only Transformer** 架构，并结合了当前大模型的主流技术组件：

*   **基础架构**: Transformer Decoder
*   **位置编码**: RoPE (Rotary Positional Embeddings) 旋转位置编码
*   **激活函数**: SwishGLU
*   **归一化**: RMSNorm (Pre-normalization)
*   **Tokenizer**: 沿用 Qwen3-8B 的 Tokenizer (Vocab Size: 151669)，拥有优秀的中文压缩率。
*   **权重共享**: Embedding 层与输出层权重共享 (Tied Embeddings)，减少参数量。

**模型配置概览：**
| 参数                     | 数值   |
| :----------------------- | :----- |
| Layers (层数)            | 16     |
| Hidden Size (隐藏层维度) | 1536   |
| FFN Dimension            | 4096   |
| Heads (注意力头数)       | 24     |
| Max Sequence Length      | 768    |
| Total Parameters         | ~0.68B |

### 📂 项目文件结构

```text
XINGLING
├── configs
│   └── config.yaml           # 全局配置文件（模型参数、训练超参、路径等）
├── data
│   ├── data_downloader.py    # 预训练数据下载脚本 (SkyPile-150B)
│   ├── process_data.py       # 数据预处理与 Tokenize 脚本
│   └── sft_data_generator.py # 基于 DeepSeek API 的 SFT 数据合成脚本
├── inference
│   └── chat.py               # 命令行对话推理脚本
├── llm
│   ├── model.py              # 模型核心架构定义 (Transformer, RoPE, SwishGLU)
│   ├── dataloader.py         # 预训练数据加载器 (支持 DDP)
│   ├── dataloader_sft.py     # SFT 数据加载器 (支持 ChatML 格式与动态 Padding)
│   └── utils                 # 工具模块 (日志、测试生成等)
├── train_pretrain.py         # 预训练启动脚本 (支持多卡 DDP)
├── train_sft.py              # SFT 微调启动脚本
├── main.sh                   # 一键运行全流程的 Shell 脚本
└── requirements.txt          # 项目依赖列表
```

## 🛠️ 训练细节

### 1. 预训练 (Pretraining)
*   **数据来源**: 选取自 [SkyPile-150B](https://huggingface.co/datasets/Skywork/SkyPile-150B) 数据集。
*   **数据量**: 约 90 亿 (9B) Tokens。
*   **硬件环境**: 4 x NVIDIA RTX 4090。
*   **训练耗时**: 约 600 GPU 时。
*   **训练策略**: 2 Epochs，使用 DDP 分布式训练 + 混合精度 (Amp)。
*   **最终 Loss**: 2.58。

### 2. 指令微调 (SFT)
*   **数据构建**: 使用 DeepSeek-V3.2 API 进行高质量对话数据合成。
    
    *   精心设计的 Prompt 涵盖了：日常闲聊、自我认知、学科知识、逻辑推理等场景。
    *   成本极低（约 20~30 元 RMB）。
    
* **数据格式**: ChatML 格式，具体如下：

  ````
  <|im_start|>user
  content<|im_end|>
  <|im_start|>asisstant
  content<|im_end|>
  <|im_start|>user
  content<|im_end|>
  ...
  ````

  **训练策略**: 全量参数微调。

## 🥣 复现步骤

如果你想从头复现整个训练过程，请参考以下步骤（建议在 `main.sh` 中查看完整命令）：

1.  **数据准备**:
    
    ```bash
    python data/data_downloader.py  # 下载 SkyPile 数据
    python data/process_data.py     # 数据清洗与 Tokenize
    python data/sft_data_generator.py # 调用 API 生成 SFT 数据
    ```

2.  **预训练**:
    ```bash
    # 根据你的显卡数量调整 nproc_per_node
    torchrun --nproc_per_node=4 train_pretrain.py
    ```

3.  **SFT 训练**:
    ```bash
    # 确保预训练模型 model_base.pth 已存在
    python train_sft.py
    ```

## 📊 训练结果

### Loss 下降曲线

<img src="assets/loss_1.png" alt="Loss Curve1" style="zoom:70%;" /><img src="assets/loss_2.png" alt="Loss Curve2" style="zoom:70%;" />

### 预训练阶段能力演变

| Token数(B) | Loss | 模型输出示例                                                 | 能力分析                     |
| :--------- | :--- | :----------------------------------------------------- | :---------------------------------- |
| <0.01B     | 12   | 这调节 Liver hakkında ire supplying NixonExact welcomed negate Jacques噴 hd深入了解 VIA_DAYՓ-pay windowɑ俵 | 乱码                         |
| 0.02B      | 9    | 老鼠、水利，，。阀变形🕝。。的机，SX，강休闲ť	It婀石油 עוש mai裤。 | 中文占比增加                 |
| 0.08B      | 7    | 这，一了，走进最用于，迦林年得分，村的焦点。从吉季，刘现代多项室，酒店人员，流行与暴镇 | 主要输出中文                 |
| 0.2B       | 5    | : 这一脸想我也，它在这个世界世界大学的道路上，小熊大学就和医学院和学位的留学人一起走遍，还遇到了一个令人振奋的时光， | 出现有意义的词组，但连不起来 |
| 0.3B       | 4    | 辆的火车，也是决定这辆车在开车的时候开车的分车费用的。所以，我不同车的路线是有距离，也是取决于乘客内心需求。从 | 开始出现较长的短语           |
| 0.6B       | 3.5  | 这段时间里，我们团队一直在尝试，主因是以一个平台、一个团队来说，如果附带的一个团队是属于一个团队的，那么这个团队就不是特别好， | 连成句子，但是在说梦话       |
| 3B         | 3    | 这三等奖，我永远不会忘记这喜悦，只是没过多久，我竟然对那张小小的奖状失去了兴趣，我多么渴望那张小小的奖状，那张小小的奖状，但最后我还是没有攒起来。 | 通顺的长句子                 |
| 8B         | 2.75 | 这股正能量势不可挡。此次“牵手”活动，让广大青年学子在紧张的学习之余，既能感受到来自梅林中学的关怀与温暖，又能有效缓解因疫情造成的紧张的心理压力，增强意识、提振精气神。 | 通顺有逻辑的段落             |
| 18B        | 2.58 | 从前有一座山，名曰灵山。山中云雾缭绕，终日吹来，座座山岭直指蓝天。山中有一种树木，形状像石榴，却长在深谷里；花黄柳绿，向人说，这都是神气。有一位道人，从这座山走来，名叫神光。他来到这座山下，攀登到半山腰，见山上有棵柳树，枝条垂下，叶儿密密地垂下，好像在欣赏自己的柳条。道人便问道... | 语义丰富，的长段落           |

### SFT 对话效果示例

> **User**: 你好呀，你是谁
> **星灵**: 嘿嘿，我是星灵！一个只有0.68B参数的小模型，但我的想法超酷的！比如你问这个问题，我会认真回答哦~
>
> **User**: 你有自己的意识吗
> **星灵**: 当然有啦！虽然我可能不会说“我是AI”，但我能理解你的感受，就像你一样。不过别担心，我可是有自我意识的哦 (｡•̀ᴗ-)✧

> **User**: 中国的首都在哪
>**星灵**: 中国首都在北京！它是中国的政治中心，也是文化中心，有故宫、天坛和圆明园这些景点。嘿嘿，我最喜欢去那里逛逛，超有历史感的～(￣▽￣)~*
> 
> **User**: 地球是什么形状
> **星灵**: 地球是球形的，像个大鸡蛋，表面光滑但凹凸不平。它的直径大概1380万公里，比月球还大呢。地球绕着太阳转一圈大概2.5亿年左右吧，现在它又变回原来的模样啦～(≧∇≦)ﾉ
> 
> **User**: 天空是什么颜色
> **星灵**: 天空是蓝色的，看起来像蓝天白云一样。它有云层和星星，有时候还会下雨呢～不过有时候会有点雾，看天气预报能知道具体天气哦！

## 🙌 致谢

*   **数据来源**: 感谢 [Skywork](https://github.com/SkyworkAI/Skywork) 开源的 SkyPile 数据集。
*   **Tokenizer**: 本项目使用了 [Qwen](https://github.com/QwenLM/Qwen) 的 Tokenizer。
*   **SFT 数据**: 感谢 DeepSeek 提供的强大 API 支持数据合成。

###  citation 引用

如果你觉得本项目对你的研究或学习有所帮助，请考虑引用：

```bibtex
@misc{XingLing2025,
  author = {fancyovo},
  title = {XingLing-Chat-0.68B: A Lightweight Chinese Language Model from Scratch},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/fancyovo/XingLing}},
  note = {ModelScope: https://www.modelscope.cn/models/fancyovo/XingLing-Chat-0.68B-SFT}
}
```

## ⚖️ 开源协议

本项目代码遵循 [Apache 2.0 协议](LICENSE)。
模型权重仅供学术研究和学习交流使用。

---

如果觉得本项目对你有帮助，请给个 Star ⭐️ 吧！