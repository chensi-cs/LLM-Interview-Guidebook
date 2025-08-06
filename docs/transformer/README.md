# 📊 Transformer基础结构

## 🎯 概述
Transformer是一种基于注意力机制的神经网络架构，由Vaswani等人在2017年提出，彻底改变了自然语言处理领域。

## 🏗️ 核心组件

### 1️⃣ 编码器-解码器架构
- **编码器**：将输入序列转换为隐藏表示
- **解码器**：基于编码器输出生成目标序列

### 2️⃣ 关键创新
- **自注意力机制**：并行处理序列，捕获长距离依赖
- **位置编码**：为模型提供序列位置信息
- **残差连接**：缓解深层网络训练问题
- **层归一化**：稳定训练过程

## 📋 架构详解

### 编码器结构
每个编码器层包含：
1. **多头自注意力**：计算输入序列内部关系
2. **前馈神经网络**：非线性变换
3. **残差连接和层归一化**

### 解码器结构
每个解码器层包含：
1. **掩码多头自注意力**：防止信息泄露
2. **编码器-解码器注意力**：关注输入序列
3. **前馈神经网络**
4. **残差连接和层归一化**

## 🔍 数学原理

### 缩放点积注意力
$$
\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

### 多头注意力
$$
\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1,...,\text{head}_h)W^O
$$
其中 $\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$

## 🚀 代码示例

```python
import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # 自注意力 + 残差连接
        attn_output, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络 + 残差连接
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x
```

## 📚 深入阅读
- [原始论文：Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [分词器详解](../tokenizer/README.md)
- [注意力机制详解](../attention/README.md)

## 🎯 面试重点
1. **为什么使用多头注意力？**
2. **位置编码的作用是什么？**
3. **残差连接和层归一化的作用？**
4. **Transformer相比RNN的优势？**