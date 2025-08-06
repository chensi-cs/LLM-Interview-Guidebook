# 🧠 前馈神经网络

## 🎯 概述
前馈网络(FFN)是Transformer中的重要组件，提供非线性变换能力。

## 🏗️ 网络结构

### 1️⃣ 标准FFN
- **结构**：Linear → Activation → Linear
- **公式**：$\text{FFN}(x) = \text{Linear}(\text{Activation}(\text{Linear}(x)))$
- **扩展系数**：通常4倍隐藏维度

### 2️⃣ 混合专家模型 (MoE)
- **原理**：稀疏激活的专家网络
- **特点**：
  - 参数量大但计算高效
  - 动态路由机制
  - 专家并行

## ⚡ 激活函数

### 1️⃣ ReLU
- **公式**：$\text{ReLU}(x) = \max(0, x)$
- **特点**：简单高效，但可能神经元死亡

### 2️⃣ GELU
- **公式**：$\text{GELU}(x) = x \cdot \Phi(x)$
- **特点**：平滑激活，BERT使用

### 3️⃣ SwiGLU
- **公式**：$\text{SwiGLU}(x) = \text{SiLU}(xW) \otimes (xV)$
- **特点**：GLU变体，LLaMA使用

## 📊 结构对比
| 类型 | 参数量 | 计算量 | 表达能力 |
|---|---|---|---|
| **标准FFN** | 少 | 少 | 中 |
| **MoE** | 多 | 中 | 强 |