# ⚖️ 归一化技术

## 🎯 概述
归一化技术在大模型中起到稳定训练、加速收敛的关键作用。

## 🏗️ 归一化方法

### 1️⃣ LayerNorm
- **原理**：对特征维度归一化
- **公式**：$\text{LayerNorm}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$
- **应用**：Transformer标准配置

### 2️⃣ RMSNorm
- **原理**：去除均值计算，仅使用方差
- **公式**：$\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \cdot \gamma$
- **优点**：计算更高效
- **应用**：LLaMA、RWKV

### 3️⃣ Pre-norm vs Post-norm
- **Pre-norm**：归一化在残差连接前
- **Post-norm**：归一化在残差连接后
- **趋势**：现代模型倾向Pre-norm

## 📊 对比分析
| 方法 | 计算量 | 稳定性 | 现代应用 |
|---|---|---|---|
| **LayerNorm** | 高 | 高 | 标准Transformer |
| **RMSNorm** | 中 | 高 | LLaMA、RWKV |
| **Pre-norm** | - | 更高 | 现代架构 |