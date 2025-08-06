# 🎲 解码策略

## 🎯 概述
解码策略决定模型如何从概率分布中生成文本，平衡创造性和准确性。

## 🏗️ 解码方法

### 1️⃣ 贪婪解码
- **原理**：每一步选择概率最高的词
- **特点**：确定性、重复性高
- **代码**：
```python
def greedy_decode(model, input_ids, max_length):
    for _ in range(max_length):
        outputs = model(input_ids)
        next_token = outputs.logits[:, -1, :].argmax(dim=-1)
        input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
    return input_ids
```

### 2️⃣ Beam Search
- **原理**：保留top-k个候选序列
- **参数**：beam width
- **平衡**：质量vs多样性

### 3️⃣ 随机采样

#### Temperature Sampling
- **公式**：$P(w_i) = \frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}$
- **温度T**：控制随机性

#### Top-k Sampling
- **原理**：只考虑概率最高的k个词
- **优点**：减少低概率词的影响

#### Top-p (Nucleus) Sampling
- **原理**：累积概率达到p的最小词集
- **优点**：动态调整候选词数量

## 📊 解码策略对比
| 方法 | 多样性 | 质量 | 计算成本 | 适用场景 |
|---|---|---|---|---|
| **贪婪** | 低 | 中 | 低 | 确定性任务 |
| **Beam** | 中 | 高 | 中 | 翻译、摘要 |
| **Top-p** | 高 | 高 | 低 | 创意写作 |