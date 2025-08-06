# 🗜️ 模型压缩与量化

## 🎯 概述
模型压缩通过减少模型大小和计算量，使大模型能够在资源受限的环境中部署。

## 🏗️ 压缩技术

### 1️⃣ 权重量化
- **INT8量化**：将FP32权重压缩到INT8，4倍压缩
- **INT4量化**：进一步压缩到4位，8倍压缩
- **GPTQ**：基于二阶信息的量化方法

### 2️⃣ 激活量化
- **动态量化**：运行时量化激活值
- **静态量化**：校准数据集预计算量化参数
- **SmoothQuant**：解决激活异常值问题

### 3️⃣ 稀疏化
- **非结构化稀疏**：随机权重置零
- **结构化稀疏**：通道/块级稀疏
- **N:M稀疏**：每M个权重保留N个

### 4️⃣ 知识蒸馏
- **量化感知蒸馏**：结合量化和蒸馏
- **渐进式量化**：逐步降低精度

## 📊 量化方法对比
| 方法 | 压缩比 | 精度损失 | 推理速度 | 实现难度 |
|---|---|---|---|---|
| **INT8** | 4x | <1% | 2-3x | 低 |
| **INT4** | 8x | 1-3% | 3-4x | 中 |
| **GPTQ** | 8x | <1% | 3-4x | 中 |
| **AWQ** | 8x | <0.5% | 3-4x | 中 |

## 🎯 实战代码
```python
# 使用bitsandbytes进行量化
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# INT4量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# 加载量化模型
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)
```
## 剪枝

## 🎯 面试重点
1. **INT8和INT4量化的区别？**
2. **如何解决量化后的精度损失？**
3. **GPTQ和AWQ的算法原理？**
4. **量化对推理速度的影响？**