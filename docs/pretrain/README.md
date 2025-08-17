# 🚀 预训练技巧

## 🎯 预训练概述
预训练是大模型能力的基石，涉及大规模数据、分布式训练、优化策略等关键技术。

## ⚙️ 核心训练技术

### 1️⃣ 混合精度训练

混合精度训练是现代深度学习训练中的关键技术，它通过在不同计算环节使用不同精度（fp32, fp16, bf16）的数值表示来加速训练并减少内存占用。

#### 为什么需要混合精度？

深度学习模型训练默认使用 32 位浮点数（FP32） 进行计算和参数存储，但实践中发现：

- 计算效率：FP16（16 位浮点数）或 BF16（脑浮点数）的计算速度比 FP32 快 2-8 倍（尤其在支持 CUDA 的 GPU 上，如 NVIDIA 的 Tensor Core 专门优化低精度计算）。
- 内存占用：低精度数据类型的内存占用仅为 FP32 的 1/2（FP16/BF16），可支持更大的 batch size、更深的模型或更高分辨率的输入。
  
- 精度冗余：模型参数和计算过程中存在精度冗余，并非所有操作都需要 FP32 精度才能保持模型性能。

混合精度训练的核心是 **“按需分配精度”**：对精度敏感的操作（如参数更新、损失计算）保留高精度（FP32），对精度不敏感的计算（如卷积、矩阵乘法）使用低精度（FP16/BF16），兼顾效率与精度。

#### 混合精度训练中各个阶段的参数精度

1. **模型初始化：** 模型权重以 FP32 形式存储，保证权重的精确性。
2. **前向传播阶段：** 前向传播时，会复制一份 FP32 格式的权重并强制转化为 FP16 格式进行计算，利用 FP16 计算速度快和显存占用少的优势加速运算。
3. **损失计算阶段：** 通常与前向传播一致，使用 FP16 精度计算损失
4. **损失缩放阶段：** FP16 精度 。在反向传播前，由于反向传播采用 FP16 格式计算梯度，而损失值可能很小，容易出现数值稳定性问题（如梯度下溢），所以引入损失缩放。将损失值乘以一个缩放因子，把可能下溢的数值提升到 FP16 可以表示的范围，确保梯度在 FP16 精度下能被有效表示。
5. **反向传播阶段：** 计算权重的梯度（FP16 精度），以加快计算速度。
6. **权重更新阶段：** 先将FP16 梯度反缩放（除以缩放因子，恢复原始幅值），此时梯度仍为 FP16，然后将其转换为 FP32​​ ，用于优化器更新，然后用FP32​的梯度（AdamW的FP32​的一阶矩和二阶矩）更新 FP32 的权重


```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
for data, target in dataloader:
    optimizer.zero_grad()
    with autocast():
        output = model(data)
        loss = criterion(output, target)
    # 反向传播和梯度缩放
    scaler.scale(loss).backward()
    #反缩放（因为梯度裁剪需要在原始梯度上进行）
    scaler.unscale_(optimizer)
    # 梯度裁剪（使用原始梯度值，如果有的话）
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    # 执行优化器步骤（scaler会自动处理缩放状态）
    scaler.step(optimizer)
    # 更新缩放因子
    scaler.update()
```

### 2️⃣ 分布式训练策略

#### 数据并行 (Data Parallel)
- **原理**：复制模型到多GPU，分割数据
- **实现**：`torch.nn.DataParallel`

#### 模型并行 (Model Parallel)
- **原理**：分割模型到多GPU
- **适用**：超大模型

#### 流水线并行 (Pipeline Parallel)
- **原理**：分层并行处理
- **实现**：GPipe、PipeDream

### 3️⃣ DeepSpeed优化

#### ZeRO优化器
- **ZeRO-1**：优化器状态分片
- **ZeRO-2**：梯度分片
- **ZeRO-3**：参数分片
- **ZeRO-Offload**：CPU/GPU混合

```python
import deepspeed

model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config_params=ds_config
)
```

### 4️⃣ FlashAttention优化
- **原理**：IO感知的精确注意力
- **优势**：内存高效、速度快
- **实现**：FlashAttention-2

### 5️⃣ 学习率调度

#### Warmup策略
```python
from transformers import get_linear_schedule_with_warmup

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=1000,
    num_training_steps=10000
)
```

#### 余弦退火
```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=1000
)
```


### 梯度累积

### 梯度裁剪

### 优化器选择

## 📊 训练配置对比

| 技术 | 内存节省 | 速度提升 | 实现复杂度 |
|---|---|---|---|
| **混合精度** | 50% | 1.5-2x | 低 |
| **ZeRO-1** | 4x | 轻微 | 中 |
| **ZeRO-2** | 8x | 轻微 | 中 |
| **FlashAttention** | 7.6x | 2-4x | 中 |

## 🎯 面试重点
1. **ZeRO各阶段的区别？**
2. **FlashAttention如何优化内存？**
3. **warmup的作用是什么？**
4. **如何选择并行策略？**