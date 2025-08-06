# 🚀 预训练技巧

## 🎯 预训练概述
预训练是大模型能力的基石，涉及大规模数据、分布式训练、优化策略等关键技术。

## ⚙️ 核心训练技术

### 1️⃣ 混合精度训练
- **原理**：FP16 + FP32混合计算
- **优点**：内存减半、速度提升
- **实现**：NVIDIA Apex、PyTorch AMP

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
for data, target in dataloader:
    optimizer.zero_grad()
    
    with autocast():
        output = model(data)
        loss = criterion(output, target)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
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