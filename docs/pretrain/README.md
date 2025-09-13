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
4. **损失缩放阶段：** FP16 精度 。在反向传播前，由于反向传播会采用 FP16 格式计算梯度，而损失值可能很小，容易出现数值稳定性问题（如梯度下溢），所以引入损失缩放。将损失值乘以一个缩放因子，把可能下溢的数值提升到 FP16 可以表示的范围，确保梯度在 FP16 精度下能被有效表示。
5. **反向传播阶段：** 计算权重的梯度（FP16 精度），以加快计算速度。
6. **权重更新阶段：** 先将FP16 梯度反缩放（除以缩放因子，恢复原始幅值），此时梯度仍为 FP16，然后将其转换为 FP32 ，优化器（假设是AdamW，优化器的一阶矩和二阶矩从初始化到更新始终保持 FP32 精度）用FP32的梯度，FP32的一阶矩和二阶矩，更新 FP32 的权重


```python
import torch
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
for data, target in dataloader:
    optimizer.zero_grad()
    with autocast():
        output = model(data)
        loss = criterion(output, target)
    # 反向传播：先缩放损失，再计算梯度（避免 FP16 梯度下溢）
    scaler.scale(loss).backward()
    #反缩放（因为梯度裁剪需要在原始梯度上进行）
    scaler.unscale_(optimizer)
    # 梯度裁剪（可选，防止梯度爆炸）
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
     # 更新参数：用缩放后的梯度更新，内部会自动调整缩放因子
    scaler.step(optimizer)
    # 更新缩放因子
    scaler.update()
```

### 2️⃣ 分布式训练策略

#### 为什么要分布式训练？
- 随着模型规模（如千亿参数大模型）和数据集大小（如 TB 级数据）的爆炸式增长，单设备已无法满足训练需求。
- 大型数据集 / 模型的训练可能需要数周甚至数月，分布式训练通过多设备并行计算，可显著缩短训练时间
- 缺点：分布式训练涉及模型同步、通信开销等问题

主流并行策略有以下三种（3D并行）：数据并行、模型并行、流水线并行

#### 数据并行 (Data Parallel)
- **原理**：将训练数据拆分到多个设备，每个设备保存完整的模型副本，独立计算局部梯度后通过通信同步梯度，最终所有设备用相同的梯度更新模型，保持参数一致。
- **模型更新流程：**
    - **前向传播：**每个 GPU 用自己的分片数据，通过本地模型副本计算输出和损失。
    - **反向传播：**每个 GPU 计算本地梯度（针对自己的分片数据）。
    - **梯度同步：**通过 **“all-reduce”** 通信操作，所有 GPU 交换本地梯度并计算平均值（确保每个 GPU 得到相同的全局梯度）。
    - **参数更新：**每个 GPU 用同步后的全局梯度更新自己的模型副本（因梯度相同，更新后所有副本参数完全一致）。
- **适用场景：**模型较小（单卡可容纳）、数据量大（大规模文本语料）。
- **优势：**实现简单（如 PyTorch 的 DDP，DP），通信成本低（仅需同步梯度）。

#### 模型并行 (Model Parallel)
- **原理**：模型通常用于非常大的模型，这些模型无法完整地放入单个设备的内存中。模型并行具体是指将一个模型的不同部分分配到不同的设备上运行，如**按层拆分（结构拆分）** 和 **张量拆分（维度拆分）**
- **模型更新流程（按层划分）**：
  1. **前向传播**：输入数据先传入第一个设备（如 GPU0），经其负责的模型部分计算后，将中间结果传递给下一个设备（如 GPU1），直至得到最终输出和损失。
  2. **反向传播**：损失的梯度从最后一个设备（如 GPU1）开始反向计算，依次传递给前序设备（如 GPU0），每个设备仅计算自己负责的模型部分的梯度。
  3. **参数更新**：每个设备仅用自己计算的梯度更新本地保存的那部分模型参数（无需全局同步，因每个设备只负责部分参数）。
- **适用场景：**模型极大（单卡无法容纳完整模型），如千亿参数的 LLM。
- **优势：**突破单卡显存限制，可训练超大模型。
- **劣势：**设备间需频繁传递中间结果，通信成本较高（尤其是模型拆分过细时）。

#### 流水线并行 (Pipeline Parallel)

- **原理**：模型并行的 “升级版”，将模型按层拆分为多个 “阶段”（每个阶段包含连续的若干层），每个阶段分配到一个设备，通过 “流水线调度” 让不同设备同时处理不同批次的数据，提高设备利用率。



#### 分布式训练中的通信概念

在分布式训练中，模型和数据被分到不同机器上，每个机器会执行局部计算，然后 **同步和交换信息**，这就是**分布式训练中的通信概念**

常见的通信策略如下：

##### Broadcast（广播）

1对多，一个节点的数据（同样的数据）发送给所有其他节点，常用于初始化模型参数。（发送同一份完整数据给所有节点，所有人收到相同数据）

<img src="https://ask.qcloudimg.com/draft/1215004/qg6ezsg9va.png" style="zoom:80%;" />

##### Scatter（分发）

1对多，一个节点的数据（不同的数据）发送给所有其他节点。（把一份大数据拆分成多块发给各节点

每个节点收到不同的数据块）

<img src="https://cdn.jsdelivr.net/gh/chensi-cs/images/img/202509131538702.png" alt="image.png" style="zoom:80%;" />

##### Gather（汇总）

多对1，把多个节点的数据收集到一个节点上

<img src="https://cdn.jsdelivr.net/gh/chensi-cs/images/img/202509131540849.png" alt="image.png" style="zoom:80%;" />

##### Reduce（归约/聚合）

多对一，多个节点各自发送一份数据到目标节点，目标节点会对这些数据进行**规约操作**（求和，求平均，求最值，逻辑运算等）得到一个最终结果。

<img src="https://cdn.jsdelivr.net/gh/chensi-cs/images/img/202509131545484.png" alt="image.png" style="zoom:80%;" />

##### All-Reduce

All-Reduce = Reduce + Broadcast，多对多，所有节点先将自己的数据进行归约（reduce，如求和/平均），然后将 归约结果广播（broadcast）给所有节点，最终每个节点都得到相同的聚合结果。

All-Reduce（全归约） 是分布式训练中最核心、最常用的通信原语之一，常用于**数据并行中的梯度同步**：

- 每个 GPU 计算本地数据的梯度
- 通过 All-Reduce 汇总求平均
- 所有 GPU 得到相同的平均梯度，然后再各自更新参数

All-Reduce还可以用于统计信息同步，例如训练中的损失、准确率、样本数等

![image.png](https://cdn.jsdelivr.net/gh/chensi-cs/images/img/202509132053192.png)

##### All-Gather

All-Gather=Gather+Broadcast，多对多，从各个节点收集数据，并将收集到的数据广播给所有节点。

![image-20250913155501510](https://cdn.jsdelivr.net/gh/chensi-cs/images/img/202509131555543.png)

##### Reduce-Scatter

 Reduce-Scatter=Reduce+Scatter，多对多：**先做 Reduce（归约）**，把所有节点上的数据进行归约（求和/平均等），**再做 Scatter（分发）**，把归约后的结果按等份切分，分发给所有节点，每个节点得到一部分结果。

> 与 All-Reduce 的区别是：
>
> - All-Reduce：每个节点得到 **完整的聚合结果**
> - Reduce-Scatter：每个节点只得到 **聚合结果的一部分**



##### Ring All-Reduce

Ring All-Reduce是All-Reduce的一种优化方法

**背景：**

All-reduce操作可以从各个显卡上收集数据并进行聚合，再将聚合的结果分发至各个显卡，但具体如何实现最有效，减少通信量是一个问题。**All-reduce的常见实现方法有：**

- **最粗暴的方法：** 每个worker将自己的数据发给其他的所有worker，然后进行规约。假设有n个worker，则需要n(n-1)次传输。

- **主从式架构**：

  - 假设有n个worker，将一个worker设为master，其余所有worker把数据发送给master之后，由master进行规约，完成之后再分发给其余worker，这样只需要进行(n-1)*2次传输

  - Parameter Server（参数服务器）：所有node被分为server node和worker node，server node负责参数的存储和全局的聚合操作，而worker node负责计算，在计算过程中，所有数据会被划分到每个worker node上，每个worker node首先从server node中pull 模型的参数，并利用本地的数据进行计算得到本地的梯度，并将其push到server node上。而server node根据所有worker node的梯度进行求和得到全局的梯度，进行参数的更新，然后worker node再从server node中pull新的参数进行下一轮迭代。

    **缺点：**所有 Worker 节点的梯度推送和参数拉取都需经过 PS 节点，随着 Worker 数量增加，PS 节点的网络带宽会成为瓶颈

**动机：** Parameter Server的网络带宽瓶颈主要是由PS节点的中心化导致的，Ring All-Reduce 的思想是“去中心化”

**思想：** 将所有设备安排在一个逻辑环中，每个GPU应该有一个左邻和一个右邻，设备只会给它的右邻居发送数据，并从它的左邻居接收数据，整个计算主要包含两个过程：

- **Scatter-Reduce**：N个GPU，每个 GPU 把第 i 块发送给下一个 GPU，并接收上一个 GPU 发来的第 i 块，重复 N-1 次，每个 GPU 拿到自己负责的 **1 块聚合结果**
- **All-gather**：每个 GPU 依次把自己那块聚合结果传给下一个 GPU，重复直到每个 GPU 都得到完整的所有块，类似Scatter-Reduce，但操作由规约变为重写。
- 最终每个 GPU 都拥有归约后的完整向量。

 下面是一个Ring All-Reduce示意图，来自[Ring AllReduce简介](https://mp.weixin.qq.com/s/K8l7H2zCUr9sGzYehizFMA)

 **Scatter-Reduce:**

<img src="https://cdn.jsdelivr.net/gh/chensi-cs/images/img/202509132053937.webp" alt="图片" style="zoom:80%;" />

<img src="https://cdn.jsdelivr.net/gh/chensi-cs/images/img/202509132053521.webp" alt="图片" style="zoom:80%;" />

<img src="https://cdn.jsdelivr.net/gh/chensi-cs/images/img/202509131719055.webp" alt="图片" style="zoom:80%;" />

<img src="https://cdn.jsdelivr.net/gh/chensi-cs/images/img/202509131719410.webp" alt="图片" style="zoom:80%;" />

 **ALL-Gather:**

<img src="https://cdn.jsdelivr.net/gh/chensi-cs/images/img/202509131720826.webp" alt="图片" style="zoom:80%;" />

<img src="https://cdn.jsdelivr.net/gh/chensi-cs/images/img/202509131720561.webp" alt="图片" style="zoom:80%;" />

 **Ring Allreduce的通信成本**：

我们来计算下整个All-reduce过程的通信成本。假设我们有N个设备，总数据大小为K，在一次Allreduce过程中，我们进行了N-1次Scatter-Reduce操作和N-1次Allgather操作，每一次操作所需要传递的数据大小为K/N，所以整个Allreduce过程所传输的数据大小为2(N-1) * K/N (< 2K)，而整个Allreduce的通信速度只受限于逻辑环中最慢的两个GPU的连接

##### 通信库与工具

- **NCCL**：NVIDIA 的 GPU 间通信库，优化 All-Reduce、All-Gather 等操作，支持多机多卡。



### 3️⃣ DeepSpeed

DeepSpeed 是由微软开发的一个开源深度学习优化库，专注于大规模深度学习模型的训练优化与分布式并行策略，核心技术包括**ZeRO 冗余优化技术**，**3D 并行训练框架**，**混合精度训练**，**推理优化**等

其中**ZeRO 冗余优化技术**是DeepSpeed的一大创新，可用来减少分布式训练中的显存冗余。在介绍ZeRO之前，先介绍一下大模型训练的显存占用。

#### 大模型训练的显存占用

**1）有关模型的显存占用：**

- **Parameters：**模型参数
- **Gradients：**模型梯度，用于参数更新
- **Optimizer States：**优化器维护的一些附加信息或变量（如Adam中的momentum和variance）用于辅助参数更新，以及模型参数和梯度的副本。

**2）其余显存占用：**

- **Activations ：**激活值，模型前向传播过程中会计算并存储每一层的激活值，这些激活值在后向传播时被用来计算梯度
- **Temporary buffers ：**临时缓冲区，临时存储各种中间结果，计算临时数据
- **Memory Fragmentation  ：**显存碎片，如果没有足够的连续显存满足显存请求，则对显存的请求将失败，即使总可用显存大于请求

假设使用Adam优化器对参数量为 Φ 的模型进行混合精度训练，显存占用如下：

- 使用较低精度前向传播和反向传播计算梯度，包括FP16的模型参数和梯度（2Φ+2Φ）
- 优化器使用较高精度更新参数，包括FP32的模型参数副本，FP32的动量和方差，需要4Φ+4Φ+4Φ=12Φ字节 ，令 K=12 表示优化器中的显存是用
- 总共需要2Φ+2Φ+KΦ=16Φ字节 ，对1.5B参数的,模型，需要1.5*16=24的显存空间
- 注：论文中的显存计算说法如上，但可能还会有一个FP32的梯度累积，这里没有计入这种情况
- 有关模型的显存占用是ZeRO的重点优化对象，其中优化器状态显存占比 75% ，是第一个要被优化的。



#### ZeRO冗余优化

参考论文《ZeRO: Memory Optimizations Toward Training Trillion Parameter Models》【2020】

ZeRO（**Ze**ro **R**edundancy **O**ptimizer） 通过减少显存冗余和优化资源分配，可支持超大规模模型的训练。

**背景：**数据并行是分布式训练的一种简单有效的方式，但需要在每个机器上都复制一份模型状态（包括模型参数，梯度，优化器状态），这会引入大量显存冗余，并且这些模型状态并不是随时都有用的。

**动机：**ZeRO 通过对优化器状态、梯度和参数进行“分区”而非“直接复制”所有模型状态的方式，消除数据并行中的由模型状态产生的显存冗余

**方法：**ZeRO 的分区主要分为三个阶段，ZeRO-1 分区优化器状态，ZeRO-2 分区优化器状态和梯度，ZeRO-3 对优化器状态、梯度和模型参数都进行分区。

| ZeRO Stage | 主要优化点                                        |
| ---------- | ------------------------------------------------- |
| **ZeRO-1** | 分区优化器状态，每个 GPU 仍保存全量模型参数和梯度 |
| **ZeRO-2** | 分区优化器状态和梯度，每个 GPU 仍保存全量模型参数 |
| **ZeRO-3** | 分区优化器状态、梯度和模型参数                    |

下图是ZeRO论文中对分区阶段的说明和示意图：

假设 DP degree 为  $N_d $，即有 $N_d $个GPU，

- **ZeRO-1**：分区优化器状态，每个 GPU 仍保存全部的模型参数和梯度，但每个GPU只有$1/ N_d $的优化器状态。（下图中的$P_{os}$ ）

  由于模型参数和梯度在每个GPU上是完整的，反向传播计算完梯度后，各个GPU通过`All-Reduce`操作同步梯度，每个GPU都得到了平均梯度，用于更新参数。但由于优化器状态分区，第 i 个GPU 仅对应于第 i 个分区的优化器状态，因此仅更新这部分参数（$1/ N_d $），所有GPU更新完成后，执行一次 `All-Gather` 操作，使得所有GPU中都拥有完全更新的参数。通信开销与传统数据并行相当。

- **ZeRO-2**：分区优化器状态和梯度，每个 GPU 仍保存全量模型参数，但每个GPU只有$1/ N_d $的优化器状态和梯度。（下图中的 $P_{os+g}$ ）。由于梯度分区，反向传播时，每个 GPU 只计算本地梯度，在 `backward` 结束后，进行 `All-Reduce` 把每个 GPU的梯度进行规约得到平均梯度，然后再分发到所有 GPU。由于优化器状态分区， i 个GPU 仅对应于第 i 个分区的优化器状态，因此仅更新这部分参数，所有GPU更新完成后，执行一次 `All-Gather` 操作，使得所有GPU中都拥有完全更新的参数。因此ZeRO-2相比ZeRO-1多了一个梯度的All-Reduce操作，通信总量和传统数据并行接近
- **ZeRO-2**：分区优化器状态、梯度和模型参数，每个GPU只有$1/ N_d $的（优化器状态+梯度+参数）。（下图中的  $P_{os+g+p}$），此时每张卡的模型状态所需显存是 16Φ/N字节（参照下图），当 N 比较大时，趋向于 0 （论文中：允许每个设备的内存使用量随数据并行度线性扩展）。每个 GPU 只存储与其分区对应的参数，因此在前向传播和后向传播过程中需要频繁地从其他 GPU 那里接收必要的模型参数，会产生额外的通信开销。如前向传播前需要`All-Gather` GPU上的参数，反向传播计算梯度后，立刻 **Reduce-Scatter** 把梯度分给对应持有参数 的 GPU，优化器在本地GPU分区上完成更新，然后循环。

![image-20250913152102884](https://cdn.jsdelivr.net/gh/chensi-cs/images/img/202509132053184.png)

#### ZeRO-Offload

除了以上的ZeRO的三个阶段，Deepspeed的**ZeRO-Offload**还支持将数据和计算卸载到CPU或NVMe来实现大型模型训练，降低 GPU 显存占用。具体来说，**ZeRO-Offload**将训练过程抽象为一个Data-Flow Graph，将fp16参数保存在GPU上，并在GPU上进行前向和后向计算，所有fp32模型状态以及fp16梯度分配在CPU上，并且在CPU上计算参数更新。下图是ZeRO-Offload的示意图：

<img src="https://cdn.jsdelivr.net/gh/chensi-cs/images/img/202509132053604.png" alt="image-20250913201821662" style="zoom: 67%;" />



### 4️⃣ FlashAttention优化

- **原理**：IO感知的精确注意力
- **优势**：内存高效、速度快
- **实现**：FlashAttention-2

### 5️⃣ 学习率选择

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



## 🎯 面试重点

1. **ZeRO各阶段的区别？**
2. **FlashAttention如何优化内存？**
3. **warmup的作用是什么？**
4. **如何选择并行策略？**
