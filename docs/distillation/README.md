# 🫗 知识蒸馏

## 🎯 概述

知识蒸馏（Knowledge Distillation，KD） 是一种模型压缩技术，旨在将一个大型、复杂的模型（通常称为 教师模型）的知识迁移到一个较小、更高效的模型（称为 学生模型）上。其核心思想是，学生模型通过模仿教师模型的输出或中间表示，学习到教师模型的知识，达到接近甚至超越教师模型性能的效果。

## 🤔 为什么需要知识蒸馏？

- **模型压缩：** 大模型通常参数量巨大，占用大量计算资源。知识蒸馏可以将大模型的知识浓缩到一个小模型中，从而降低模型的存储和计算成本。
- **加速推理：** 小模型的推理速度比大模型快得多，这在实时应用中非常重要。
- **提高泛化能力：** 通过学习大模型的知识，小模型可以更好地泛化到未见过的数据上。


## 🏗️ 常见模型蒸馏策略

传统模型的训练是直接通过真实数据的 **硬标签** 计算损失的，硬标签即离散的、确定的类别标签，通常直接对应数据的真实类别或目标输出


### 经典蒸馏方法（Hinton 方法）

#### 训练过程

- 使用传统方法训练一个高性能的教师模型
- 设计相对简单的学生模型结构，具体训练方式如下：
  - 输入样本 x 到教师模型，得到教师模型的**软标签**输出 $p_t$ (软标签指教师模型对样本的输出概率分布，是原始输出logits的归一化值)
  - 输入同样的样本 x 到学生模型，得到学生模型的**软标签**输出 $p_s$
  - 定义损失函数，让学生的输出概率分布 p 尽可能接近教师模型的输出概率分布 q，同时考虑样本x的真实硬标签 y

#### 损失函数

总损失为软损失与硬损失的加权和：$ L = \alpha \times L_{soft} + (1-\alpha) \times L_{hard} $
- $\alpha$：权重系数
- $L_{soft}$：软损失，即学生模型输出 $p_s$ 与教师模型软标签 $p_t$ 之间KL散度（衡量分布差异，保证对教师模型的拟合），$L_{soft} = KL( p_t || p_s)$
- $L_{hard}$：硬损失，即学生模型输出与真实硬标签之间的交叉熵损失（保证对真实类别的拟合）

#### KL散度计算
在机器学习、信息论和概率统计中，KL 散度（Kullback-Leibler Divergence） 是一种衡量两个概率分布之间差异的指标，也被称为相对熵（Relative Entropy）

给定两个概率分布：
- 真实分布 P（ground truth 或 target）
- 近似分布 Q（model output 或 prediction）

KL 散度用来衡量“ 用近似分布 Q 来表示真实分布 P 时所损失的信息量”， 定义如下：

-   离散概率分布：$ KL ( P || Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)} $
-   连续概率分布：$ KL ( P || Q) = \int_x P(x) \log \frac{P(x)}{Q(x)} dx $

KL散度具有以下性质：
-   非负性：$KL(P||Q) \geq 0$，当前仅当对所有x , $ P(x) = Q(x)$ 时，$KL(P||Q) = 0$
-   不对称性：$KL(P||Q) \neq KL(Q||P)$, 意味着 “用 Q 近似 P” 和 “用  P 近似 Q” 的损失可能不同

#### 带温度的软标签

教师模型对输入 x 的 logits（未归一化输出）为 $z_t$，，通过带温度的 softmax 生成软标签 $ p_t = softmax(z_t / T)$，其中 T 是温度参数（ T>1 时分布更平滑，保留更多类别关联信息）。

🤔：为什么要加温度？
↩️：如果不加温度，当某个类别的得分 $z_i$ 原大于其他类别的得分时，softmax归一化后， $p_i$ 接近1，而其他类别的概率接近0，失去了其他类别的概率分布信息，因此引入一个温度参数 T，使输出分布更平滑，保留更多类别关联信息。


### 特征蒸馏（Feature Distillation）
经典蒸馏仅利用教师的输出层知识，而中间层特征（Transformer 隐藏状态、注意力权重）包含丰富的语义信息。此类方法通过对齐师生中间层特征传递知识。

实现方式：

- 选择关键特征层：从教师模型中选取对任务最关键的中间层，通常是语义信息丰富的深层

- 特征映射与对齐：由于学生模型结构可能与教师不同（如层数、维度更少），需设计 “映射函数”（如线性变换）将学生的特征转换为与教师特征兼容的维度。

- 定义特征损失：通过损失函数衡量师生特征的差异，常用MSE 损失（均方误差）或余弦相似度损失


## 🎯 蒸馏案例

```python

import torch.nn.functional as F
class DistillationLoss(nn.Module):
    def __init__(self, temperature=4.0, alpha=0.7):
        super().__init__()
        self.T = temperature
        self.alpha = alpha
        
    def forward(self, student_logits, teacher_logits, labels):
        # 蒸馏损失
        soft_targets = F.softmax(teacher_logits / self.T, dim=-1)
        soft_pred = F.log_softmax(student_logits / self.T, dim=-1)
        distill_loss = F.kl_div(soft_pred, soft_targets, reduction='batchmean') * (self.T ** 2)
        
        # 学生损失
        student_loss = F.cross_entropy(student_logits, labels)
        
        return self.alpha * distill_loss + (1 - self.alpha) * student_loss
```
### F.kl_div 介绍
假设 P 为目标分布，Q为正式分布（假设为离散型）
因为 $  KL ( P || Q)  = \sum_x P(x) \log \frac{P(x)}{Q(x)} = \sum_x P(x)\log{P(x)} -  P(x)\log{Q(x)} = \sum_x P(x) ( \log{P(x)} -\log{Q(x)} ) $ 

F.kl_div内部的KL散度的实现逻辑是 $  KL(input,target) = \sum target \times (log(target) - input )$
所以输入给F.kl_div的input实际是已经经过log和softmax的logits，而输入给 F.kl_div的target只是softmax过的logits

## 蒸馏分类
### 黑盒蒸馏
黑盒蒸馏中，学生模型无法访问教师模型的内部结构或中间输出，仅能获取教师模型的最终预测结果（如输入样本对应的输出概率分布），通过模仿这些 “外部输出” 来学习知识，蒸馏主要依赖教师的最终输出（软标签）作为蒸馏信号，损失函数通常仅基于输出概率的匹配（如 KL 散度损失）。

### 白盒蒸馏
白盒蒸馏中，学生模型可以直接访问教师模型的内部结构和中间输出（如隐藏层特征、注意力权重、logits 等），并通过模仿这些内部信息来学习教师的知识

## 🎯 面试重点
1. **知识蒸馏的三个层次？**
2. **温度参数的作用？**
3. **如何选择蒸馏目标？**
4. **蒸馏与剪枝、量化的区别？**