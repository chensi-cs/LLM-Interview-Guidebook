# 🫗 知识蒸馏

## 🎯 概述
知识蒸馏通过教师-学生框架，将大模型的知识迁移到小模型，实现性能与效率的平衡。

## 🏗️ 蒸馏类型

### 1️⃣ 响应蒸馏 (Response Distillation)
- **原理**：匹配教师和学生模型的输出logits
- **损失函数**：$\mathcal{L}_{KD} = \alpha T^2 \cdot \text{KL}(p_T^T || p_S^T) + (1-\alpha) \cdot \text{CE}(y, p_S)$
- **温度T**：控制softmax平滑度

### 2️⃣ 特征蒸馏 (Feature Distillation)
- **原理**：匹配中间层特征表示
- **方法**：Hinton Loss、FitNets、Attention Transfer
- **优势**：保留更多语义信息

### 3️⃣ 关系蒸馏 (Relational Distillation)
- **原理**：匹配样本间关系
- **方法**：RKD、CRD、Relational Knowledge Distillation

## 📊 蒸馏策略对比
| 类型 | 信息来源 | 效果 | 计算成本 | 适用场景 |
|---|---|---|---|---|
| **响应蒸馏** | 输出层 | 中 | 低 | 快速部署 |
| **特征蒸馏** | 中间层 | 好 | 中 | 精度优先 |
| **关系蒸馏** | 样本关系 | 很好 | 高 | 复杂任务 |

## 🎯 实战案例
```python
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

## 🎯 面试重点
1. **知识蒸馏的三个层次？**
2. **温度参数的作用？**
3. **如何选择蒸馏目标？**
4. **蒸馏与剪枝、量化的区别？**