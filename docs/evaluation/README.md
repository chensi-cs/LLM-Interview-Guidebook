# 📈 模型评估

## 🎯 概述
模型评估是衡量大模型性能的关键环节，涉及能力评估、安全性评估和效率评估等多个维度。

## 🏗️ 评估维度

### 1️⃣ 基础能力评估
- **语言理解**：GLUE、SuperGLUE
- **知识问答**：MMLU、C-Eval、CMMLU
- **推理能力**：GSM8K、MATH、HumanEval
- **代码能力**：HumanEval、MBPP、CodeContests

### 2️⃣ 对齐评估
- **有用性**：帮助用户完成任务的能力
- **无害性**：避免有害或不当输出
- **诚实性**：承认知识边界，避免幻觉

### 3️⃣ 效率评估
- **推理延迟**：首token延迟、token间延迟
- **吞吐量**：tokens/second
- **资源消耗**：显存使用、功耗

## 📊 评估基准
| 基准 | 评估能力 | 语言 | 样本数 |
|---|---|---|---|
| **MMLU** | 多学科知识 | 英文 | 15,908 |
| **C-Eval** | 中文综合能力 | 中文 | 13,948 |
| **GSM8K** | 数学推理 | 英文 | 8,500 |
| **HumanEval** | 代码生成 | 英文 | 164 |

## 🎯 评估方法
```python
# 使用Hugging Face Evaluate库
import evaluate

# 加载评估指标
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
accuracy = evaluate.load("accuracy")

# 评估示例
predictions = ["Hello world", "How are you"]
references = [["Hello world"], ["How are you today"]]

bleu_score = bleu.compute(predictions=predictions, references=references)
rouge_score = rouge.compute(predictions=predictions, references=references)
```

## 🎯 面试重点
1. **如何评估大模型的幻觉问题？**
2. **MMLU和C-Eval的区别？**
3. **如何设计领域特定的评估指标？**
4. **人工评估vs自动评估的权衡？**