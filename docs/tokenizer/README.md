# 🔤 分词器详解

## 🎯 概述
分词器(Tokenizers)是将文本转换为模型可理解的数字序列的关键组件，直接影响模型的性能和效率。

## 🏗️ 主流分词算法

### 1️⃣ BPE (Byte Pair Encoding)

**原理**：通过合并高频字符对来构建词汇表

**优点**：
- 有效处理未登录词
- 词汇量可控
- 多语言支持好

**缺点**：
- 可能产生不完整的词
- 对中文支持有限

**实现示例**：
```python
from tokenizers import Tokenizer
from tokenizers.models import BPE

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
```

### 2️⃣ WordPiece

**原理**：基于最大似然估计逐步合并词片段

**特点**：
- Google开发，用于BERT
- 在词前添加`##`标记子词
- 更适合英文

**示例**：
```
"playing" -> ["play", "##ing"]
```

### 3️⃣ SentencePiece

**原理**：将文本视为Unicode序列，不依赖空格分词

**优势**：
- 语言无关性
- 支持中文、日文等无空格语言
- 可逆转换

**配置示例**：
```python
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input='input.txt',
    model_prefix='tokenizer',
    vocab_size=32000,
    model_type='bpe'
)
```

## 📊 算法对比

| 特性 | BPE | WordPiece | SentencePiece |
|---|---|---|---|
| **分词粒度** | 子词 | 子词 | 子词/字符 |
| **语言支持** | 英文为主 | 英文为主 | 多语言 |
| **空格处理** | 依赖空格 | 依赖空格 | 不依赖空格 |
| **训练速度** | 快 | 中等 | 慢 |
| **模型大小** | 小 | 中等 | 大 |

## 🎯 实战应用

### 中文分词最佳实践
```python
# 使用SentencePiece处理中文
import sentencepiece as spm

# 训练中文分词器
spm.SentencePieceTrainer.train(
    input='chinese_corpus.txt',
    model_prefix='chinese_sp',
    vocab_size=32000,
    character_coverage=0.995,  # 覆盖99.5%字符
    model_type='bpe'
)

# 使用分词器
sp = spm.SentencePieceProcessor(model_file='chinese_sp.model')
tokens = sp.encode('大模型面试宝典', out_type=str)
print(tokens)  # ['大', '模型', '面试', '宝典']
```

### 英文分词示例
```python
# 使用Hugging Face Tokenizers
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
tokens = tokenizer.tokenize("transformer architecture")
print(tokens)  # ['transform', '##er', 'arch', '##itecture']
```

## 🔍 技术细节

### 词汇表构建流程
1. **预处理**：清洗文本，标准化
2. **训练**：基于语料库学习分词规则
3. **验证**：检查分词质量
4. **优化**：调整超参数

### 特殊标记处理
- `[PAD]`：填充标记
- `[UNK]`：未知词标记
- `[CLS]`：分类标记
- `[SEP]`：分隔标记
- `[MASK]`：掩码标记（用于MLM）

## 📚 深入阅读
- [注意力机制详解](../attention/README.md)
- [主流大模型结构](../models/README.md)

## 🎯 面试重点
1. **BPE和WordPiece的区别？**
2. **如何处理中文分词？**
3. **词汇表大小如何选择？**
4. **OOV(未登录词)问题如何解决？**