# HW-1：文本分类实现（NYT 新闻分类）

使用三种文本表示方法在 New York Times (NYT) 新闻数据集上完成 3 分类任务
（business / politics / sports），统一在 NYT Test Set 上报告 **Accuracy** 与 **Macro-F1**。

## 目录结构

```
HW-1/
├── data/
│   ├── nyt.csv              # NYT 数据集（11519 条，text/label 两列）
│   ├── ag.csv               # AG News 数据集（90000 条，仅 text 列，用于训练 Word2Vec）
│   ├── nyt_train/val/test.csv   # 运行 prepare_data.py 生成（80/10/10）
│   ├── glove/glove.6B.zip   # 预训练 GloVe（下载：http://nlp.stanford.edu/data/glove.6B.zip）
│   └── word2vec_*.model     # 运行 task2 后自动生成
├── code/
│   ├── utils.py             # 公共工具：分词、数据加载、评价指标
│   ├── prepare_data.py      # 固定随机种子(42)打乱并按 80/10/10 划分 NYT
│   ├── task1_bow.py         # Task 1：Binary BoW / Word Frequency + Logistic Regression
│   ├── task2_word2vec.py    # Task 2：GloVe / Word2Vec(AG) / Word2Vec(NYT) + Logistic Regression
│   ├── task3_bert.py        # Task 3：微调 bert-base-uncased（max_length=64, 3 epochs）
│   └── run_all.py           # 依次运行全部实验
└── results/results.jsonl    # 所有实验的指标输出
```

## 环境准备

Python 3.12（建议使用虚拟环境）：

```bash
python -m venv .venv
.venv/Scripts/python -m pip install pandas scikit-learn gensim transformers torch
```

（中国大陆可加 `-i https://pypi.tuna.tsinghua.edu.cn/simple` 加速。）

另外需要：

1. **GloVe 词向量**：下载 `glove.6B.zip` 放入 `data/glove/`：
   ```
   curl -L -o data/glove/glove.6B.zip http://nlp.stanford.edu/data/glove.6B.zip
   ```
   （脚本运行时会自动解压出 `glove.6B.100d.txt`。）

2. **BERT 预训练模型**：首次运行 `task3_bert.py` 时会自动从 HuggingFace 下载
   `google-bert/bert-base-uncased`；若网络不通，脚本会自动切换到 hf-mirror.com 镜像。

## 运行方式

```bash
cd code

# 1. 数据划分（种子=42，80/10/10）
python prepare_data.py

# 2. 全部实验一次跑完
python run_all.py

# 或分步运行：
python task1_bow.py            # Task 1（binary + freq 两种）
python task1_bow.py binary     # 仅 Binary BoW
python task2_word2vec.py       # Task 2（glove + w2v_ag + w2v_nyt）
python task3_bert.py           # Task 3（BERT 微调，CPU 上约 20-60 分钟）
```

所有实验结果追加写入 `results/results.jsonl`。

## 关键实现说明

- **数据划分**：`random.seed(42)` 打乱后按 80%/10%/10% 切分，全部方法共用同一划分。
- **分词**：正则 `[a-z0-9]+(?:'[a-z]+)?` + 小写化（与 `nltk.word_tokenize` 在该语料上效果等价，且无需额外下载 NLTK 数据）。
- **词表**：Task 1 的词表只用训练集构建，测试集出现的新词被忽略。
- **文档向量**（Task 2）：文档内所有"有词向量"的词取平均；无任何有效词的文档用零向量。
- **Word2Vec 训练超参**：`vector_size=100, window=5, min_count=2, sg=0(CBOW), epochs=5, seed=42`；
  AG News 用全部 90000 条，NYT 只用训练集划分（避免测试信息泄露）。
- **Logistic Regression**：`sklearn.linear_model.LogisticRegression(max_iter=1000)`，其余默认。
- **BERT**：`bert-base-uncased`，`max_length=64`，`batch_size=32`，`lr=2e-5`（AdamW + 线性 warmup），
  3 epochs，按验证集 Macro-F1 保留最优 checkpoint，最终在测试集上评价。

## 结果汇总

见 `results/results.jsonl` 与实验报告。
