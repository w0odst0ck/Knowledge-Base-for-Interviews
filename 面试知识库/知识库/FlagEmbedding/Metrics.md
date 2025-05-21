#  Similarity  相似度
## 1. Jaccard Similarity 
- 在直接计算嵌入向量之间的相似度之前，让我们先看看衡量两个句子相似度的原始方法：Jaccard 相似度。
- 定义：对于集合 A 和 B ，它们的 Jaccard 指数，或称 Jaccard 相似系数，是它们的交集大小除以它们的并集大小：$J(A,B)=\frac{|A∩B|}{|A∪B|}$
- J(A,B) 的值位于 [0,1] 的范围内。
```python
def jaccard_similarity(sentence1, sentence2):
    set1 = set(sentence1.split(" "))
    set2 = set(sentence2.split(" "))
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection)/len(union)
s1 = "Hawaii is a wonderful place for holiday"
s2 = "Peter's favorite place to spend his holiday is Hawaii"
s3 = "Anna enjoys baking during her holiday"
In [3]:
jaccard_similarity(s1, s2)

Out[3]:
0.3333333333333333

In [4]:
jaccard_similarity(s1, s3)

Out[4]:
0.08333333333333333
```
我们可以看到句子 1 和 2 共享了'夏威夷'、'地方'和'假日'。因此，它们的相似度得分（0.333）高于句子 1 和 3 的相似度得分（0.083），后者只共享了'假日'。

## 2. Euclidean Distance 
- 定义：对于向量 A 和 B ，它们之间的欧几里得距离或 L2 距离定义为：$d(A,B)=∥A−B∥_2=\sqrt{\sum_{i=1}^{n}(A_i−B_i)^2}$  
- d(A,B) 的值落在 $[0, +∞ )$ 的范围内。由于这是距离的测量，值越接近 0，两个向量越相似。值越大，两个向量越不相似。
- 您可以逐步计算欧几里得距离或直接调用 torch.cdist()
### (Maximum inner-product search)  
## 3. Cosine Similarity
- 对于向量 A 和 B ，它们的余弦相似度定义为：$cos⁡(θ)=\frac{A⋅B}{∥A∥∥B∥}$
- cos⁡(θ) 的值位于 $[−1,1]$ 的范围内。与欧几里得距离不同，接近 -1 表示完全不相似，而接近 +1 则表示非常相似。
### 3.1 Naive Approach 
- 朴素的方法只是展开表达式：$\frac{A⋅B}{∥A∥∥B∥}=\frac{\sum_{i=1}^{i=n}A_iB_i}{\sqrt{\sum_{i=1}^nA_i^2}⋅\sqrt{\sum_{i=1}^nB_i^2}}$
### 3.2 PyTorch Implementation  
- 朴素方法的问题很少：
- 分子和分母可能会丢失精度
- 精度丢失可能导致计算出的余弦相似度 > 1.0
- 因此，PyTorch 使用以下方式：$frac{A⋅B}{∥A∥∥B∥}=\frac{A}{∥A∥}⋅\frac{B}{∥B∥}$
### 3.3 PyTorch Function Call  
- 实际上，最方便的方式是直接在 torch.nn.functional 中使用 cosine_similarity()：
```python
In [11]:

import torch.nn.functional as F

F.cosine_similarity(A, B).item()

Out[11]:

0.802726686000824
```
## 4. Inner Product/Dot Product  
- 坐标定义：$A⋅B=\sum_{i=1}^{i=n}A_iB_i$
- 几何定义：$A⋅B=∥A∥∥B∥cos⁡(θ)$
```python
In [12]:

dot_prod = A @ B.T
dot_prod.item()

Out[12]:

68.0
```
### Relationship with Cosine similarity  
- 对于计算两个向量之间的距离/相似度，点积和余弦相似度密切相关。余弦相似度只关注角度差异（因为它通过两个向量长度的乘积进行归一化），而点积则同时考虑**长度和角度**。因此，这两个指标在不同应用场景中更受欢迎。
- BGE 系列模型已经将输出嵌入向量归一化，使其大小为 1。因此，使用点积和余弦相似度将得到相同的结果。
```python
In [14]:

from FlagEmbedding import FlagModel

model = FlagModel('BAAI/bge-large-en-v1.5',
                  query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
                  use_fp16=True)

In [15]:

sentence = "I am very interested in natural language processing"
embedding = torch.tensor(model.encode(sentence))
torch.norm(embedding).item()

Out[15]:

1.0
```
## 5. Examples  5. 示例
```python
In [16]:

sentence_1 = "I will watch a show tonight"
sentence_2 = "I will show you my watch tonight"
sentence_3 = "I'm going to enjoy a performance this evening"

```
- 很明显，在句子 1 中，“watch”是动词，“show”是名词。
- 但是，在句子 2 中，“show”是动词，“watch”是名词，这导致两个句子的意义不同。
- 虽然第 3 句与第 1 句意义非常相似。
- 现在让我们看看不同的相似度指标如何告诉我们句子的关系。
```python 
In [17]:

print(jaccard_similarity(sentence_1, sentence_2))
print(jaccard_similarity(sentence_1, sentence_3))

0.625
0.07692307692307693
```
- 结果显示，句子 1 和 2（0.625）比句子 1 和 3（0.077）要相似得多，这与我们得出的结论相反。
- 现在让我们首先获取这些句子的嵌入表示。
```python
In [18]:

embeddings = torch.from_numpy(model.encode([sentence_1, sentence_2, sentence_3]))
embedding_1 = embeddings[0].view(1, -1)
embedding_2 = embeddings[1].view(1, -1)
embedding_3 = embeddings[2].view(1, -1)

print(embedding_1.shape)

torch.Size([1, 1024])

Then let's compute the Euclidean distance:  
然后让我们计算欧几里得距离：

In [19]:

euc_dist1_2 = torch.cdist(embedding_1, embedding_2, p=2).item()
euc_dist1_3 = torch.cdist(embedding_1, embedding_3, p=2).item()
print(euc_dist1_2)
print(euc_dist1_3)

0.714613139629364
0.5931472182273865
```
- 然后，让我们看看余弦相似度：
```python
In [20]:

cos_dist1_2 = F.cosine_similarity(embedding_1, embedding_2).item()
cos_dist1_3 = F.cosine_similarity(embedding_1, embedding_3).item()
print(cos_dist1_2)
print(cos_dist1_3)

0.7446640729904175
0.8240882158279419
```
- 使用嵌入，我们可以得到与 Jaccard 相似度不同的正确结果，即句子 1 和句子 2 应该比句子 1 和句子 3 更相似，无论是使用欧几里得距离还是余弦相似度作为度量标准。
#   Evaluation Metrics  评估指标
## 0. Preparation  0. 准备
- 假设我们有一个包含文档 ID 从 0 到 30 的语料库。
- `ground_truth` 包含每个查询的实际相关文档 ID。
- `results` 包含每个查询的检索系统搜索结果。
```python
In [1]:
import numpy as np
ground_truth = [
    [11,  1,  7, 17, 21],
    [ 4, 16,  1],
    [26, 10, 22,  8],
]
results = [
    [11,  1, 17,  7, 21,  8,  0, 28,  9, 20],
    [16,  1,  6, 18,  3,  4, 25, 19,  8, 14],
    [24, 10, 26,  2,  8, 28,  4, 23, 13, 21],
]
In [63]:
np.intersect1d(ground_truth, results)
Out[63]:
array([ 0,  1,  2,  3,  4,  6,  7,  8,  9, 10, 11, 13, 14, 16, 17, 18, 19,
       21, 22, 24, 25, 26, 28])
In [65]:
np.isin(ground_truth, results).astype(int)
Out[65]:
array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
       [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
And we are interested in the following cutoffs:  
我们感兴趣的是以下截止值：
In [2]:
cutoffs = [1, 5, 10]
```
## 1. Recall  
- 召回表示模型从数据集中所有实际正样本中正确预测正实例的能力。
$Recall=\frac{True Positives}{TruePositives+False Negatives}$
- 将信息检索的形式写出来，即检索到的相关文档与语料库中总相关文档的比例。在实际操作中，我们通常将分母设置为当前截止值（通常是 1、5、10、100 等）与语料库中总相关文档数中的最小值：$Recall=\frac{|\{Relevant docs\}∩\{Retrieved docs\}|}{min(|\{Retrieved docs\}|,|\{Relevant docs\}|)}$
```python
In [3]:
def calc_recall(preds, truths, cutoffs):
    recalls = np.zeros(len(cutoffs))
    for text, truth in zip(preds, truths):
        for i, c in enumerate(cutoffs):
            hits = np.intersect1d(truth, text[:c])
            recalls[i] += len(hits) / max(min(c, len(truth)), 1)
    recalls /= len(preds)
    return recalls

In [4]:
recalls = calc_recall(results, ground_truth, cutoffs)
for i, c in enumerate(cutoffs):
    print(f"recall@{c}: {recalls[i]}")

recall@1: 0.6666666666666666
recall@5: 0.8055555555555555
recall@10: 0.9166666666666666
```
## 2. MRR
- 平均倒数排名（MRR）是信息检索中广泛使用的指标，用于评估系统的有效性。它衡量搜索结果列表中**第一个相关结果**的排名位置。$MRR=\frac{1}{|Q|}\sum_{i=1}^{|Q|}\frac{1}{rank_i}$
- |Q| 是查询总数。
- ranki 是第 i 个查询的第一个相关文档的排名位置。
```python
In [5]:
def calc_MRR(preds, truth, cutoffs):
    mrr = [0 for _ in range(len(cutoffs))]
    for pred, t in zip(preds, truth):
        for i, c in enumerate(cutoffs):
            for j, p in enumerate(pred):
                if j < c and p in t:
                    mrr[i] += 1/(j+1)
                    break
    mrr = [k/len(preds) for k in mrr]
    return mrr

In [6]:
mrr = calc_MRR(results, ground_truth, cutoffs)
for i, c in enumerate(cutoffs):
    print(f"MRR@{c}: {mrr[i]}")

MRR@1: 0.6666666666666666
MRR@5: 0.8333333333333334
MRR@10: 0.8333333333333334
```
## 3. nDCG
- 标准化折算累积增益（nDCG）通过考虑相关文档的位置及其评分的相关性来衡量搜索结果排名列表的质量。nDCG 的计算涉及两个主要步骤：
1. 折扣累积增益（DCG）衡量检索任务中的排序质量。$DCG_p=\sum_{i=1}^p\frac{2^{rel_i}−1}{log_2⁡(i+1)}$
2. 通过理想 DCG 进行归一化，以便在查询之间进行比较。$nDCG_p=\frac{DCG_p}{IDCG_p}$其中 IDCG 是给定文档集的最大可能 DCG，假设它们按相关性完美排序。
```python
In [7]:

pred_hard_encodings = []
for pred, label in zip(results, ground_truth):
    pred_hard_encoding = list(np.isin(pred, label).astype(int))
    pred_hard_encodings.append(pred_hard_encoding)

In [8]:

from sklearn.metrics import ndcg_score

for i, c in enumerate(cutoffs):
    nDCG = ndcg_score(pred_hard_encodings, results, k=c)
    print(f"nDCG@{c}: {nDCG}")

nDCG@1: 0.0
nDCG@5: 0.3298163165186628
nDCG@10: 0.5955665344840209

```
## 4. Precision 
```python
In [9]:

def calc_precision(preds, truths, cutoffs):
    prec = np.zeros(len(cutoffs))
    for text, truth in zip(preds, truths):
        for i, c in enumerate(cutoffs):
            hits = np.intersect1d(truth, text[:c])
            prec[i] += len(hits) / c
    prec /= len(preds)
    return prec

In [10]:

precisions = calc_precision(results, ground_truth, cutoffs)
for i, c in enumerate(cutoffs):
    print(f"precision@{c}: {precisions[i]}")

precision@1: 0.6666666666666666
precision@5: 0.6666666666666666
precision@10: 0.3666666666666667
```
## 5. MAP
- 平均精度均值（MAP）衡量系统在多个查询中返回相关文档的有效性。
- 首先，平均精度（AP）评估相关文档在检索到的文档中的排名效果。它是通过平均所有检索到的文档中每个相关文档在排名中的精度值来计算的：$AP=\frac{\sum_{k=1}^{M}Relevance(k)×Precision(k)}{|{Relevant Docs}|}$
- M 是检索到的文档总数。
- Relevance(k) 是一个二进制值，表示位置 k 的文档是否相关（=1）或无关（=0）。
- Precision(k) 是仅考虑检索到的最顶部 k 项时的精度。
- 然后计算多个查询的平均 AP 以获得 MAP：$MAP=\frac{1}{N}\sum_{i=1}^{N}AP_i$
- N 是查询总数。
- APi 是第 ith 查询的平均精度。
```python
In [11]:
def calc_AP(encoding):
    rel = 0
    precs = 0.0
    for k, hit in enumerate(encoding, start=1):
        if hit == 1:
            rel += 1
            precs += rel/k

    return 0 if rel == 0 else precs/rel

In [12]:
def calc_MAP(encodings, cutoffs):
    res = []
    for c in cutoffs:
        ap_sum = 0.0
        for encoding in encodings:
            ap_sum += calc_AP(encoding[:c])
        res.append(ap_sum/len(encodings))
        
    return res

In [14]:
maps = calc_MAP(pred_hard_encodings, cutoffs)
for i, c in enumerate(cutoffs):
    print(f"MAP@{c}: {maps[i]}")

MAP@1: 0.6666666666666666
MAP@5: 0.862962962962963
MAP@10: 0.8074074074074075
```
#   Indexing Using Faiss 
- 使用索引来使我们的检索快速且整洁。
## Step 0
### faiss-gpu on Linux (x86_64)  
- 创建 conda 虚拟环境并运行：
`conda install -c pytorch -c nvidia faiss-gpu=1.8.0`
### faiss-cpu
- 安装 `faiss-cpu`
```python
In [ ]:

%pip install -U faiss-cpu

```
## Step 1: Dataset  
```python
In [1]:

corpus = [
    "Michael Jackson was a legendary pop icon known for his record-breaking music and dance innovations.",
    "Fei-Fei Li is a professor in Stanford University, revolutionized computer vision with the ImageNet project.",
    "Brad Pitt is a versatile actor and producer known for his roles in films like 'Fight Club' and 'Once Upon a Time in Hollywood.'",
    "Geoffrey Hinton, as a foundational figure in AI, received Turing Award for his contribution in deep learning.",
    "Eminem is a renowned rapper and one of the best-selling music artists of all time.",
    "Taylor Swift is a Grammy-winning singer-songwriter known for her narrative-driven music.",
    "Sam Altman leads OpenAI as its CEO, with astonishing works of GPT series and pursuing safe and beneficial AI.",
    "Morgan Freeman is an acclaimed actor famous for his distinctive voice and diverse roles.",
    "Andrew Ng spread AI knowledge globally via public courses on Coursera and Stanford University.",
    "Robert Downey Jr. is an iconic actor best known for playing Iron Man in the Marvel Cinematic Universe.",
]
And a few queries (add your own queries and check the result!):  
并且一些查询（添加您自己的查询并检查结果！）：

In [2]:

queries = [
    "Who is Robert Downey Jr.?",
    "An expert of neural network",
    "A famous female singer",
]
```
## Step 2: Text Embedding 
- 嵌入前 500 个文档。
```python
In [12]:

from FlagEmbedding import FlagModel

# get the BGE embedding model
model = FlagModel('BAAI/bge-base-en-v1.5',
                  query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
                  use_fp16=True)

# get the embedding of the corpus
corpus_embeddings = model.encode(corpus)

print("shape of the corpus embeddings:", corpus_embeddings.shape)
print("data type of the embeddings: ", corpus_embeddings.dtype)

shape of the corpus embeddings: (10, 768)
data type of the embeddings:  float32
```
- Faiss 仅接受 float32 输入。
- 确保在将它们添加到索引之前，corpus_embeddings 的数据类型是 float32。
```python
In [13]:

import numpy as np

corpus_embeddings = corpus_embeddings.astype(np.float32)
```
## Step 3: Indexing  
- 构建一个索引并将嵌入向量添加到其中。
```python
In [14]:

import faiss

# get the length of our embedding vectors, vectors by bge-base-en-v1.5 have length 768
dim = corpus_embeddings.shape[-1]

# create the faiss index and store the corpus embeddings into the vector space
index = faiss.index_factory(dim, 'Flat', faiss.METRIC_INNER_PRODUCT)

# if you installed faiss-gpu, uncomment the following lines to make the index on your GPUs.

# co = faiss.GpuMultipleClonerOptions()
# index = faiss.index_cpu_to_all_gpus(index, co)
```
 - 无需使用“Flat”量化器和 METRIC_INNER_PRODUCT 作为度量时进行训练。一些其他索引在使用量化时可能需要训练。
```python
In [15]:

# check if the index is trained
print(index.is_trained)  
# index.train(corpus_embeddings)

# add all the vectors to the index
index.add(corpus_embeddings)

print(f"total number of vectors: {index.ntotal}")

True
total number of vectors: 10
```
### Step 3.5 (Optional): Saving Faiss index  
```python
In [16]:
# change the path to where you want to save the index
path = "./index.bin"
faiss.write_index(index, path)

If you already have stored index in your local directory, you can load it by:  
如果您已经在本地目录中存储了索引，您可以通过以下方式加载它：

In [17]:
index = faiss.read_index("./index.bin")
```
## Step 4: Find answers to the query  
- 获取所有查询的嵌入表示：
```python
In [18]:

query_embeddings = model.encode_queries(queries)

Then, use the Faiss index to do a knn search in the vector space:  
然后，使用 Faiss 索引在向量空间中进行 knn 搜索：

In [19]:

dists, ids = index.search(query_embeddings, k=3)
print(dists)
print(ids)

[[0.6686779  0.37858668 0.3767978 ]
 [0.6062041  0.59364545 0.527691  ]
 [0.5409331  0.5097007  0.42427146]]
[[9 7 2]
 [3 1 8]
 [5 0 4]]
```
- 结果：
```python
In [20]:

for i, q in enumerate(queries):
    print(f"query:\t{q}\nanswer:\t{corpus[ids[i][0]]}\n")

query:	Who is Robert Downey Jr.?
answer:	Robert Downey Jr. is an iconic actor best known for playing Iron Man in the Marvel Cinematic Universe.

query:	An expert of neural network
answer:	Geoffrey Hinton, as a foundational figure in AI, received Turing Award for his contribution in deep learning.

query:	A famous female singer
answer:	Taylor Swift is a Grammy-winning singer-songwriter known for her narrative-driven music.
```