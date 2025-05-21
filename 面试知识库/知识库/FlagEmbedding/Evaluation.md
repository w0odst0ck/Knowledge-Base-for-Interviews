#   Evaluation  
- 在 MS Marco 上评估嵌入模型性能的整个流程，并使用三个指标来展示其性能。
## Step 0: 
- 安装依赖项。
```python
In [ ]:

%pip install -U FlagEmbedding faiss-cpu
```
## Step 1: Load Dataset  
- 从 Huggingface Dataset 下载查询和 MS Marco。
```python
In [4]:

from datasets import load_dataset
import numpy as np

data = load_dataset("namespace-Pt/msmarco", split="dev")
```
- 截断数据集。 queries 包含数据集的前 100 个查询。 corpus 由前 5,000 个查询的正面组成。
```python
In [5]:

queries = np.array(data[:100]["query"])
corpus = sum(data[:5000]["positive"], [])
```
 - GPU 
```python
In [ ]:

# data = load_dataset("namespace-Pt/msmarco", split="dev")
# queries = np.array(data["query"])

# corpus = load_dataset("namespace-PT/msmarco-corpus", split="train")
```
## Step 2: Embedding  
- 选择我们想要评估的嵌入模型，并将语料库编码为嵌入。
```python
In [6]:
from FlagEmbedding import FlagModel
# get the BGE embedding model
model = FlagModel('BAAI/bge-base-en-v1.5',
                  query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
                  use_fp16=True)

# get the embedding of the corpus
corpus_embeddings = model.encode(corpus)

print("shape of the corpus embeddings:", corpus_embeddings.shape)
print("data type of the embeddings: ", corpus_embeddings.dtype)

Inference Embeddings: 100%|██████████| 21/21 [02:10<00:00,  6.22s/it]

shape of the corpus embeddings: (5331, 768)
data type of the embeddings:  float32
```
## Step 3: Indexing
- 使用 index_factory() 函数来创建我们想要的 Faiss 索引：
- 第一个参数 `dim` 是向量空间的维度，如果你使用 bge-base-en-v1.5，则为 768。
    
- 第二个参数 `'Flat'` 使索引执行穷举搜索。
    
- 第三参数 `faiss.METRIC_INNER_PRODUCT` 告诉索引使用内积作为距离度量。
    
```python
In [7]:

import faiss

# get the length of our embedding vectors, vectors by bge-base-en-v1.5 have length 768
dim = corpus_embeddings.shape[-1]

# create the faiss index and store the corpus embeddings into the vector space
index = faiss.index_factory(dim, 'Flat', faiss.METRIC_INNER_PRODUCT)
corpus_embeddings = corpus_embeddings.astype(np.float32)
# train and add the embeddings to the index
index.train(corpus_embeddings)
index.add(corpus_embeddings)

print(f"total number of vectors: {index.ntotal}")

total number of vectors: 5331
```
- 由于嵌入过程耗时，保存索引以用于复制或其他实验是个不错的选择。
- 取消以下行的注释以保存索引。
```python
In [8]:

# path = "./index.bin"
# faiss.write_index(index, path)

If you already have stored index in your local directory, you can load it by:  
如果您已经在本地目录中存储了索引，您可以通过以下方式加载它：

In [ ]:

# index = faiss.read_index("./index.bin")
```
## Step 4: Retrieval  
- 获取所有查询的嵌入，并获取它们对应的真实答案以进行评估。
```python
In [10]:

query_embeddings = model.encode_queries(queries)
ground_truths = [d["positive"] for d in data]
corpus = np.asarray(corpus)
```
- 使用 faiss 索引搜索每个查询的前 k 个答案。
```python
In [11]:

from tqdm import tqdm

res_scores, res_ids, res_text = [], [], []
query_size = len(query_embeddings)
batch_size = 256
# The cutoffs we will use during evaluation, and set k to be the maximum of the cutoffs.
cut_offs = [1, 10]
k = max(cut_offs)

for i in tqdm(range(0, query_size, batch_size), desc="Searching"):
    q_embedding = query_embeddings[i: min(i+batch_size, query_size)].astype(np.float32)
    # search the top k answers for each of the queries
    score, idx = index.search(q_embedding, k=k)
    res_scores += list(score)
    res_ids += list(idx)
    res_text += list(corpus[idx])

Searching: 100%|██████████| 1/1 [00:00<00:00, 20.91it/s]
```
## Step 5: Evaluate 

### 5.1 Recall 
- 召回率表示模型从数据集中所有实际正样本中正确预测正实例的能力。$Recall=\frac{True Positives}{True Positives+False Negatives}$
- 当误报成本较高时，召回率很有用。换句话说，我们试图找到所有正类对象，即使这会导致一些误报。这个属性使得召回率成为文本检索任务的有用指标。
```python
In [13]:

def calc_recall(preds, truths, cutoffs):
    recalls = np.zeros(len(cutoffs))
    for text, truth in zip(preds, truths):
        for i, c in enumerate(cutoffs):
            recall = np.intersect1d(truth, text[:c])
            recalls[i] += len(recall) / max(min(c, len(truth)), 1)
    recalls /= len(preds)
    return recalls

recalls = calc_recall(res_text, ground_truths, cut_offs)
for i, c in enumerate(cut_offs):
    print(f"recall@{c}: {recalls[i]}")

recall@1: 0.97
recall@10: 1.0
```
### 5.2 MRR
- 平均倒数排名（MRR）是信息检索中广泛使用的指标，用于评估系统的有效性。它衡量搜索结果列表中第一个相关结果的排名位置。$MRR=\frac{1}{|Q|}\sum_{i=1}^{|Q|}\frac{1}{rank_i}$
- 其中
- |Q| 是查询总数。
- $rank_i$ 是第 i 个查询的第一个相关文档的排名位置。
```python
In [14]:

def MRR(preds, truth, cutoffs):
    mrr = [0 for _ in range(len(cutoffs))]
    for pred, t in zip(preds, truth):
        for i, c in enumerate(cutoffs):
            for j, p in enumerate(pred):
                if j < c and p in t:
                    mrr[i] += 1/(j+1)
                    break
    mrr = [k/len(preds) for k in mrr]
    return mrr

In [15]:

mrr = MRR(res_text, ground_truths, cut_offs)
for i, c in enumerate(cut_offs):
    print(f"MRR@{c}: {mrr[i]}")

MRR@1: 0.97
MRR@10: 0.9825
```
### 5.3 nDCG
- 标准化折算累积增益（nDCG）通过考虑相关文档的位置及其评分的相关性来衡量排序搜索结果的列表质量。nDCG 的计算涉及两个主要步骤：
1. 折算累积增益（DCG）衡量检索任务中的排序质量。$DCG_p=\sum_{i=1}^p\frac{2^{rel_i}−1}{log_2⁡(i+1)}$
2. 通过理想 DCG 进行标准化，以便跨查询进行比较。$nDCG_p=\frac{DCG_p}{IDCG_p}$其中 IDCG 是给定文档集的最大可能 DCG，假设它们按相关性完美排序。
```python
In [16]:

pred_hard_encodings = []
for pred, label in zip(res_text, ground_truths):
    pred_hard_encoding = list(np.isin(pred, label).astype(int))
    pred_hard_encodings.append(pred_hard_encoding)

In [17]:

from sklearn.metrics import ndcg_score

for i, c in enumerate(cut_offs):
    nDCG = ndcg_score(pred_hard_encodings, res_scores, k=c)
    print(f"nDCG@{c}: {nDCG}")

nDCG@1: 0.97
nDCG@10: 0.9869253606521631
```
#   MTEB
- 为了评估嵌入模型，MTEB 是最著名的基准之一。
## 0. Installation 

Install the packages we will use in your environment:  
在您的环境中安装我们将要使用的包：
```python
In [ ]:

%%capture
%pip install sentence_transformers mteb
```
## 1. Intro 
- 大规模文本嵌入基准（MTEB）是一个用于评估文本嵌入模型在广泛自然语言处理（NLP）任务中性能的大规模评估框架。MTEB 旨在标准化和改进文本嵌入的评估，对于评估这些模型在各种实际应用中的泛化能力至关重要。它包含八个主要 NLP 任务和不同语言的广泛数据集，并提供了一个易于使用的评估管道。
- MTEB 也因其 MTEB 排行榜而广为人知，该排行榜包含最新的顶级嵌入模型排名。我们将在下一教程中介绍这一点。现在让我们看看如何使用 MTEB 轻松进行评估。
```python
In [12]:

import mteb
from sentence_transformers import SentenceTransformer
```
- 使用 MTEB 进行快速评估。
- 加载我们想要评估的模型：
```python
In [13]:

model_name = "BAAI/bge-base-en-v1.5"
model = SentenceTransformer(model_name)

```
- MTEB 英文排行榜所使用的检索数据集列表。
- MTEB 直接使用开源基准 BEIR 在其检索部分，其中包含 15 个数据集（注意 CQADupstack 有 12 个子集）。
```python
In [14]:

retrieval_tasks = [
    "ArguAna",
    "ClimateFEVER",
    "CQADupstackAndroidRetrieval",
    "CQADupstackEnglishRetrieval",
    "CQADupstackGamingRetrieval",
    "CQADupstackGisRetrieval",
    "CQADupstackMathematicaRetrieval",
    "CQADupstackPhysicsRetrieval",
    "CQADupstackProgrammersRetrieval",
    "CQADupstackStatsRetrieval",
    "CQADupstackTexRetrieval",
    "CQADupstackUnixRetrieval",
    "CQADupstackWebmastersRetrieval",
    "CQADupstackWordpressRetrieval",
    "DBPedia",
    "FEVER",
    "FiQA2018",
    "HotpotQA",
    "MSMARCO",
    "NFCorpus",
    "NQ",
    "QuoraRetrieval",
    "SCIDOCS",
    "SciFact",
    "Touche2020",
    "TRECCOVID",
]
```
- "ArguAna"。
- 要查看 MTEB 支持的所有任务和语言列表，请查看页面。
```python
In [15]:

tasks = mteb.get_tasks(tasks=retrieval_tasks[:1])
```
- 使用我们选择的任务创建并初始化一个 MTEB 实例，并运行评估过程。
```python
In [16]:

# use the tasks we chose to initialize the MTEB instance
evaluation = mteb.MTEB(tasks=tasks)

# call run() with the model and output_folder
results = evaluation.run(model, output_folder="results")

───────────────────────────────────────────────── Selected tasks  

Retrieval

    - ArguAna, s2p

Batches: 100%|██████████| 44/44 [00:41<00:00,  1.06it/s]
Batches: 100%|██████████| 272/272 [03:36<00:00,  1.26it/s]
```
- 结果应存储在 `{output_folder}/{model_name}/{model_revision}/{task_name}.json` 。
- 打开 json 文件，您应该看到以下内容，这是在"ArguAna"上不同阈值从 1 到 1000 的评估结果。
```json
{
  "dataset_revision": "c22ab2a51041ffd869aaddef7af8d8215647e41a",
  "evaluation_time": 260.14976954460144,
  "kg_co2_emissions": null,
  "mteb_version": "1.14.17",
  "scores": {
    "test": [
      {
        "hf_subset": "default",
        "languages": [
          "eng-Latn"
        ],
        "main_score": 0.63616,
        "map_at_1": 0.40754,
        "map_at_10": 0.55773,
        "map_at_100": 0.56344,
        "map_at_1000": 0.56347,
        "map_at_20": 0.56202,
        "map_at_3": 0.51932,
        "map_at_5": 0.54023,
        "mrr_at_1": 0.4139402560455192,
        "mrr_at_10": 0.5603739077423295,
        "mrr_at_100": 0.5660817425350153,
        "mrr_at_1000": 0.5661121884705748,
        "mrr_at_20": 0.564661930998293,
        "mrr_at_3": 0.5208629682313899,
        "mrr_at_5": 0.5429113323850182,
        "nauc_map_at_1000_diff1": 0.15930478114759905,
        "nauc_map_at_1000_max": -0.06396189194646361,
        "nauc_map_at_1000_std": -0.13168797291549253,
        "nauc_map_at_100_diff1": 0.15934819555197366,
        "nauc_map_at_100_max": -0.06389635013430676,
        "nauc_map_at_100_std": -0.13164524259533786,
        "nauc_map_at_10_diff1": 0.16057318234658585,
        "nauc_map_at_10_max": -0.060962623117325254,
        "nauc_map_at_10_std": -0.1300413865104607,
        "nauc_map_at_1_diff1": 0.17346152653542332,
        "nauc_map_at_1_max": -0.09705499215630589,
        "nauc_map_at_1_std": -0.14726476953035533,
        "nauc_map_at_20_diff1": 0.15956349246366208,
        "nauc_map_at_20_max": -0.06259296677860492,
        "nauc_map_at_20_std": -0.13097093150054095,
        "nauc_map_at_3_diff1": 0.15620049317363813,
        "nauc_map_at_3_max": -0.06690213479396273,
        "nauc_map_at_3_std": -0.13440904793529648,
        "nauc_map_at_5_diff1": 0.1557795701081579,
        "nauc_map_at_5_max": -0.06255283252590663,
        "nauc_map_at_5_std": -0.1355361594910923,
        "nauc_mrr_at_1000_diff1": 0.1378988612808882,
        "nauc_mrr_at_1000_max": -0.07507962333910836,
        "nauc_mrr_at_1000_std": -0.12969109830101241,
        "nauc_mrr_at_100_diff1": 0.13794450668758515,
        "nauc_mrr_at_100_max": -0.07501290390362861,
        "nauc_mrr_at_100_std": -0.12964855554504057,
        "nauc_mrr_at_10_diff1": 0.1396047981645623,
        "nauc_mrr_at_10_max": -0.07185174301688693,
        "nauc_mrr_at_10_std": -0.12807325096717753,
        "nauc_mrr_at_1_diff1": 0.15610387932529113,
        "nauc_mrr_at_1_max": -0.09824591983546396,
        "nauc_mrr_at_1_std": -0.13914318784294258,
        "nauc_mrr_at_20_diff1": 0.1382786098284509,
        "nauc_mrr_at_20_max": -0.07364476417961506,
        "nauc_mrr_at_20_std": -0.12898192060943495,
        "nauc_mrr_at_3_diff1": 0.13118224861025093,
        "nauc_mrr_at_3_max": -0.08164985279853691,
        "nauc_mrr_at_3_std": -0.13241573571401533,
        "nauc_mrr_at_5_diff1": 0.1346130730317385,
        "nauc_mrr_at_5_max": -0.07404093236468848,
        "nauc_mrr_at_5_std": -0.1340775377068567,
        "nauc_ndcg_at_1000_diff1": 0.15919987960292029,
        "nauc_ndcg_at_1000_max": -0.05457945565481172,
        "nauc_ndcg_at_1000_std": -0.12457339152558143,
        "nauc_ndcg_at_100_diff1": 0.1604091882521101,
        "nauc_ndcg_at_100_max": -0.05281549383775287,
        "nauc_ndcg_at_100_std": -0.12347288098914058,
        "nauc_ndcg_at_10_diff1": 0.1657018523692905,
        "nauc_ndcg_at_10_max": -0.036222943297402846,
        "nauc_ndcg_at_10_std": -0.11284619565817842,
        "nauc_ndcg_at_1_diff1": 0.17346152653542332,
        "nauc_ndcg_at_1_max": -0.09705499215630589,
        "nauc_ndcg_at_1_std": -0.14726476953035533,
        "nauc_ndcg_at_20_diff1": 0.16231721725673165,
        "nauc_ndcg_at_20_max": -0.04147115653921931,
        "nauc_ndcg_at_20_std": -0.11598700704312062,
        "nauc_ndcg_at_3_diff1": 0.15256475371124711,
        "nauc_ndcg_at_3_max": -0.05432154580979357,
        "nauc_ndcg_at_3_std": -0.12841084787822227,
        "nauc_ndcg_at_5_diff1": 0.15236205846534961,
        "nauc_ndcg_at_5_max": -0.04356123278888682,
        "nauc_ndcg_at_5_std": -0.12942556865700913,
        "nauc_precision_at_1000_diff1": -0.038790629929866066,
        "nauc_precision_at_1000_max": 0.3630826341915611,
        "nauc_precision_at_1000_std": 0.4772189839676386,
        "nauc_precision_at_100_diff1": 0.32118609204433185,
        "nauc_precision_at_100_max": 0.4740132817600036,
        "nauc_precision_at_100_std": 0.3456396169952022,
        "nauc_precision_at_10_diff1": 0.22279659689895104,
        "nauc_precision_at_10_max": 0.16823918613191954,
        "nauc_precision_at_10_std": 0.0377209694331257,
        "nauc_precision_at_1_diff1": 0.17346152653542332,
        "nauc_precision_at_1_max": -0.09705499215630589,
        "nauc_precision_at_1_std": -0.14726476953035533,
        "nauc_precision_at_20_diff1": 0.23025740175221762,
        "nauc_precision_at_20_max": 0.2892313928157665,
        "nauc_precision_at_20_std": 0.13522755012490692,
        "nauc_precision_at_3_diff1": 0.1410889527057097,
        "nauc_precision_at_3_max": -0.010771302313530132,
        "nauc_precision_at_3_std": -0.10744937823276193,
        "nauc_precision_at_5_diff1": 0.14012953903010988,
        "nauc_precision_at_5_max": 0.03977485677045894,
        "nauc_precision_at_5_std": -0.10292184602358977,
        "nauc_recall_at_1000_diff1": -0.03879062992990034,
        "nauc_recall_at_1000_max": 0.36308263419153386,
        "nauc_recall_at_1000_std": 0.47721898396760526,
        "nauc_recall_at_100_diff1": 0.3211860920443005,
        "nauc_recall_at_100_max": 0.4740132817599919,
        "nauc_recall_at_100_std": 0.345639616995194,
        "nauc_recall_at_10_diff1": 0.22279659689895054,
        "nauc_recall_at_10_max": 0.16823918613192046,
        "nauc_recall_at_10_std": 0.037720969433127145,
        "nauc_recall_at_1_diff1": 0.17346152653542332,
        "nauc_recall_at_1_max": -0.09705499215630589,
        "nauc_recall_at_1_std": -0.14726476953035533,
        "nauc_recall_at_20_diff1": 0.23025740175221865,
        "nauc_recall_at_20_max": 0.2892313928157675,
        "nauc_recall_at_20_std": 0.13522755012490456,
        "nauc_recall_at_3_diff1": 0.14108895270570979,
        "nauc_recall_at_3_max": -0.010771302313529425,
        "nauc_recall_at_3_std": -0.10744937823276134,
        "nauc_recall_at_5_diff1": 0.14012953903010958,
        "nauc_recall_at_5_max": 0.039774856770459645,
        "nauc_recall_at_5_std": -0.10292184602358935,
        "ndcg_at_1": 0.40754,
        "ndcg_at_10": 0.63616,
        "ndcg_at_100": 0.66063,
        "ndcg_at_1000": 0.6613,
        "ndcg_at_20": 0.65131,
        "ndcg_at_3": 0.55717,
        "ndcg_at_5": 0.59461,
        "precision_at_1": 0.40754,
        "precision_at_10": 0.08841,
        "precision_at_100": 0.00991,
        "precision_at_1000": 0.001,
        "precision_at_20": 0.04716,
        "precision_at_3": 0.22238,
        "precision_at_5": 0.15149,
        "recall_at_1": 0.40754,
        "recall_at_10": 0.88407,
        "recall_at_100": 0.99147,
        "recall_at_1000": 0.99644,
        "recall_at_20": 0.9431,
        "recall_at_3": 0.66714,
        "recall_at_5": 0.75747
      }
    ]
  },
  "task_name": "ArguAna"
}
```
- 现在我们已经成功使用 mteb 运行了评估！在下一教程中，我们将展示如何评估您的模型在英语 MTEB 的全部 56 个任务上，并与排行榜上的模型进行竞争。
#   MTEB Leaderboard  
- 进行全面评估，并将结果与 MTEB 英语排行榜进行比较。
- 注意：在满载 Eng MTEB 上进行评估即使使用 GPU 也非常耗时。
## 0. Installation  

Install the packages we will use in your environment:  
在您的环境中安装我们将使用的包：
```python
In [ ]:

%%capture
%pip install sentence_transformers mteb
```
## 1. Run the Evaluation 
- MTEB 英语排行榜包含 7 个任务上的 56 个数据集：

1. **Classification**: 使用嵌入在训练集上训练逻辑回归，并在测试集上评分。F1 是主要指标。
2. **Clustering**: 使用批大小为 32 和 k 等于不同标签数量的 mini-batch k-means 模型进行训练。然后使用 v-measure 进行评分。
3. **Pair Classification**: 提供一对文本输入和一个标签，该标签是一个二进制变量，需要分配。主要指标是平均精确度评分。
4. **Reranking**: 根据查询对一系列相关和不相关的参考文本进行排序。指标是平均 MRR@k 和 MAP。
5. **Retrieval**: 每个数据集包含语料库、查询以及将每个查询与其在语料库中相关文档的映射。目标是检索每个查询的相关文档。主要指标是 nDCG@k。MTEB 直接采用 BEIR 进行检索任务。
6. **Semantic Textual Similarity (STS)**: 确定每对句子之间的相似度。基于余弦相似度的 Spearman 相关系数作为主要指标。
7. **Summarization**: 本任务仅使用 1 个数据集。通过计算其嵌入的距离来评估机器生成的摘要与人工编写的摘要之间的得分。主要指标同样是基于余弦相似度的 Spearman 相关系数。
- 导入 `MTEB_MAIN_EN` 以检查所有 56 个数据集。
```python
In [2]:

import mteb
from mteb.benchmarks import MTEB_MAIN_EN

print(MTEB_MAIN_EN.tasks)

['AmazonCounterfactualClassification', 'AmazonPolarityClassification', 'AmazonReviewsClassification', 'ArguAna', 'ArxivClusteringP2P', 'ArxivClusteringS2S', 'AskUbuntuDupQuestions', 'BIOSSES', 'Banking77Classification', 'BiorxivClusteringP2P', 'BiorxivClusteringS2S', 'CQADupstackAndroidRetrieval', 'CQADupstackEnglishRetrieval', 'CQADupstackGamingRetrieval', 'CQADupstackGisRetrieval', 'CQADupstackMathematicaRetrieval', 'CQADupstackPhysicsRetrieval', 'CQADupstackProgrammersRetrieval', 'CQADupstackStatsRetrieval', 'CQADupstackTexRetrieval', 'CQADupstackUnixRetrieval', 'CQADupstackWebmastersRetrieval', 'CQADupstackWordpressRetrieval', 'ClimateFEVER', 'DBPedia', 'EmotionClassification', 'FEVER', 'FiQA2018', 'HotpotQA', 'ImdbClassification', 'MSMARCO', 'MTOPDomainClassification', 'MTOPIntentClassification', 'MassiveIntentClassification', 'MassiveScenarioClassification', 'MedrxivClusteringP2P', 'MedrxivClusteringS2S', 'MindSmallReranking', 'NFCorpus', 'NQ', 'QuoraRetrieval', 'RedditClustering', 'RedditClusteringP2P', 'SCIDOCS', 'SICK-R', 'STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STS17', 'STS22', 'STSBenchmark', 'SciDocsRR', 'SciFact', 'SprintDuplicateQuestions', 'StackExchangeClustering', 'StackExchangeClusteringP2P', 'StackOverflowDupQuestions', 'SummEval', 'TRECCOVID', 'Touche2020', 'ToxicConversationsClassification', 'TweetSentimentExtractionClassification', 'TwentyNewsgroupsClustering', 'TwitterSemEval2015', 'TwitterURLCorpus']
```
- 加载要评估的模型：
```python
In [ ]:

from sentence_transformers import SentenceTransformer

model_name = "BAAI/bge-base-en-v1.5"
model = SentenceTransformer(model_name)

Alternatively, MTEB provides popular models on their leaderboard in order to reproduce their results.  
另外，MTEB 在其排行榜上提供流行的模型，以便重现其结果。

In [ ]:

model_name = "BAAI/bge-base-en-v1.5"
model = mteb.get_model(model_name)

```
- 对每个数据集进行评估：
```python
In [ ]:

for task in MTEB_MAIN_EN.tasks:
    # get the test set to evaluate on
    eval_splits = ["dev"] if task == "MSMARCO" else ["test"]
    evaluation = mteb.MTEB(
        tasks=[task], task_langs=["en"]
    )  # Remove "en" to run all available languages
    evaluation.run(
        model, output_folder="results", eval_splits=eval_splits
    )

```
## 2. Submit to MTEB Leaderboard  
- 评估结果应存储在 `results/{model_name}/{model_revision}` 。
- 运行以下 shell 命令以创建 model_card.md。
```python
In [ ]:

!mteb create_meta --results_folder results/{model_name}/{model_revision} --output_path model_card.md
```
- 对于该模型的 readme 已存在的情况：
```python
In [ ]:

# !mteb create_meta --results_folder results/{model_name}/{model_revision} --output_path model_card.md --from_existing your_existing_readme.md 
```
- 将 model_card.md 的内容复制粘贴到您在 HF Hub 上的模型 README.md 的顶部。现在放松并等待排行榜的每日刷新。您的模型很快就会出现！
## 3. Partially Evaluate  
- 不需要完成所有任务就可以进入排行榜。
- 例如，您微调模型在聚类方面的能力。您只关心您的模型在聚类方面的表现，而不关心其他任务。那么，您只需测试其在 MTEB 的聚类任务上的性能，并将其提交到排行榜即可。
```python
In [ ]:

TASK_LIST_CLUSTERING = [
    "ArxivClusteringP2P",
    "ArxivClusteringS2S",
    "BiorxivClusteringP2P",
    "BiorxivClusteringS2S",
    "MedrxivClusteringP2P",
    "MedrxivClusteringS2S",
    "RedditClustering",
    "RedditClusteringP2P",
    "StackExchangeClustering",
    "StackExchangeClusteringP2P",
    "TwentyNewsgroupsClustering",
]
```
- 运行仅包含聚类任务的评估：
```python
In [ ]:

evaluation = mteb.MTEB(tasks=TASK_LIST_CLUSTERING)

results = evaluation.run(model, output_folder="results")
```
- 重复步骤 2 以提交您的模型。排行榜刷新后，您可以在排行榜的“聚类”部分找到您的模型。
