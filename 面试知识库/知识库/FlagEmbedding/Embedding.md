
## BAAI General Embedding (BGE)
- BGE 代表 BAAI 通用嵌入，它是一系列由北京人工智能研究院（BAAI）开发和发布的嵌入模型。BAAI 通用嵌入（BGE）提供了一系列开源模型，可以满足各种需求。
- 本质：将词语映射到低维空间中的向量，该空间能够捕捉语义和关系信息。
-  BERT、RoBERTa 和 GPT 等更复杂的模型在捕捉复杂词语关系和上下文中表现出色
- GitHub 上的 FlagEmbedding 维护了对 BGE API 及其相关用法的全面支持。
### BGE
| Model                                                         | Language | Parameters | Model Size | Description       | Base Model |
| ------------------------------------------------------------- | -------- | ---------- | ---------- | ----------------- | ---------- |
| [BAAI/bge-large-en](https://huggingface.co/BAAI/bge-large-en) | English  | 500M       | 1.34 GB    | 嵌入模型，将文本映射到向量     | BERT       |
| [BAAI/bge-base-en](https://huggingface.co/BAAI/bge-base-en)   | English  | 109M       | 438 MB     | 一个基础规模模型，但具有类似的能力 | BERT       |
| [BAAI/bge-small-en](https://huggingface.co/BAAI/bge-small-en) | English  | 33.4M      | 133 MB     | 一个小规模模型，但性能具有竞争力  | BERT       |
| [BAAI/bge-large-zh](https://huggingface.co/BAAI/bge-large-zh) | Chinese  | 326M       | 1.3 GB     | 嵌入模型，将文本映射到向量     | BERT       |
| [BAAI/bge-base-zh](https://huggingface.co/BAAI/bge-base-zh)   | Chinese  | 102M       | 409 MB     | 一个基础规模模型，但具有类似的能力 | BERT       |
| [BAAI/bge-small-zh](https://huggingface.co/BAAI/bge-small-zh) | Chinese  | 24M        | 95.8 MB    | 一个小规模模型，但性能具有竞争力  | BERT       |

### BGE v1.5
BGE 1.5 缓解相似度分布问题，无需指令即可增强检索能力

|Model|Language|Parameters|Model Size|Description|Base Model|
|---|---|---|---|---|---|
|[BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5)|English|335M|1.34 GB|version 1.5 with more reasonable similarity distribution|BERT|
|[BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5)|English|109M|438 MB|version 1.5 with more reasonable similarity distribution|BERT|
|[BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5)|English|33.4M|133 MB|version 1.5 with more reasonable similarity distribution|BERT|
|[BAAI/bge-large-zh-v1.5](https://huggingface.co/BAAI/bge-large-zh-v1.5)|Chinese|326M|1.3 GB|version 1.5 with more reasonable similarity distribution|BERT|
|[BAAI/bge-base-zh-v1.5](https://huggingface.co/BAAI/bge-base-zh-v1.5)|Chinese|102M|409 MB|version 1.5 with more reasonable similarity distribution|BERT|
|[BAAI/bge-small-zh-v1.5](https://huggingface.co/BAAI/bge-small-zh-v1.5)|Chinese|24M|95.8 MB|version 1.5 with more reasonable similarity distribution|BERT|
### BGE M3
- 多功能性：同时执行嵌入模型的三种常见检索功能：密集检索、多向量检索和稀疏检索。
- 多语言支持
- 多粒度：可以处理不同粒度的输入，范围从短句到长达 8192 个 token 的长文档。
```python
BGEM3FlagModel.encode(
    sentences, 
    batch_size=12, 
    max_length=8192, 
    return_dense=True, 
    return_sparse=False, 
    return_colbert_vecs=False
)

It returns a dictionary like:  
它返回一个类似字典的结构：

{
    'dense_vecs':       # array of dense embeddings of inputs if return_dense=True, otherwise None,
    'lexical_weights':  # array of dictionaries with keys and values are ids of tokens and their corresponding weights if return_sparse=True, otherwise None,
    'colbert_vecs':     # array of multi-vector embeddings of inputs if return_cobert_vecs=True, otherwise None,'
}
```
### BGE Multilingual Gemma2
- BGE 多语言 Gemma2 是一个基于 LLM 的多语言嵌入模型。(基于 Google 的 Gemma 2 架构开发)
### BGE ICL
| Model                                                     | Language    | Parameters | Model Size | Description                | Base Model |
| --------------------------------------------------------- | ----------- | ---------- | ---------- | -------------------------- | ---------- |
| [BAAI/bge-en-icl](https://huggingface.co/BAAI/bge-en-icl) | English  英语 | 7.11B      | 28.5 GB    | 基于LLM的英语嵌入模型，具有出色的上下文学习能力。 | Mistral-7B |
```python
documents = [
    "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
    "Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments."
]

examples = [
    {
        'instruct': 'Given a web search query, retrieve relevant passages that answer the query.',
        'query': 'what is a virtual interface',
        'response': "A virtual interface is a software-defined abstraction that mimics the behavior and characteristics of a physical network interface. It allows multiple logical network connections to share the same physical network interface, enabling efficient utilization of network resources. Virtual interfaces are commonly used in virtualization technologies such as virtual machines and containers to provide network connectivity without requiring dedicated hardware. They facilitate flexible network configurations and help in isolating network traffic for security and management purposes."
    },
    {
        'instruct': 'Given a web search query, retrieve relevant passages that answer the query.',
        'query': 'causes of back pain in female for a week',
        'response': "Back pain in females lasting a week can stem from various factors. Common causes include muscle strain due to lifting heavy objects or improper posture, spinal issues like herniated discs or osteoporosis, menstrual cramps causing referred pain, urinary tract infections, or pelvic inflammatory disease. Pregnancy-related changes can also contribute. Stress and lack of physical activity may exacerbate symptoms. Proper diagnosis by a healthcare professional is crucial for effective treatment and management."
    }
]

queries = ["how much protein should a female eat", "summit define"]
```
# BGE Explanation
## 1. Encode sentences
```python
from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")
model = AutoModel.from_pretrained("BAAI/bge-base-en-v1.5")

sentences = ["embedding", "I love machine learning and nlp"]
```
- bge-base-en-v1.5,使用 BERT-base 作为基础模型，具有 12 个编码层和 768 的隐藏维度。

对句子进行分词。
```python
inputs = tokenizer(
    sentences, 
    padding=True, 
    truncation=True, 
    return_tensors='pt', 
    max_length=512
)
inputs

{'input_ids': tensor([[  101,  7861,  8270,  4667,   102,     0,     0,     0,     0],
        [  101,  1045,  2293,  3698,  4083,  1998, 17953,  2361,   102]]), 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1]])}
```
从结果中，我们可以看到每个句子都以标记 101 开始并以标记 102 结束，这两个标记是 BERT 中使用的 `[CLS]` 和 `[SEP]` 特殊标记。
```python
last_hidden_state = model(**inputs, return_dict=True).last_hidden_state
last_hidden_state.shape
torch.Size([2, 9, 768])
```
这里我们实现了池化函数，有两种选择：使用 `[CLS]` 的最后一个隐藏状态，或者对整个最后一个隐藏状态进行平均池化。
```python
def pooling(last_hidden_state: torch.Tensor, pooling_method='cls', attention_mask: torch.Tensor = None):
    if pooling_method == 'cls':
        return last_hidden_state[:, 0]
    elif pooling_method == 'mean':
        s = torch.sum(last_hidden_state * attention_mask.unsqueeze(-1).float(), dim=1)
        d = attention_mask.sum(dim=1, keepdim=True).float()
        return s / d
```
不同于更常用的平均池化，BGE 被训练使用 `[CLS]` 的最后一个隐藏状态作为句子嵌入：
```python
sentence_embeddings = model_output[0][:, 0]
```
如果您使用平均池化，性能将显著下降。因此，请确保使用正确的方法来获取句子向量。
```python
embeddings = pooling(
    last_hidden_state, 
    pooling_method='cls', 
    attention_mask=inputs['attention_mask']
)
embeddings.shape

torch.Size([2, 768])
```
将它们组装在一起，我们得到完整的编码函数：
```python
def _encode(sentences, max_length=512, convert_to_numpy=True):

    # handle the case of single sentence and a list of sentences
    input_was_string = False
    if isinstance(sentences, str):
        sentences = [sentences]
        input_was_string = True

    inputs = tokenizer(
        sentences, 
        padding=True, 
        truncation=True, 
        return_tensors='pt', 
        max_length=max_length
    )

    last_hidden_state = model(**inputs, return_dict=True).last_hidden_state
    
    embeddings = pooling(
        last_hidden_state, 
        pooling_method='cls', 
        attention_mask=inputs['attention_mask']
    )

    # normalize the embedding vectors
    embeddings = torch.nn.functional.normalize(embeddings, dim=-1)

    # convert to numpy if needed
    if convert_to_numpy:
        embeddings = embeddings.detach().numpy()

    return embeddings[0] if input_was_string else embeddings
```
## 2. Comparison
```python
from FlagEmbedding import FlagModel

model = FlagModel('BAAI/bge-base-en-v1.5')

embeddings = model.encode(sentences)
print(f"Embeddings:\n{embeddings}")

scores = embeddings @ embeddings.T
print(f"Similarity scores:\n{scores}")

Embeddings:
[[ 1.4549762e-02 -9.6840411e-03  3.7761475e-03 ... -8.5092714e-04
   2.8417887e-02  6.3214332e-02]
 [ 3.3924331e-05 -3.2998275e-03  1.7206438e-02 ...  3.5703944e-03
   1.8721525e-02 -2.0371782e-02]]
Similarity scores:
[[0.9999997 0.6077381]
 [0.6077381 0.9999999]]
```
# BGE-M3
## 1. BGE-M3 structure
```python
from transformers import AutoTokenizer, AutoModel
import torch, os

tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
raw_model = AutoModel.from_pretrained("BAAI/bge-m3")
```
- BGE-M3 的基础模型是 XLM-RoBERTa-large，它是 RoBERTa 的多语言版本。
```python
raw_model.eval()

XLMRobertaModel(
  (embeddings): XLMRobertaEmbeddings(
    (word_embeddings): Embedding(250002, 1024, padding_idx=1)
    (position_embeddings): Embedding(8194, 1024, padding_idx=1)
    (token_type_embeddings): Embedding(1, 1024)
    (LayerNorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
    (dropout): Dropout(p=0.1, inplace=False)
  )
  (encoder): XLMRobertaEncoder(
    (layer): ModuleList(
      (0-23): 24 x XLMRobertaLayer(
        (attention): XLMRobertaAttention(
          (self): XLMRobertaSelfAttention(
            (query): Linear(in_features=1024, out_features=1024, bias=True)
            (key): Linear(in_features=1024, out_features=1024, bias=True)
            (value): Linear(in_features=1024, out_features=1024, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (output): XLMRobertaSelfOutput(
            (dense): Linear(in_features=1024, out_features=1024, bias=True)
            (LayerNorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (intermediate): XLMRobertaIntermediate(
          (dense): Linear(in_features=1024, out_features=4096, bias=True)
          (intermediate_act_fn): GELUActivation()
        )
        (output): XLMRobertaOutput(
          (dense): Linear(in_features=4096, out_features=1024, bias=True)
          (LayerNorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
    )
  )
  (pooler): XLMRobertaPooler(
    (dense): Linear(in_features=1024, out_features=1024, bias=True)
    (activation): Tanh()
  )
)
```
## 2. Multi-Functionality
### 2.1 Dense Retrieval
- 使用特殊标记[CLS]的归一化隐藏状态作为嵌入：$eq=norm(Hq[0])$
- 然后计算查询与段落之间的相关性得分：$s_{dense}=f_{sim}(e_p,e_q)$
- 其中 ep,eq 分别是段落和查询的嵌入向量。
- fsim 是计算两个嵌入相似度的得分函数（例如内积和 L2 距离）
### 2.2 Sparse Retrieval
- 将 `return_sparse` 设置为 true 以使模型返回稀疏向量。如果一个词项在句子中多次出现，我们只保留其最大权重。
- BGE-M3 通过添加一个线性层和紧随隐藏状态的 ReLU 激活函数来生成稀疏嵌入：$w_{qt}=Relu(W_{lex}^TH_q[i])$
- 其中 $W_{lex}$ 代表线性层的权重， Hq[i] 是编码器对 $i^{th}$ 令牌的输出。
```python
output_1 = model.encode(sentences_1, return_sparse=True)
output_2 = model.encode(sentences_2, return_sparse=True)

# you can see the weight for each token:
print(model.convert_id_to_token(output_1['lexical_weights']))

[{'What': 0.08362077, 'is': 0.081469566, 'B': 0.12964639, 'GE': 0.25186998, 'M': 0.17001738, '3': 0.26957875, '?': 0.040755156}, {'De': 0.050144322, 'fin': 0.13689369, 'ation': 0.045134712, 'of': 0.06342201, 'BM': 0.25167602, '25': 0.33353207}]
```
- 基于查询和段落中标记的权重，它们之间的相关性得分是通过查询和段落中共同存在术语的联合重要性来计算的：$s_{lex}=∑_{t∈q∩p}(w_{qt}∗w_{pt})$ where wqt,wpt are the importance weights of each co-existed term t in query and passage, respectively.  
 ```python
# compute the scores via lexical mathcing
s_lex_10_20 = model.compute_lexical_matching_score(output_1['lexical_weights'][0], output_2['lexical_weights'][0])
s_lex_10_21 = model.compute_lexical_matching_score(output_1['lexical_weights'][0], output_2['lexical_weights'][1])

print(s_lex_10_20)
print(s_lex_10_21)

0.19554448500275612
0.00880391988903284
``` 
### 2.3 Multi-Vector  2.3 多向量

- 多向量方法利用整个输出嵌入来表示查询 Eq 和段落 Ep 的表示。$E_q=norm(W_{mul}^TH_q)$
Wmul 是可学习的投影矩阵。
```python
output_1 = model.encode(sentences_1, return_dense=True, return_sparse=True, return_colbert_vecs=True)
output_2 = model.encode(sentences_2, return_dense=True, return_sparse=True, return_colbert_vecs=True)

print(f"({len(output_1['colbert_vecs'][0])}, {len(output_1['colbert_vecs'][0][0])})")
print(f"({len(output_2['colbert_vecs'][0])}, {len(output_2['colbert_vecs'][0][0])})")

(8, 1024)
(30, 1024)

```
- 在 ColBert 之后，我们使用后期交互来计算细粒度相关性得分：$s_{mul}=\frac{1}{N}\sum_{N}^{i=1} max_{j=1}^ME_q[i]⋅E_p^T[j]$
- Eq,Ep 分别是查询和段落的全局输出嵌入。 
- 这是每个 v∈Eq 与 Ep 中向量最大相似度平均值的汇总
```python
s_mul_10_20 = model.colbert_score(output_1['colbert_vecs'][0], output_2['colbert_vecs'][0]).item()
s_mul_10_21 = model.colbert_score(output_1['colbert_vecs'][0], output_2['colbert_vecs'][1]).item()

print(s_mul_10_20)
print(s_mul_10_21)

0.7796662449836731
0.4621177911758423
```
### 2.4 Hybrid Ranking 
- BGE-M3 的多功能性提供了混合排名的可能性以改善检索。首先，由于多向量方法成本高昂，我们可以通过密集或稀疏方法之一检索候选结果。然后，为了获得最终结果，我们可以根据综合相关性分数重新排名候选结果：$s_{rank}=w_1⋅s_{dense}+w_2⋅s_{lex}+w_3⋅s_{mul}$
- 在 w1,w2 和 w3 选择的值取决于下游场景（这里 1/3 只是为了演示）。
```python
s_rank_10_20 = 1/3 * s_dense[0][0] + 1/3 * s_lex_10_20 + 1/3 * s_mul_10_20
s_rank_10_21 = 1/3 * s_dense[0][1] + 1/3 * s_lex_10_21 + 1/3 * s_mul_10_21

print(s_rank_10_20)
print(s_rank_10_21)

0.5337047390639782
0.27280585498859483
```
#   BGE-EN-ICL
- 在这个教程中，我们将介绍 BGE-EN-ICL，这是一个基于LLM的嵌入模型，具有强大的零样本和少样本嵌入能力。
## 1. BGE-EN-ICL structure  BGE-EN-ICL 结构
```python
from transformers import AutoTokenizer, AutoModel
import torch, os

tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-en-icl")
raw_model = AutoModel.from_pretrained("BAAI/bge-en-icl")

sentences = ["embedding", "I love machine learning and nlp"]
```
- 不同于之前的 BGE 嵌入模型，这些模型仅使用编码器，BGE-EN-ICL **仅使用解码器**作为基础模型，即LLM Mistral-7B。
```python
raw_model.eval()

MistralModel(
  (embed_tokens): Embedding(32003, 4096)
  (layers): ModuleList(
    (0-31): 32 x MistralDecoderLayer(
      (self_attn): MistralSdpaAttention(
        (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
        (k_proj): Linear(in_features=4096, out_features=1024, bias=False)
        (v_proj): Linear(in_features=4096, out_features=1024, bias=False)
        (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
        (rotary_emb): MistralRotaryEmbedding()
      )
      (mlp): MistralMLP(
        (gate_proj): Linear(in_features=4096, out_features=14336, bias=False)
        (up_proj): Linear(in_features=4096, out_features=14336, bias=False)
        (down_proj): Linear(in_features=14336, out_features=4096, bias=False)
        (act_fn): SiLU()
      )
      (input_layernorm): MistralRMSNorm((4096,), eps=1e-05)
      (post_attention_layernorm): MistralRMSNorm((4096,), eps=1e-05)
    )
  )
  (norm): MistralRMSNorm((4096,), eps=1e-05)
)
```
## 2. New Pooling Method 
- BERT-like 编码器仅网络因其双向注意力结构而具有强大的表示学习能力。一些先前工作在嵌入训练阶段将单向注意力替换为双向注意力。但这可能与模型的预训练设计不匹配，这可能会损害其在上下文中的学习和生成特性。
- 因此，BGE-EN-ICL 引入了一个 [EOS] 令牌的输出嵌入来解决此问题。
```python
inputs = tokenizer(
    sentences,
    padding=True,
    return_tensors='pt',
)
inputs

{'input_ids': tensor([[    0,     0,     0,     0,     0,     0,     1, 28643,     2],
        [    1,   315,  2016,  5599,  5168,   304,   307, 12312,     2]]), 'attention_mask': tensor([[0, 0, 0, 0, 0, 0, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1]])}

last_hidden_state = raw_model(**inputs, return_dict=True).last_hidden_state
last_hidden_state.shape

torch.Size([2, 9, 4096])
```
- 最后标记/[EOS]池化方法可以描述为：
- 考虑到标记化输入序列 T:[BOS],t1,...,tN 被发送到 LLM：$h_t=LLM(T)[EOS]$，其中 ht 代表从特殊标记 [EOS] 的输出嵌入中提取的文本嵌入
```python
def last_token_pool(last_hidden_states: torch.Tensor,
                    attention_mask: torch.Tensor) -> torch.Tensor:
    
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

In [10]:

embeddings = last_token_pool(
    last_hidden_state,  
    attention_mask=inputs['attention_mask']
)
embeddings.shape

Out[10]:

torch.Size([2, 4096])
```
## 3. In-Context Learning  
- BGE-EN-ICL 在嵌入模型中集成 LLM 的强上下文学习，同时仍保持强大的零样本嵌入能力。
- 对于零样本推理，它与 BGE v1&1.5 完全相同。对于少样本推理，请使用以下方法：
```python
In [11]:

examples = [
    {
        'instruct': 'Given a web search query, retrieve relevant passages that answer the query.',
        'query': 'what is a virtual interface',
        'response': "A virtual interface is a software-defined abstraction that mimics the behavior and characteristics of a physical network interface. It allows multiple logical network connections to share the same physical network interface, enabling efficient utilization of network resources. Virtual interfaces are commonly used in virtualization technologies such as virtual machines and containers to provide network connectivity without requiring dedicated hardware. They facilitate flexible network configurations and help in isolating network traffic for security and management purposes."
    },
    {
        'instruct': 'Given a web search query, retrieve relevant passages that answer the query.',
        'query': 'causes of back pain in female for a week',
        'response': "Back pain in females lasting a week can stem from various factors. Common causes include muscle strain due to lifting heavy objects or improper posture, spinal issues like herniated discs or osteoporosis, menstrual cramps causing referred pain, urinary tract infections, or pelvic inflammatory disease. Pregnancy-related changes can also contribute. Stress and lack of physical activity may exacerbate symptoms. Proper diagnosis by a healthcare professional is crucial for effective treatment and management."
    }
]

queries = ["how much protein should a female eat", "summit define"]
documents = [
    "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
    "Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments."
]

In [16]:

from FlagEmbedding import FlagICLModel

model = FlagICLModel('BAAI/bge-en-icl', 
                     examples_for_task=examples,  # set `examples_for_task=None` to use model without examples
                     examples_instruction_format="<instruct>{}\n<query>{}\n<response>{}", # specify the format to use examples_for_task
                     devices=[0],
                    )

embeddings_1 = model.encode_queries(queries)
embeddings_2 = model.encode_corpus(documents)
similarity = embeddings_1 @ embeddings_2.T

print(similarity)

[[0.6064 0.302 ]
 [0.257  0.5366]]