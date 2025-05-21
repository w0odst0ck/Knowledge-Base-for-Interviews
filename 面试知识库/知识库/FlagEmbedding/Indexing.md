#   Faiss GPU
无缝地结合 Faiss 和 GPU。
## 1. Installation 
## 2. Data Preparation
- 创建两个数据集，包含语料库和查询的“伪造嵌入”
```python
In [1]:

import faiss
import numpy as np

dim = 768
corpus_size = 1000
# np.random.seed(111)

corpus = np.random.random((corpus_size, dim)).astype('float32')
```
## 3. Create Index on CPU  
### Option 1:  
- Faiss 通过直接初始化提供大量索引选择：
```python
In [2]:

# first build a flat index (on CPU)
index = faiss.IndexFlatIP(dim)

### Option 2:  选项 2:

Besides the basic index class, we can also use the index_factory function to produce composite Faiss index.  
除了基本的索引类之外，我们还可以使用 index_factory 函数来生成复合 Faiss 索引。

In [3]:

index = faiss.index_factory(dim, "Flat", faiss.METRIC_L2)
```
## 4. Build GPU Index and Search  
- 所有 GPU 索引都是使用 `StandardGpuResources` 对象构建的。它包含每个正在使用的 GPU 所需的所有资源。默认情况下，它将分配总 VRAM 的 18%作为临时暂存空间。
 - `GpuClonerOptions` 和 `GpuMultipleClonerOptions` 对象在从 CPU 创建索引到 GPU 时是可选的。它们用于调整 GPU 存储对象的方式。
### Single GPU:  
```python
In [4]:

# use a single GPU
rs = faiss.StandardGpuResources()
co = faiss.GpuClonerOptions()

# then make it to gpu index
index_gpu = faiss.index_cpu_to_gpu(provider=rs, device=0, index=index, options=co)

In [5]:

%%time
index_gpu.add(corpus)
D, I = index_gpu.search(corpus, 4)

CPU times: user 5.31 ms, sys: 6.26 ms, total: 11.6 ms
Wall time: 8.94 ms
```
### All Available GPUs 
- 如果您的系统包含多个 GPU，Faiss 提供了部署所有可用 GPU 的选项。您可以通过 `GpuMultipleClonerOptions` 来控制它们的用法，例如是否在 GPU 之间分片或复制索引。
```python
In [7]:

# cloner options for multiple GPUs
co = faiss.GpuMultipleClonerOptions()

index_gpu = faiss.index_cpu_to_all_gpus(index=index, co=co)

In [8]:

%%time
index_gpu.add(corpus)
D, I = index_gpu.search(corpus, 4)

CPU times: user 29.8 ms, sys: 26.8 ms, total: 56.6 ms
Wall time: 33.9 ms
```
### Multiple GPUs  
- 也有选择使用多个 GPU 但不全部使用的选项：
```python
In [10]:

ngpu = 4
resources = [faiss.StandardGpuResources() for _ in range(ngpu)]
```
创建 GpuResources 和 divices 的向量，然后将它们传递给 index_cpu_to_gpu_multiple()函数。
```python
In [11]:

vres = faiss.GpuResourcesVector()
vdev = faiss.Int32Vector()
for i, res in zip(range(ngpu), resources):
    vdev.push_back(i)
    vres.push_back(res)
index_gpu = faiss.index_cpu_to_gpu_multiple(vres, vdev, index)

In [12]:

%%time
index_gpu.add(corpus)
D, I = index_gpu.search(corpus, 4)

CPU times: user 3.49 ms, sys: 13.4 ms, total: 16.9 ms
Wall time: 9.03 ms
```
## 5. Results  
- 检查：
```python
In [13]:

# The nearest neighbor of each vector in the corpus is itself
assert np.all(corpus[:] == corpus[I[:, 0]])
```
并且相应的距离应为 0。
```python
In [14]:

print(D[:3])

[[  0.       111.30057  113.2251   113.342316]
 [  0.       111.158875 111.742325 112.09038 ]
 [  0.       116.44429  116.849915 117.30502 ]]
```
#   Faiss Indexes  
- 介绍 Faiss 中几个广泛使用的索引，以及如何使用它们。
## Preparation  

 - CPU 使用
```python
In [1]:

%pip install faiss-cpu

For GPU on Linux x86_64 system, use Conda:  
对于 Linux x86_64 系统上的 GPU，使用 Conda：

`conda install -c pytorch -c nvidia faiss-gpu=1.8.0`

In [2]:

import faiss
import numpy as np

np.random.seed(768)

data = np.random.random((1000, 128))
```
## 1. `IndexFlat*`
- 平面索引是极其基本的索引结构。它不对传入的向量进行任何预处理。所有向量都直接存储，不进行压缩或量化。因此，平面索引不需要进行训练。
- 在搜索时，Flat 索引将按顺序解码所有向量并计算与查询向量的相似度得分。因此，Flat 索引保证了结果的全局最优。
- 平面索引族很小：仅有 `IndexFlatL2` 和 `IndexFlatIP` ，它们只是欧几里得距离和内积相似度指标的不同。

Usage:  使用方法：
```python
In [3]:

d = 128  # dimension of the vector
k = 3    # number of nearest neighbors to search
# just simply create the index and add all the data
index = faiss.IndexFlatL2(d)
index.add(data)
```
Sanity check:  
```python
In [4]:

# search for the k nearest neighbor for the first element in data
D, I = index.search(data[:1], k)

print(f"closest elements: {I}")
print(f"distance: {D}")

closest elements: [[  0 471 188]]
distance: [[ 0.       16.257435 16.658928]]
```
- 平面索引保证了完美的质量，但速度极慢。它在小型数据集或速度不是关键因素的情况下表现良好。
- 但是，当速度很重要时怎么办？不可能面面俱到。因此，我们希望有一些索引，它们在尽可能小的牺牲质量的情况下提高速度->近似最近邻（ANN）算法
- 现在，我们将介绍一些在向量搜索中使用的流行 ANN 方法。
## 2. `IndexIVF*`

### Intro  
- 倒排文件平面（IVF）索引是一种广泛接受的技术，通过使用 k-means 或 Voronoi 图在整体空间中创建多个单元格（或者说，簇）来加速搜索。然后，当给定一个查询时，将搜索一定数量的最近单元格。之后，将在这些单元格中搜索与查询最近的 `k` 个元素。
- `quantizer` 是另一个将向量分配到倒排列表的索引/量化器。
- `nlist` 是要划分的空间中的单元格数量。
- `nprob` 是查询时搜索要访问的最近单元格的数量。
### Tradeoff  
- 增加 `nlist` 将缩小每个单元格的大小，这会加快搜索过程。但较小的覆盖范围将牺牲准确性并增加上述边缘/表面问题发生的可能性。
- 增加 `nprob` 将具有更广泛的范围，通过牺牲速度来优先考虑搜索质量。
### Shortage  
- 当查询向量落在细胞边缘/表面时可能会出现问题。可能最接近的元素落在相邻细胞中，但由于 `nprob` 不够大，可能没有被考虑。
### Example  
```python
In [5]:

nlist = 5
nprob = 2

# the quantizer defines how to store and compare the vectors
quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFFlat(quantizer, d, nlist)

# note different from flat index, IVF index first needs training to create the cells
index.train(data)
index.add(data)

In [6]:

# set nprob before searching
index.nprobe = 8
D, I = index.search(data[:1], k)

print(f"closest elements: {I}")
print(f"distance: {D}")

closest elements: [[  0 471 188]]
distance: [[ 0.       16.257435 16.658928]]
```
## 3. `IndexHNSW*`
### Intro  
- Hierarchical Navigable Small World (HNSW)
- 分层可导航小世界（HNSW）索引是一种基于图的方法，它是可导航小世界（NSW）的扩展。它构建了一个多层图，其中节点（向量）根据其邻近性连接，形成“小世界”结构，允许高效地在该空间中导航。
- `M` 是图中每个向量拥有的邻居数量。
- `efConstruction` 是构建索引时探索的入口点数量。
- `efSearch` 是在搜索时探索的入口点数量。
### Tradeoff  权衡
- 增加 `M` 或 `efSearch` 将使保真度更高，但需要更长的合理时间。更大的 `efConstruction` 主要增加索引构建时间。
- HNSW 搜索质量高且速度快。但由于图结构，它占用内存较多。将 `M` 扩展会导致内存使用量线性增加。
- 请注意，HNSW 索引不支持向量的删除，因为删除节点将破坏图结构。
- 因此，当 RAM 不是限制因素时，HNSW 是一个很好的索引选择。
### Example  
```python
In [7]:

M = 32
ef_search = 16
ef_construction = 32

index = faiss.IndexHNSWFlat(d, M)
# set the two parameters before adding data
index.hnsw.efConstruction = ef_construction
index.hnsw.efSearch = ef_search

index.add(data)

In [8]:

D, I = index.search(data[:1], k)

print(f"closest elements: {I}")
print(f"distance: {D}")

closest elements: [[  0 471 188]]
distance: [[ 0.       16.257435 16.658928]]
```
### 4. `IndexLSH`
### Intro  
- Locality Sensitive Hashing (LSH) 
- 局部敏感哈希（LSH）是一种将数据点哈希到桶中的 ANN 方法。虽然像字典/哈希表这样的哈希函数的已知用例试图避免哈希冲突，但 LSH 试图最大化哈希冲突。相似向量将被分组到相同的哈希桶中。
- 在 Faiss 中， `IndexLSH` 是一个平面索引，具有二进制码。向量被哈希成二进制码，并通过汉明距离进行比较。
- `nbits` 可以被视为散列向量的“分辨率”。
### Tradeoff  
- 增加 `nbits` 可以以更多内存和更长的搜索时间为代价获得更高的保真度。
- LSH 在使用较大的 `d` 时遭受维度灾难。为了获得相似搜索质量，需要将 `nbits` 的值放大以保持搜索质量。
### Shortage  
- LSH 通过合理的质量牺牲来加速搜索时间。但这仅适用于小维度 `d` 。即使是 128 也已经太大，不适合 LSH。因此，对于由基于 transformer 的嵌入模型生成的向量，LSH 索引不是一个常见的选择。

### Example  
```python
In [9]:

nbits = d * 8

index = faiss.IndexLSH(d, nbits)
index.train(data)
index.add(data)

In [10]:

D, I = index.search(data[:1], k)

print(f"closest elements: {I}")
print(f"distance: {D}")

closest elements: [[  0 471 392]]
distance: [[  0. 197. 199.]]
```
#   Faiss Quantizers  
- 介绍 Faiss 中的量化器对象及其使用方法。
## Preparation  
- CPU 
```python
In [ ]:

%pip install faiss-cpu

For GPU on Linux x86_64 system, use Conda:  
对于 Linux x86_64 系统上的 GPU，使用 Conda：

`conda install -c pytorch -c nvidia faiss-gpu=1.8.0`

In [1]:

import faiss
import numpy as np

np.random.seed(768)

data = np.random.random((1000, 128))
```
## 1. Scalar Quantizer  
- 向量的嵌入数据类型通常是 32 位浮点数。标量量化将 32 位浮点数表示转换为，例如，8 位整数。因此，大小减少了 4 倍。这样，我们可以将其视为将每个维度分配到 256 个桶中。

| Name  姓名               | Class  类                | Parameters  参数                                                              |
| ---------------------- | ----------------------- | --------------------------------------------------------------------------- |
| `ScalarQuantizer`      | Quantizer class  量化器类   | `d` : 向量维度  <br>`qtype` : 将地图维度映射到 2qtype 个簇                                |
| `IndexScalarQuantizer` | Flat index class  平面索引类 | `d` : 向量维度  <br>`qtype` : 将地图维度映射到 2qtype 个簇  <br>`metric` : 相似度度量（L2 或 IP） |

| `IndexIVFScalarQuantizer` | IVF 索引类 | `d` : 向量维度  
`nlist`: number of cells/clusters to partition the inverted file space  
`nlist` ：将倒排文件空间分区成单元格/聚类的数量  
`qtype`: map dimension into 2qtype clusters  
`qtype` : 将地图维度映射到 2qtype 个簇  
`metric`: similarity metric (L2 or IP)  
`metric` : 相似度度量（L2 或 IP）
- 量化类对象用于在添加到索引之前压缩数据。平面索引类对象和 IVF 索引类对象可以直接用作索引。量化将自动进行。

### Scalar Quantizer  标量量化器
```python
In [2]:

d = 128
qtype = faiss.ScalarQuantizer.QT_8bit

quantizer = faiss.ScalarQuantizer(d, qtype)

quantizer.train(data)
new_data = quantizer.compute_codes(data)

print(new_data[0])

[156 180  46 226  13 130  41 187  63 251  16 199 205 166 117 122 214   2
 206 137  71 186  20 131  59  57  68 114  35  45  28 210  27  93  74 245
 167   5  32  42  44 128  10 189  10  13  42 162 179 221 241 104 205  21
  70  87  52 219 172 138 193   0 228 175 144  34  59  88 170   1 233 220
  20  64 245 241   5 161  41  55  30 247 107   8 229  90 201  10  43 158
 238 184 187 114 232  90 116 205  14 214 135 158 237 192 205 141 232 176
 124 176 163  68  49  91 125  70   6 170  55  44 215  84  46  48 218  56
 107 176]
```
### Scalar Quantizer Index 
```python
In [3]:

d = 128
k = 3
qtype = faiss.ScalarQuantizer.QT_8bit
# nlist = 5

index = faiss.IndexScalarQuantizer(d, qtype, faiss.METRIC_L2)
# index = faiss.IndexIVFScalarQuantizer(d, nlist, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_L2)

index.train(data)
index.add(data)

In [4]:

D, I = index.search(data[:1], k)

print(f"closest elements: {I}")
print(f"distance: {D}")

closest elements: [[  0 471 188]]
distance: [[1.6511828e-04 1.6252808e+01 1.6658131e+01]]
```
## 2. Product Quantizer 
- 当速度和内存是搜索中的关键因素时，产品量化器成为首选。它是有效减少内存大小的量化器之一。
- PQ 的第一步是将维度为 `d` 的原始向量划分为维度为 `d/m` 的更小、低维的子向量。其中 `m` 是子向量的数量。
- 然后使用聚类算法创建具有固定数量质心的代码簿。
- 接下来，每个向量的子向量被替换为其对应代码簿中最近质心的索引。现在，每个向量将只存储索引而不是完整向量。
- 当计算查询向量之间的距离时。仅计算到代码簿中质心的距离，从而实现快速近似最近邻搜索。

| Name  姓名           | Class  类                  | Parameters  参数                                                                                                                                                                      |
| ------------------ | ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ProductQuantizer` | Quantizer class  量化器类     | `d` : 向量维度  <br>`M` : D % M 等于 0 的子向量的数量  <br>`nbits` : 每个子量化器的位数，因此每个包含 2nbits 中心点                                                                                                 |
| `IndexPQ`          | Flat index class  平面索引类   | `d` : 向量维度    <br>`M` : D % M 等于 0 的子向量的数量  <br>`nbits` : 每个子量化器的位数，因此每个包含 2nbits 中心点  <br>`metric` : 相似度度量（L2 或 IP）                                                                |
| `IndexIVFPQ`       | IVF index class  IVF 指数类别 | `quantizer` ：计算距离相位的量化器。  <br>`d` : 向量维度  <br>`nlist` ：将倒排文件空间分区成单元格/聚类的数量  <br>`M` : D % M 等于 0 的子向量的数量  <br>`nbits` : 每个子量化器的位数，因此每个包含 2nbits 中心点   <br>`metric` : 相似度度量（L2 或 IP） |

### Product Quantizer  
```python
In [5]:

d = 128
M = 8
nbits = 4

quantizer = faiss.ProductQuantizer(d, M, nbits)

quantizer.train(data)
new_data = quantizer.compute_codes(data)

print(new_data.max())
print(new_data[:2])

255
[[ 90 169 226  45]
 [ 33  51  34  15]]
```
### Product Quantizer Index  
```python
In [6]:

index = faiss.IndexPQ(d, M, nbits, faiss.METRIC_L2)

index.train(data)
index.add(data)

In [7]:

D, I = index.search(data[:1], k)

print(f"closest elements: {I}")
print(f"distance: {D}")

closest elements: [[  0 946 330]]
distance: [[ 8.823908 11.602461 11.746731]]
```
### Product Quantizer IVF Index  
```python
In [8]:

nlist = 5

quantizer = faiss.IndexFlat(d, faiss.METRIC_L2)
index = faiss.IndexIVFPQ(quantizer, d, nlist, M, nbits, faiss.METRIC_L2)

index.train(data)
index.add(data)

In [9]:

D, I = index.search(data[:1], k)

print(f"closest elements: {I}")
print(f"distance: {D}")

closest elements: [[  0 899 521]]
distance: [[ 8.911423 12.088312 12.104569]]
```
#   Choosing Index 
提供大量索引和量化器，如何选择一个？
## 0. Preparation 

### Packages 

- CPU 
```python
In [ ]:

# %pip install -U faiss-cpu numpy h5py

For GPU on Linux x86_64 system, use Conda:  
对于 Linux x86_64 系统上的 GPU，使用 Conda：

`conda install -c pytorch -c nvidia faiss-gpu=1.8.0`

In [1]:

from urllib.request import urlretrieve
import h5py
import faiss
import numpy as np
```
### Dataset 
- 使用 SIFT1M，一个非常流行的用于 ANN 评估的数据集，作为我们的数据集来展示比较。
- 下载数据集
```python
In [ ]:

data_url = "http://ann-benchmarks.com/sift-128-euclidean.hdf5"
destination = "data.hdf5"
urlretrieve(data_url, destination)

Then load the data from the hdf5 file.  
然后从 hdf5 文件中加载数据。

In [2]:

with h5py.File('data.hdf5', 'r') as f:
    corpus = f['train'][:]
    query = f['test'][:]

print(corpus.shape, corpus.dtype)
print(query.shape, corpus.dtype)

(1000000, 128) float32
(10000, 128) float32

In [3]:

d = corpus[0].shape[0]
k = 100
```
### Helper function 
- 计算召回率的辅助函数。
```python
In [4]:
# compute recall from the prediction results and ground truth
def compute_recall(res, truth):
    recall = 0
    for i in range(len(res)):
        intersect = np.intersect1d(res[i], truth[i])
        recall += len(intersect) / len(res[i])
    recall /= len(res)

    return recall
```
## 1. Flat Index  
- 平面索引使用暴力搜索每个查询的邻居。它保证了 100%的召回率，从而我们使用它的结果作为基准事实。
```python
In [5]:

%%time
index = faiss.IndexFlatL2(d)
index.add(corpus)

CPU times: user 69.2 ms, sys: 80.6 ms, total: 150 ms
Wall time: 149 ms

In [6]:

%%time
D, I_truth = index.search(query, k)

CPU times: user 17min 30s, sys: 1.62 s, total: 17min 31s
Wall time: 2min 1s
```
## 2. IVF Index  
```python
In [7]:

%%time
nlist = 5
nprob = 3

quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFFlat(quantizer, d, nlist)
index.nprobe = nprob

index.train(corpus)
index.add(corpus)

CPU times: user 10.6 s, sys: 831 ms, total: 11.4 s
Wall time: 419 ms

In [8]:

%%time
D, I = index.search(query, k)

CPU times: user 9min 15s, sys: 598 ms, total: 9min 16s
Wall time: 12.5 s

In [9]:

recall = compute_recall(I, I_truth)
print(f"Recall: {recall}")

Recall: 0.9999189999999997
```
- IVFFlatL2 在搜索速度方面有很好的提升，同时召回率损失非常小。

## 3. HNSW Index  
```python
In [10]:

%%time
M = 64
ef_search = 32
ef_construction = 64

index = faiss.IndexHNSWFlat(d, M)
# set the two parameters before adding data
index.hnsw.efConstruction = ef_construction
index.hnsw.efSearch = ef_search

index.add(corpus)

CPU times: user 11min 21s, sys: 595 ms, total: 11min 22s
Wall time: 17 s

In [11]:

%%time
D, I = index.search(query, k)

CPU times: user 5.14 s, sys: 3.94 ms, total: 5.14 s
Wall time: 110 ms

In [12]:

recall = compute_recall(I, I_truth)
print(f"Recall: {recall}")

Recall: 0.8963409999999716
```
- 从搜索时间小于 1 秒来看，我们可以看出 HNSW 在搜索阶段追求极致速度时是最佳选择之一。召回率的降低是可以接受的。但索引创建时间较长和较大的内存占用需要考虑。

## 4. LSH
```python
In [13]:

%%time
nbits = d * 8

index = faiss.IndexLSH(d, nbits)
index.train(corpus)
index.add(corpus)

CPU times: user 13.7 s, sys: 660 ms, total: 14.4 s
Wall time: 12.1 s

In [14]:

%%time
D, I = index.search(query, k)

CPU times: user 3min 20s, sys: 84.2 ms, total: 3min 20s
Wall time: 5.64 s

In [15]:

recall = compute_recall(I, I_truth)
print(f"Recall: {recall}")

Recall: 0.5856720000000037
```
- 如我们在上一笔记本中所述，当数据维度较大时，LSH 不是一个好的选择。在这里，128 已经对 LSH 构成了负担。正如我们所见，即使我们选择相对较小的 d * 8 的 `nbits` ，索引创建时间和搜索时间仍然相当长。大约 58.6%的召回率并不令人满意。

## 5. Scalar Quantizer Index  
```python
In [16]:

%%time
qtype = faiss.ScalarQuantizer.QT_8bit
metric = faiss.METRIC_L2

index = faiss.IndexScalarQuantizer(d, qtype, metric)
index.train(corpus)
index.add(corpus)

CPU times: user 550 ms, sys: 18 ms, total: 568 ms
Wall time: 87.4 ms

In [17]:

%%time
D, I = index.search(query, k)

CPU times: user 7min 36s, sys: 169 ms, total: 7min 36s
Wall time: 12.7 s

In [18]:

recall = compute_recall(I, I_truth)
print(f"Recall: {recall}")

Recall: 0.990444999999872
```
- 这里标量量化索引的性能看起来与 Flat 索引非常相似。因为 SIFT 数据集中向量的元素是范围在[0, 218]内的整数。因此，在标量量化过程中，索引不会丢失太多信息。对于具有更复杂分布的 float32 数据集，差异将更加明显。

## 6. Product Quantizer Index  
```python
In [19]:

%%time
M = 16
nbits = 8
metric = faiss.METRIC_L2

index = faiss.IndexPQ(d, M, nbits, metric)

index.train(corpus)
index.add(corpus)

CPU times: user 46.7 s, sys: 22.3 ms, total: 46.7 s
Wall time: 1.36 s

In [20]:

%%time
D, I = index.search(query, k)

CPU times: user 1min 37s, sys: 106 ms, total: 1min 37s
Wall time: 2.8 s

In [21]:

recall = compute_recall(I, I_truth)
print(f"Recall: {recall}")

Recall: 0.630898999999999
```
- 产品量化索引在任何一方面都不突出。但它在一定程度上平衡了权衡。它与其他索引（如 IVF 或 HNSW）结合广泛应用于实际应用中。
