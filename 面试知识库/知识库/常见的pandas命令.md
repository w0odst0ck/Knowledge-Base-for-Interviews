`pandas` 是 Python 中用于数据分析和处理的强大库，广泛应用于数据清洗、转换、分析和可视化。以下是 `pandas` 库中一些常见的命令和操作，按功能分类介绍。

---

### **1. 导入和安装**

#### 安装 Pandas

bash复制

```bash
pip install pandas
```

#### 导入 Pandas

Python复制

```python
import pandas as pd
```

---

### **2. 创建数据结构**

#### 创建 DataFrame

Python复制

```python
data = {
    "Name": ["Alice", "Bob", "Charlie"],
    "Age": [25, 30, 35],
    "City": ["New York", "Los Angeles", "Chicago"]
}
df = pd.DataFrame(data)
print(df)
```

#### 从 CSV 文件读取数据

Python复制

```python
df = pd.read_csv("data.csv")
print(df)
```

#### 从 Excel 文件读取数据

```python
df = pd.read_excel("data.xlsx")
print(df)
```
#### 创建 Series

```python
s = pd.Series([1, 2, 3, 4], index=["a", "b", "c", "d"])
print(s)
```

---

### **3. 数据查看和基本信息**

#### 查看 DataFrame 的前几行

Python复制

```python
print(df.head())  # 默认显示前 5 行
print(df.head(3))  # 显示前 3 行
```

#### 查看 DataFrame 的后几行

Python复制

```python
print(df.tail())  # 默认显示后 5 行
print(df.tail(2))  # 显示后 2 行
```

#### 查看数据的形状

Python复制

```python
print(df.shape)  # 输出 (行数, 列数)
```

#### 查看数据的列名

Python复制

```python
print(df.columns)
```

#### 查看数据的索引

Python复制

```python
print(df.index)
```

#### 查看数据的统计信息

Python复制

```python
print(df.describe())  # 描述性统计（均值、标准差、最小值等）
```

#### 查看数据类型

Python复制

```python
print(df.dtypes)
```

---

### **4. 数据选择和筛选**

#### 选择单列

Python复制

```python
print(df["Name"])  # 或 df.Name
```

#### 选择多列

Python复制

```python
print(df[["Name", "Age"]])
```

#### 选择行（基于索引）

Python复制

```python
print(df.iloc[0])  # 第一行
print(df.iloc[1:3])  # 第 2 行到第 3 行
```

#### 选择行（基于条件）

Python复制

```python
print(df[df["Age"] > 30])  # 筛选年龄大于 30 的行
```

#### 使用 `query` 方法筛选

Python复制

```python
print(df.query("Age > 30"))
```

---

### **5. 数据清洗和处理**

#### 删除重复行

Python复制

```python
df = df.drop_duplicates()
print(df)
```

#### 删除列

Python复制

```python
df = df.drop(columns=["City"])
print(df)
```

#### 重命名列

Python复制

```python
df = df.rename(columns={"Name": "Full Name"})
print(df)
```

#### 填充缺失值

Python复制

```python
df = df.fillna(value={"Age": 0, "City": "Unknown"})
print(df)
```

#### 删除缺失值

Python复制

```python
df = df.dropna()
print(df)
```

#### 数据类型转换

Python复制

```python
df["Age"] = df["Age"].astype(int)
print(df.dtypes)
```

---

### **6. 数据排序和分组**

#### 按列排序

Python复制

```python
df = df.sort_values(by="Age", ascending=False)
print(df)
```

#### 按多列排序

Python复制

```python
df = df.sort_values(by=["Age", "Name"], ascending=[False, True])
print(df)
```

#### 分组聚合

Python复制

```python
grouped = df.groupby("City").mean()
print(grouped)
```

#### 分组后应用函数

Python复制

```python
grouped = df.groupby("City").agg({"Age": ["mean", "max"], "Name": "count"})
print(grouped)
```

---

### **7. 数据合并和连接**

#### 合并 DataFrame（按列合并）

Python复制

```python
df1 = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
df2 = pd.DataFrame({"A": [5, 6], "B": [7, 8]})
result = pd.concat([df1, df2], axis=0)
print(result)
```

#### 合并 DataFrame（按行合并）

Python复制

```python
result = pd.concat([df1, df2], axis=1)
print(result)
```

#### 使用 `merge` 合并 DataFrame

Python复制

```python
df1 = pd.DataFrame({"Key": ["A", "B"], "Value": [1, 2]})
df2 = pd.DataFrame({"Key": ["A", "C"], "Value": [3, 4]})
result = pd.merge(df1, df2, on="Key", how="inner")  # inner, outer, left, right
print(result)
```

---

### **8. 数据导出**

#### 导出为 CSV 文件

Python复制

```python
df.to_csv("output.csv", index=False)
```

#### 导出为 Excel 文件

Python复制

```python
df.to_excel("output.xlsx", index=False)
```

---

### **9. 数据可视化（结合 Matplotlib）**

#### 绘制柱状图

Python复制

```python
import matplotlib.pyplot as plt

df.plot(kind="bar", x="Name", y="Age")
plt.show()
```

#### 绘制折线图

Python复制

```python
df.plot(kind="line", x="Name", y="Age")
plt.show()
```

---

### **10. 常见操作示例**

#### 示例：处理缺失值并导出

Python复制

```python
# 填充缺失值
df = df.fillna({"Age": df["Age"].mean(), "City": "Unknown"})

# 导出处理后的数据
df.to_csv("cleaned_data.csv", index=False)
```

#### 示例：分组聚合并可视化

Python复制

```python
# 分组聚合
grouped = df.groupby("City").mean()

# 可视化
grouped.plot(kind="bar")
plt.show()
```

---

### **11. 注意事项**

1. **索引问题**：在合并或筛选数据后，索引可能会被打乱，可以使用 `df.reset_index(drop=True)` 重置索引。
    
2. **性能优化**：对于大数据集，尽量避免循环操作，使用向量化操作或分组聚合。
    
3. **数据类型**：确保数据类型正确，否则可能影响计算结果。
    

通过以上命令和操作，你可以高效地使用 `pandas` 进行数据处理和分析。