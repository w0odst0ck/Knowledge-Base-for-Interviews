在 Python 中，`json` 模块是处理 JSON 数据的标准库，而 JSONL（JSON Lines）文件是一种特殊的 JSON 格式，每行包含一个独立的 JSON 对象。以下是使用 Python `json` 模块处理 JSON 和 JSONL 文件的常见命令和示例。

---

### **1. JSON 文件处理**

#### **1.1 读取 JSON 文件**

从文件中加载 JSON 数据并解析为 Python 对象（通常是字典或列表）。

Python复制

```python
import json

# 打开 JSON 文件并加载数据
with open("data.json", "r", encoding="utf-8") as file:
    data = json.load(file)

print(data)  # 输出解析后的数据
```

#### **1.2 写入 JSON 文件**

将 Python 对象（字典或列表）序列化为 JSON 格式并写入文件。

Python复制

```python
import json

data = {
    "name": "Alice",
    "age": 25,
    "city": "New York"
}

# 将数据写入 JSON 文件
with open("data.json", "w", encoding="utf-8") as file:
    json.dump(data, file, ensure_ascii=False, indent=4)
```

- `ensure_ascii=False`：允许非 ASCII 字符（如中文）直接写入文件。
    
- `indent=4`：格式化输出，使 JSON 文件更易读。
    

#### **1.3 解析 JSON 字符串**

将 JSON 格式的字符串解析为 Python 对象。

Python复制

```python
import json

json_string = '{"name": "Alice", "age": 25}'
data = json.loads(json_string)
print(data)  # 输出：{'name': 'Alice', 'age': 25}
```

#### **1.4 序列化为 JSON 字符串**

将 Python 对象转换为 JSON 格式的字符串。

Python复制

```python
import json

data = {"name": "Alice", "age": 25}
json_string = json.dumps(data, ensure_ascii=False)
print(json_string)  # 输出：{"name": "Alice", "age": 25}
```

---

### **2. JSONL 文件处理**

JSONL 文件是一种每行包含一个独立 JSON 对象的格式，通常用于处理大规模数据集。

#### **2.1 读取 JSONL 文件**

逐行读取 JSONL 文件，并解析每一行为一个 Python 对象。

Python复制

```python
import json

# 读取 JSONL 文件
with open("data.jsonl", "r", encoding="utf-8") as file:
    for line in file:
        data = json.loads(line)
        print(data)  # 输出每行解析后的数据
```

#### **2.2 写入 JSONL 文件**

将 Python 对象逐行写入 JSONL 文件。

Python复制

```python
import json

data_list = [
    {"name": "Alice", "age": 25},
    {"name": "Bob", "age": 30},
    {"name": "Charlie", "age": 35}
]

# 写入 JSONL 文件
with open("data.jsonl", "w", encoding="utf-8") as file:
    for data in data_list:
        json.dump(data, file, ensure_ascii=False)
        file.write("\n")  # 每个 JSON 对象占一行
```

---

### **3. 常见操作示例**

#### **3.1 处理嵌套 JSON 数据**

JSON 数据可以是嵌套的，例如包含列表或字典。

Python复制

```python
import json

# 嵌套的 JSON 数据
json_string = '''
{
    "name": "Alice",
    "age": 25,
    "address": {
        "city": "New York",
        "zip": "10001"
    },
    "hobbies": ["reading", "coding", "traveling"]
}
'''

data = json.loads(json_string)
print(data["address"]["city"])  # 输出：New York
print(data["hobbies"][1])       # 输出：coding
```

#### **3.2 更新 JSON 数据并写回文件**

读取 JSON 文件，修改数据后写回文件。

Python复制

```python
import json

# 读取 JSON 文件
with open("data.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# 修改数据
data["age"] = 26

# 写回文件
with open("data.json", "w", encoding="utf-8") as file:
    json.dump(data, file, ensure_ascii=False, indent=4)
```

#### **3.3 从 JSONL 文件中提取特定字段**

逐行读取 JSONL 文件，提取特定字段。

Python复制

```python
import json

# 提取特定字段
with open("data.jsonl", "r", encoding="utf-8") as file:
    names = [json.loads(line)["name"] for line in file]

print(names)  # 输出：['Alice', 'Bob', 'Charlie']
```

---

### **4. 注意事项**

1. **编码问题**：在读写文件时，建议显式指定文件编码（如 `utf-8`），以避免编码错误。
    
2. **JSON 格式错误**：如果 JSON 数据格式不正确，`json.loads()` 或 `json.load()` 会抛出 `json.JSONDecodeError`。
    
3. **JSONL 文件的大小**：JSONL 文件通常用于大规模数据集，因此在处理时可能需要考虑内存和性能优化。
    
