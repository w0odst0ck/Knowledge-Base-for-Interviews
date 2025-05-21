## **1.1 Python 基础**

### **1.1.1 Python 核心语法**

Python 的核心语法包括变量、数据类型、运算符、控制流（如条件语句和循环）等。以下是一些关键点：

- **变量与数据类型**：Python 是动态类型语言，变量不需要显式声明类型。常见的数据类型包括整数（int）、浮点数（float）、字符串（str）、布尔值（bool）等。
  
  ```python
  # 变量与数据类型示例
  x = 10          # 整数
  y = 3.14        # 浮点数
  name = "Alice"  # 字符串
  is_active = True  # 布尔值
  ```

- **运算符**：Python 支持算术运算符（+、-、*、/）、比较运算符（==、!=、>、<）、逻辑运算符（and、or、not）等。

  ```python
  # 运算符示例
  a = 5
  b = 3
  print(a + b)  # 输出 8
  print(a > b)  # 输出 True
  print(a == b and b > 0)  # 输出 False
  ```

- **控制流**：Python 使用 `if`、`elif`、`else` 进行条件判断，使用 `for` 和 `while` 进行循环。

  ```python
  # 控制流示例
  age = 18
  if age >= 18:
      print("You are an adult.")
  else:
      print("You are a minor.")

  # for 循环示例
  for i in range(5):
      print(i)  # 输出 0 到 4

  # while 循环示例
  count = 0
  while count < 3:
      print("Count:", count)
      count += 1
  ```

### **1.1.2 数据结构（列表、字典、集合、元组）**

Python 提供了多种内置的数据结构，常用的有列表、字典、集合和元组。

- **列表（List）**：有序的可变序列，可以包含不同类型的元素。

  ```python
  # 列表示例
  fruits = ["apple", "banana", "cherry"]
  fruits.append("orange")  # 添加元素
  print(fruits[1])  # 输出 "banana"
  ```

- **字典（Dictionary）**：无序的键值对集合，键必须是唯一的。

  ```python
  # 字典示例
  person = {"name": "Alice", "age": 25}
  print(person["name"])  # 输出 "Alice"
  person["age"] = 26  # 修改值
  ```

- **集合（Set）**：无序且不重复的元素集合。

  ```python
  # 集合示例
  unique_numbers = {1, 2, 3, 3, 4}
  print(unique_numbers)  # 输出 {1, 2, 3, 4}
  ```

- **元组（Tuple）**：有序的不可变序列。

  ```python
  # 元组示例
  coordinates = (10.0, 20.0)
  print(coordinates[0])  # 输出 10.0
  ```

### **1.1.3 函数与类（OOP 编程）**

- **函数**：Python 使用 `def` 关键字定义函数，函数可以接受参数并返回值。

  ```python
  # 函数示例
  def greet(name):
      return f"Hello, {name}!"

  print(greet("Alice"))  # 输出 "Hello, Alice!"
  ```

- **类与对象**：Python 支持面向对象编程（OOP），使用 `class` 关键字定义类，类可以包含属性和方法。

  ```python
  # 类与对象示例
  class Dog:
      def __init__(self, name):
          self.name = name

      def bark(self):
          return f"{self.name} says woof!"

  my_dog = Dog("Buddy")
  print(my_dog.bark())  # 输出 "Buddy says woof!"
  ```

### **1.1.4 异常处理与调试**

- **异常处理**：Python 使用 `try`、`except`、`finally` 来处理异常，防止程序因错误而崩溃。

  ```python
  # 异常处理示例
  try:
      result = 10 / 0
  except ZeroDivisionError:
      print("Cannot divide by zero!")
  finally:
      print("Execution complete.")
  ```

- **调试**：Python 提供了 `pdb` 模块用于调试，也可以使用 `print` 语句或 IDE 的调试工具。

  ```python
  # 调试示例
  import pdb

  def divide(a, b):
      pdb.set_trace()  # 设置断点
      return a / b

  print(divide(10, 0))
  ```

### **1.1.5 文件操作与模块管理**

- **文件操作**：Python 使用 `open` 函数进行文件读写操作，文件操作完成后应使用 `close` 方法关闭文件，或使用 `with` 语句自动管理文件。

  ```python
  # 文件操作示例
  with open("example.txt", "w") as file:
      file.write("Hello, World!")

  with open("example.txt", "r") as file:
      content = file.read()
      print(content)  # 输出 "Hello, World!"
  ```

- **模块管理**：Python 使用 `import` 语句导入模块，模块可以是内置模块、第三方模块或自定义模块。

  ```python
  # 模块管理示例
  import math
  print(math.sqrt(16))  # 输出 4.0

  from datetime import datetime
  print(datetime.now())  # 输出当前时间
  ```
## **1.2 数据科学工具包**

数据科学工具包是 Python 生态系统中用于数据处理、分析和建模的核心库。以下是 NumPy、Pandas、Matplotlib、Seaborn 和 Scikit-learn 的详细说明及实例。

---

### **1.2.1 NumPy 基础与高级操作**

NumPy 是 Python 中用于科学计算的基础库，提供了高效的多维数组对象和数学函数。

- **基础操作**：
  - 创建数组、数组运算、索引和切片。

  ```python
  import numpy as np

  # 创建数组
  arr = np.array([1, 2, 3, 4, 5])
  print(arr)  # 输出 [1 2 3 4 5]

  # 数组运算
  print(arr + 2)  # 输出 [3 4 5 6 7]
  print(arr * 2)  # 输出 [2 4 6 8 10]

  # 索引和切片
  print(arr[1])  # 输出 2
  print(arr[1:4])  # 输出 [2 3 4]
  ```

- **高级操作**：
  - 数组形状操作、广播、矩阵运算。

  ```python
  # 数组形状操作
  arr = np.array([[1, 2, 3], [4, 5, 6]])
  print(arr.shape)  # 输出 (2, 3)

  # 广播
  arr1 = np.array([1, 2, 3])
  arr2 = np.array([[1], [2], [3]])
  print(arr1 + arr2)  # 输出 [[2 3 4], [3 4 5], [4 5 6]]

  # 矩阵运算
  mat1 = np.array([[1, 2], [3, 4]])
  mat2 = np.array([[5, 6], [7, 8]])
  print(np.dot(mat1, mat2))  # 输出 [[19 22], [43 50]]
  ```

---

### **1.2.2 Pandas 数据清洗与处理**

Pandas 是用于数据操作和分析的强大库，提供了 DataFrame 和 Series 数据结构。

- **数据读取与查看**：
  - 从 CSV、Excel 等文件读取数据，查看数据基本信息。

  ```python
  import pandas as pd

  # 读取 CSV 文件
  df = pd.read_csv("data.csv")
  print(df.head())  # 查看前 5 行数据
  print(df.info())  # 查看数据基本信息
  ```

- **数据清洗**：
  - 处理缺失值、重复值、数据类型转换。

  ```python
  # 处理缺失值
  df.fillna(0, inplace=True)  # 用 0 填充缺失值
  df.dropna(inplace=True)  # 删除包含缺失值的行

  # 处理重复值
  df.drop_duplicates(inplace=True)

  # 数据类型转换
  df["age"] = df["age"].astype(int)
  ```

- **数据处理**：
  - 数据筛选、排序、分组、聚合。

  ```python
  # 数据筛选
  filtered_df = df[df["age"] > 30]

  # 数据排序
  sorted_df = df.sort_values(by="age", ascending=False)

  # 数据分组与聚合
  grouped_df = df.groupby("gender")["age"].mean()
  print(grouped_df)
  ```

---

### **1.2.3 Matplotlib 与 Seaborn 数据可视化**

Matplotlib 是 Python 中最常用的绘图库，Seaborn 是基于 Matplotlib 的高级可视化库，提供了更美观的图表和更简单的接口。

- **Matplotlib 基础**：
  - 绘制折线图、柱状图、散点图。

  ```python
  import matplotlib.pyplot as plt

  # 折线图
  x = [1, 2, 3, 4, 5]
  y = [10, 20, 25, 30, 40]
  plt.plot(x, y)
  plt.title("Line Chart")
  plt.show()

  # 柱状图
  plt.bar(x, y)
  plt.title("Bar Chart")
  plt.show()

  # 散点图
  plt.scatter(x, y)
  plt.title("Scatter Plot")
  plt.show()
  ```

- **Seaborn 高级可视化**：
  - 绘制热力图、箱线图、分布图。

  ```python
  import seaborn as sns

  # 热力图
  data = np.random.rand(10, 10)
  sns.heatmap(data, annot=True)
  plt.title("Heatmap")
  plt.show()

  # 箱线图
  sns.boxplot(x="gender", y="age", data=df)
  plt.title("Box Plot")
  plt.show()

  # 分布图
  sns.histplot(df["age"], kde=True)
  plt.title("Distribution Plot")
  plt.show()
  ```

---

### **1.2.4 Scikit-learn 机器学习基础**

Scikit-learn 是 Python 中用于机器学习的核心库，提供了丰富的算法和工具。

- **数据预处理**：
  - 数据标准化、分割训练集和测试集。

  ```python
  from sklearn.model_selection import train_test_split
  from sklearn.preprocessing import StandardScaler

  # 数据分割
  X = df.drop("target", axis=1)
  y = df["target"]
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # 数据标准化
  scaler = StandardScaler()
  X_train = scaler.fit_transform(X_train)
  X_test = scaler.transform(X_test)
  ```

- **模型训练与评估**：
  - 使用分类、回归算法进行模型训练和评估。

  ```python
  from sklearn.linear_model import LogisticRegression
  from sklearn.metrics import accuracy_score

  # 模型训练
  model = LogisticRegression()
  model.fit(X_train, y_train)

  # 模型预测
  y_pred = model.predict(X_test)

  # 模型评估
  accuracy = accuracy_score(y_test, y_pred)
  print("Accuracy:", accuracy)
  ```

- **模型选择与调优**：
  - 使用交叉验证和网格搜索优化模型。

  ```python
  from sklearn.model_selection import GridSearchCV

  # 网格搜索
  param_grid = {"C": [0.1, 1, 10], "penalty": ["l1", "l2"]}
  grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5)
  grid_search.fit(X_train, y_train)

  # 最优参数
  print("Best Parameters:", grid_search.best_params_)
  ```

---

