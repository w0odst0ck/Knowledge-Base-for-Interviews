以下是 OpenAI Python SDK 的常用用法和示例，基于最新的搜索结果整理而成：

---

### **1. 安装 OpenAI 库**
使用 `pip` 安装 OpenAI Python SDK：
```bash
pip install openai
```
如果你正在从旧版本升级，可以参考 [OpenAI 官方文档](https://platform.openai.com/docs/libraries/python) 中的迁移指南。

---

### **2. 配置 API 密钥**
在使用 OpenAI API 之前，需要配置 API 密钥。可以通过以下方式设置：

#### **直接在代码中设置**
```python
import openai
openai.api_key = "YOUR_API_KEY"
```

#### **通过环境变量设置**
将 API 密钥保存到环境变量中：
- **macOS/Linux**：在终端中运行
  ```bash
  export OPENAI_API_KEY="YOUR_API_KEY"
  ```
- **Windows**：在 PowerShell 中运行
  ```bash
  setx OPENAI_API_KEY "YOUR_API_KEY"
  ```

---

### **3. 基本用法：文本生成**
以下是一个使用 GPT 模型生成文本的示例：
```python
from openai import OpenAI

client = OpenAI(api_key="YOUR_API_KEY")

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": "写一篇关于人工智能的文章。"}
    ],
    max_tokens=500  # 设置生成文本的最大长度
)

print(response.choices[0].message.content)
```

---

### **4. 高级用法：流式响应**
如果需要实时处理生成的文本，可以启用流式响应：
```python
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": "生成一个故事。"}
    ],
    stream=True
)

for chunk in response:
    print(chunk.choices[0].delta.content, end="", flush=True)
```

---

### **5. 错误处理与重试机制**
在实际开发中，可能会遇到 API 调用失败的情况。以下是一个简单的错误处理示例：
```python
import time

def generate_text(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error: {e}")
        time.sleep(10)  # 等待 10 秒后重试
        return generate_text(prompt)

print(generate_text("解释人工智能是什么？"))
```

---

### **6. 使用不同模型**
OpenAI 提供了多种模型，适用于不同的任务。以下是一些常见模型及其用途：

| 模型名称 | 特点 | 适用场景 |
| --- | --- | --- |
| `gpt-3.5-turbo` | 高性价比，擅长文本生成和对话 | 聊天机器人、内容生成 |
| `gpt-4` | 更强大的推理能力，上下文窗口更长 | 复杂任务、多轮对话 |
| `text-embedding-ada-002` | 将文本转换为向量 | 文本相似度比较、聚类 |

---

### **7. 函数调用（Function Calling）**
OpenAI 支持函数调用，允许模型决定何时调用外部函数。以下是一个示例：
```python
import json

def get_current_weather(location, unit="celsius"):
    weather_info = {
        "location": location,
        "temperature": "25",
        "unit": unit,
        "forecast": ["sunny", "windy"]
    }
    return json.dumps(weather_info)

functions = [
    {
        "name": "get_current_weather",
        "description": "获取指定位置的当前天气",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location"]
        }
    }
]

messages = [{"role": "user", "content": "今天北京的天气怎么样？"}]

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages,
    functions=functions,
    function_call="auto"
)

response_message = response.choices[0].message

if response_message.function_call:
    function_name = response_message.function_call.name
    function_args = json.loads(response_message.function_call.arguments)

    if function_name == "get_current_weather":
        function_response = get_current_weather(
            location=function_args.get("location"),
            unit=function_args.get("unit")
        )
        print(function_response)
```

---

### **8. 提示工程（Prompt Engineering）**
为了更好地与 AI 沟通，需要优化提示（Prompt）。以下是一些技巧：
- **明确任务目标**：确保提示清晰、具体。
- **提供上下文**：通过系统消息（`role: "system"`）设置助手的行为。
- **限制输出长度**：通过 `max_tokens` 参数控制生成的文本长度。

---

