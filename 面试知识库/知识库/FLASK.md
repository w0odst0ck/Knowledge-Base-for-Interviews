Flask 是一个轻量级的 Python Web 框架，适用于开发简单的 Web 应用或复杂的后端服务。一个典型的 Flask 应用通常包含以下结构和代码模块：

### **1. 基本代码架构**
以下是一个常见的 Flask 应用的基本代码架构：

#### **项目结构**
```
my_flask_app/
│
├── app/                     # Flask 应用的核心代码
│   ├── __init__.py          # 应用初始化
│   ├── routes.py            # 路由和视图函数
│   ├── templates/           # HTML 模板文件
│   └── static/              # 静态文件（CSS、JS、图片等）
│
├── config.py                # 配置文件
├── run.py                   # 启动脚本
└── requirements.txt         # 依赖文件
```

---

### **2. 代码模块详解**

#### **2.1 `__init__.py`：应用初始化**
```python
# app/__init__.py
from flask import Flask

def create_app():
    app = Flask(__name__)
    app.config.from_object('config.DevelopmentConfig')  # 加载配置

    # 注册蓝图
    from .routes import main_bp
    app.register_blueprint(main_bp)

    return app
```

#### **2.2 `routes.py`：路由和视图函数**
```python
# app/routes.py
from flask import Blueprint, render_template, request, jsonify

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    return render_template('index.html')

@main_bp.route('/api/data', methods=['GET'])
def get_data():
    data = {"message": "Hello, Flask!"}
    return jsonify(data)

@main_bp.route('/submit', methods=['POST'])
def submit():
    data = request.json
    print(data)
    return jsonify({"status": "success"})
```

#### **2.3 `config.py`：配置文件**
```python
# config.py
class Config:
    SECRET_KEY = 'your_secret_key'

class DevelopmentConfig(Config):
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///dev.db'

class ProductionConfig(Config):
    DEBUG = False
    SQLALCHEMY_DATABASE_URI = 'mysql+pymysql://user:password@host/dbname'
```

#### **2.4 `run.py`：启动脚本**
```python
# run.py
from app import create_app

app = create_app()

if __name__ == '__main__':
    app.run()
```

#### **2.5 `templates/index.html`：HTML 模板**
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Flask App</title>
</head>
<body>
    <h1>Welcome to Flask</h1>
</body>
</html>
```

#### **2.6 `requirements.txt`：依赖文件**
```plaintext
Flask==2.2.2
Flask-SQLAlchemy==3.0.2
```

---

### **3. 高级架构（适用于复杂应用）**

对于更复杂的 Flask 应用，可以进一步拆分代码，引入更多的模块化设计：

#### **项目结构**
```
my_flask_app/
│
├── app/                     # 核心代码
│   ├── __init__.py          # 初始化
│   ├── routes/              # 路由模块
│   │   ├── __init__.py
│   │   ├── main.py          # 主路由
│   │   └── api.py           # API 路由
│   ├── models.py            # 数据模型
│   ├── templates/           # HTML 模板
│   └── static/              # 静态文件
│
├── config.py                # 配置
├── run.py                   # 启动脚本
├── requirements.txt         # 依赖
└── tests/                   # 测试代码
```

#### **3.1 拆分路由**
```python
# app/routes/main.py
from flask import Blueprint, render_template

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    return render_template('index.html')

# app/routes/api.py
from flask import Blueprint, jsonify

api_bp = Blueprint('api', __name__)

@api_bp.route('/data', methods=['GET'])
def get_data():
    return jsonify({"message": "Hello, API!"})
```

#### **3.2 在 `__init__.py` 中注册多个蓝图**
```python
# app/__init__.py
from flask import Flask
from .routes.main import main_bp
from .routes.api import api_bp

def create_app():
    app = Flask(__name__)
    app.config.from_object('config.DevelopmentConfig')

    app.register_blueprint(main_bp)
    app.register_blueprint(api_bp, url_prefix='/api')

    return app
```

#### **3.3 数据模型**
```python
# app/models.py
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
```

#### **3.4 初始化数据库**
```python
# app/__init__.py
from .models import db

def create_app():
    app = Flask(__name__)
    app.config.from_object('config.DevelopmentConfig')

    db.init_app(app)

    with app.app_context():
        db.create_all()

    return app
```

---

### **4. 运行和测试**

#### **运行应用**
```bash
python run.py
```

#### **测试**
可以使用 `pytest` 或 `unittest` 来编写测试代码。

---

### **5. 部署**

#### **使用 WSGI 服务器（如 Gunicorn）**
```bash
pip install gunicorn
gunicorn -w 4 "app:create_app()"
```

#### **部署到云平台**
- **Heroku**：使用 `Procfile` 和 `requirements.txt`。
- **Docker**：创建 `Dockerfile` 并部署到容器化环境。

---
