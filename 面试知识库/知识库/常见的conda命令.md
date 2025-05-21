## conda
### 1. **安装和更新**

- **安装 Conda**：通常通过安装 [Anaconda](https://www.anaconda.com/products/individual) 或 [Miniconda](https://docs.conda.io/en/latest/miniconda.html) 来获取。
    
- **更新 Conda**：
    
    
    
    ```bash
    conda update conda
    ```
    
- **更新所有包**：
    
    
    
    ```bash
    conda update --all
    ```
    

### 2. **环境管理**

- **创建新环境**：
    
    
    
    ```bash
    conda create -n myenv python=3.9
    ```
    
- **激活环境**：
    
    
    
    ```bash
    conda activate myenv
    ```
    
- **退出当前环境**：
    
    
    
    ```bash
    conda deactivate
    ```
    
- **列出所有环境**：
    
    
    
    ```bash
    conda env list
    ```
    
- **删除环境**：
    
    
    
    ```bash
    conda env remove -n myenv
    ```
    

### 3. **包管理**

- **安装包**：
    
    
    
    ```bash
    conda install numpy pandas
    ```
    
- **安装指定版本的包**：
    
    
    
    ```bash
    conda install numpy=1.20
    ```
    
- **更新包**：
    
    
    
    ```bash
    conda update numpy
    ```
    
- **卸载包**：
    
    
    
    ```bash
    conda uninstall numpy
    ```
    
- **列出当前环境中的包**：
    
    
    
    ```bash
    conda list
    ```
    

### 4. **搜索和查看**

- **搜索包**：
    
    
    
    ```bash
    conda search numpy
    ```
    
- **查看包信息**：
    
    
    
    ```bash
    conda info numpy
    ```
    

### 5. **配置**

- **查看配置信息**：
    
    
    
    ```bash
    conda config --show
    ```
    
- **添加通道（Channel）**：
    
    
    
    ```bash
    conda config --add channels conda-forge
    ```
    
    （`conda-forge` 是一个常用的第三方通道）
    
- **清除缓存**：
    
    
    
    ```bash
    conda clean --all
    ```
    

### 6. **导出和导入环境**

- **导出当前环境**：
    
    
    
    ```bash
    conda env export > environment.yml
    ```
    
- **从文件创建环境**：
    
    
    
    ```bash
    conda env create -f environment.yml
    ```
    

### 7. **其他实用命令**

- **检查 Conda 状态**：
    
    
    
    ```bash
    conda info
    ```
    
- **查看当前激活的环境**：
    
    
    
    ```bash
    conda info --envs
    ```
    
