
---

### **1. 初始化和配置**

- **初始化本地仓库**：
    
    bash复制
    
    ```bash
    git init
    ```
    
    （在当前目录创建一个新的 Git 仓库）
    
- **克隆远程仓库**：
    
    bash复制
    
    ```bash
    git clone <repository-url>
    ```
    
    （将远程仓库克隆到本地）
    
- **配置用户信息**：
    
    bash复制
    
    ```bash
    git config --global user.name "Your Name"
    git config --global user.email "your_email@example.com"
    ```
    
    （设置全局用户名和邮箱）
    

---

### **2. 状态和分支**

- **查看仓库状态**：
    
    bash复制
    
    ```bash
    git status
    ```
    
    （显示当前分支状态，包括未提交的更改）
    
- **查看分支**：
    
    bash复制
    
    ```bash
    git branch
    ```
    
    （列出所有本地分支）
    
- **创建新分支**：
    
    bash复制
    
    ```bash
    git branch <branch-name>
    ```
    
    （创建新分支但不切换）
    
- **切换分支**：
    
    bash复制
    
    ```bash
    git checkout <branch-name>
    ```
    
    （切换到指定分支）
    
- **创建并切换分支**：
    
    bash复制
    
    ```bash
    git checkout -b <branch-name>
    ```
    
    （创建新分支并切换到该分支）
    
- **删除分支**：
    
    bash复制
    
    ```bash
    git branch -d <branch-name>
    ```
    
    （删除本地分支，`-D` 强制删除）
    

---

### **3. 提交更改**

- **添加文件到暂存区**：
    
    bash复制
    
    ```bash
    git add <file-name>
    ```
    
    （添加单个文件）
    
    bash复制
    
    ```bash
    git add .
    ```
    
    （添加所有更改的文件）
    
- **提交更改到本地仓库**：
    
    bash复制
    
    ```bash
    git commit -m "Your commit message"
    ```
    
    （提交暂存区的更改，并添加描述）
    
- **查看提交历史**：
    
    bash复制
    
    ```bash
    git log
    ```
    
    （显示提交历史记录）
    
- **查看最后一次提交**：
    
    bash复制
    
    ```bash
    git log -1
    ```
    
- **查看文件的更改**：
    
    bash复制
    
    ```bash
    git diff <file-name>
    ```
    
    （显示未暂存的更改）
    
    bash复制
    
    ```bash
    git diff --staged
    ```
    
    （显示暂存区的更改）
    

---

### **4. 远程操作**

- **查看远程仓库信息**：
    
    bash复制
    
    ```bash
    git remote -v
    ```
    
- **添加远程仓库**：
    
    bash复制
    
    ```bash
    git remote add origin <repository-url>
    ```
    
- **推送更改到远程仓库**：
    
    bash复制
    
    ```bash
    git push origin <branch-name>
    ```
    
- **拉取远程仓库的更改**：
    
    bash复制
    
    ```bash
    git pull origin <branch-name>
    ```
    
- **更新远程分支信息**：
    
    bash复制
    
    ```bash
    git fetch
    ```
    

---

### **5. 合并和冲突**

- **合并分支**：
    
    bash复制
    
    ```bash
    git merge <branch-name>
    ```
    
    （将指定分支的更改合并到当前分支）
    
- **解决冲突**： 手动编辑冲突文件后，运行以下命令：
    
    bash复制
    
    ```bash
    git add <file-name>
    git commit
    ```
    

---

### **6. 标签**

- **创建标签**：
    
    bash复制
    
    ```bash
    git tag <tag-name>
    ```
    
- **推送标签到远程仓库**：
    
    bash复制
    
    ```bash
    git push origin <tag-name>
    ```
    
- **查看所有标签**：
    
    bash复制
    
    ```bash
    git tag
    ```
    

---

### **7. 撤销更改**

- **撤销暂存区的更改**：
    
    bash复制
    
    ```bash
    git reset <file-name>
    ```
    
- **撤销工作区的更改**：
    
    bash复制
    
    ```bash
    git checkout -- <file-name>
    ```
    
- **撤销最后一次提交**：
    
    bash复制
    
    ```bash
    git revert HEAD
    ```
    
    （创建一个新的提交来撤销最后一次提交的更改）
    
- **强制撤销最后一次提交（不保留更改）**：
    
    bash复制
    
    ```bash
    git reset --hard HEAD~1
    ```
    

---

### **8. 其他实用命令**

- **查看文件的提交历史**：
    
    bash复制
    
    ```bash
    git blame <file-name>
    ```
    
- **查看分支的差异**：
    
    bash复制
    
    ```bash
    git diff <branch1>..<branch2>
    ```
    
- **查看文件的版本历史**：
    
    bash复制
    
    ```bash
    git log -p <file-name>
    ```
    
- **查看仓库的大小**：
    
    bash复制
    
    ```bash
    git count-objects -vH
    ```
    
- **清理未跟踪的文件**：
    
    bash复制
    
    ```bash
    git clean -fd
    ```
    

---
