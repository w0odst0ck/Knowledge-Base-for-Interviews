以下是 Git 的常见命令分类汇总，以及对应的常见报错和解决方案：

### **一、基础操作命令**

#### **仓库操作**

|命令|作用|示例|
|---|---|---|
|`git init`|初始化本地仓库|`git init myproject`|
|`git clone <url>`|克隆远程仓库|`git clone https://github.com/xxx.git`|
|`git remote -v`|查看远程仓库信息||
|`git remote add <name> <url>`|添加远程仓库|`git remote add origin https://...`|

#### **提交操作**

|命令|作用|示例|
|---|---|---|
|`git status`|查看工作区状态||
|`git add <file>`|将文件添加到暂存区|`git add README.md` 或 `git add .`|
|`git commit -m "msg"`|提交暂存区到本地仓库|`git commit -m "fix bug"`|
|`git diff`|查看工作区与暂存区差异||
|`git diff --staged`|查看暂存区与本地仓库差异||

#### **分支操作**

|命令|作用|示例|
|---|---|---|
|`git branch`|查看本地分支||
|`git branch <name>`|创建新分支|`git branch feature`|
|`git checkout <name>`|切换分支|`git checkout feature`|
|`git checkout -b <name>`|创建并切换分支|`git checkout -b feature`|
|`git merge <name>`|合并指定分支到当前分支|`git merge feature`|
|`git branch -d <name>`|删除分支|`git branch -d feature`|

#### **远程同步**

|命令|作用|示例|
|---|---|---|
|`git pull <remote> <branch>`|拉取远程分支并合并|`git pull origin main`|
|`git push <remote> <branch>`|推送本地分支到远程|`git push origin main`|
|`git fetch <remote>`|仅获取远程分支更新|`git fetch origin`|

### **二、常见报错及解决方案**

#### **1. 认证失败（HTTPS）**

- **错误信息**：  
    `remote: Invalid username or password.`  
    `fatal: Authentication failed for 'https://github.com/...'`
- **原因**：GitHub 密码错误或未配置凭据助手
- **解决方案**：
    
    bash
    
    ```bash
    # 配置Git保存密码（避免每次输入）
    git config --global credential.helper store
    
    # 若已配置SSH密钥，切换为SSH协议
    git remote set-url origin git@github.com:user/repo.git
    ```
    
      
    

#### **2. 权限不足（SSH）**

- **错误信息**：  
    `Permission denied (publickey).`  
    `fatal: Could not read from remote repository.`
- **原因**：SSH 密钥未添加到 GitHub 账户或权限错误
- **解决方案**：
    
    bash
    
    ```bash
    # 检查SSH密钥配置
    ssh -T git@github.com
    
    # 重新生成并添加SSH密钥
    ssh-keygen -t ed25519 -C "your_email@example.com"
    ```
    
      
    

#### **3. 连接被拒绝（SSH）**

- **错误信息**：  
    `kex_exchange_identification: Connection closed by remote host`  
    `Connection closed by 20.205.243.166 port 22`
- **原因**：防火墙阻止、网络代理或 GitHub 服务问题
- **解决方案**：
    
    bash
    
    ```bash
    # 测试网络连通性
    ping github.com
    nc -zv github.com 22
    
    # 使用备用SSH端口443
    git config --global sshProxy "ssh -p 443"
    ```
    
      
    

#### **4. 冲突合并（Merge Conflict）**

- **错误信息**：  
    `CONFLICT (content): Merge conflict in file.txt`  
    `Automatic merge failed; fix conflicts and then commit the result.`
- **原因**：多人修改同一文件导致冲突
- **解决方案**：
    1. 手动编辑冲突文件，删除冲突标记（`<<<<<<<`、`=======`、`>>>>>>>`）
    2. 标记冲突已解决：
        
        bash
        
        ```bash
        git add <conflicted-file>
        git commit -m "resolve merge conflict"
        ```
        
          
        

#### **5. 推送被拒绝（非快进更新）**

- **错误信息**：  
    `To https://github.com/user/repo.git`  
    `! [rejected] main -> main (non-fast-forward)`  
    `error: failed to push some refs to 'https://github.com/user/repo.git'`
- **原因**：远程仓库有新提交，本地分支落后
- **解决方案**：
    
    bash
    
    ```bash
    # 先拉取远程更新并合并
    git pull --rebase origin main
    
    # 再推送本地分支
    git push origin main
    ```
    
      
    

#### **6. 证书验证失败（HTTPS）**

- **错误信息**：  
    `SSL certificate problem: unable to get local issuer certificate`  
    `fatal: unable to access 'https://github.com/...'`
- **原因**：Git 无法验证 HTTPS 证书
- **解决方案**：
    
    bash
    
    ```bash
    # 临时忽略证书验证（不推荐）
    git config --global http.sslVerify false
    
    # 正确配置证书路径
    git config --global http.sslCAInfo "/path/to/cacert.pem"
    ```
    
      
    

### **三、其他常用命令**

|命令|作用|
|---|---|
|`git log`|查看提交历史|
|`git reset <file>`|撤销暂存区的文件|
|`git revert <commit>`|撤销指定提交|
|`git stash`|暂存当前工作区修改|
|`git stash pop`|恢复暂存的修改|
|`git tag <name>`|创建标签（用于发布版本）|
|`git config --list`|查看 Git 配置|

### **四、Git 配置命令**

|命令|作用|
|---|---|
|`git config --global user.name "Your Name"`|设置用户名|
|`git config --global user.email "you@example.com"`|设置邮箱|
|`git config --global core.editor "code --wait"`|设置编辑器（VS Code）|
|`git config --global alias.st status`|设置命令别名|

### **五、学习资源推荐**

- 官方文档：[Git Documentation](https://git-scm.com/doc)
- 交互式教程：[Git Immersion](https://gitimmersion.com/)
- 可视化指南：[Git Visualizer](https://git-school.github.io/visualizing-git/)

  

掌握这些命令和错误处理方法，基本可以应对日常开发中的 Git 使用需求。遇到复杂场景时，建议查阅官方文档或社区教程获取更详细的解决方案。