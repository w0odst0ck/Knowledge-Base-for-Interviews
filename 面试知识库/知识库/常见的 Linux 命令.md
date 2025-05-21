## Linux
### 文件和目录操作
1. **ls** - 列出目录内容
   - `ls -l`：以长格式列出
   - `ls -a`：显示隐藏文件

2. **cd** - 切换目录
   - `cd /path/to/directory`：切换到指定目录
   - `cd ..`：返回上一级目录

3. **pwd** - 显示当前工作目录

4. **mkdir** - 创建目录
   - `mkdir dirname`：创建名为 `dirname` 的目录

5. **rmdir** - 删除空目录
   - `rmdir dirname`：删除名为 `dirname` 的空目录

6. **rm** - 删除文件或目录
   - `rm filename`：删除文件
   - `rm -r dirname`：递归删除目录及其内容

7. **cp** - 复制文件或目录
   - `cp file1 file2`：复制 `file1` 到 `file2`
   - `cp -r dir1 dir2`：递归复制目录

8. **mv** - 移动或重命名文件或目录
   - `mv file1 file2`：重命名 `file1` 为 `file2`
   - `mv file1 /path/to/directory`：移动 `file1` 到指定目录

9. **touch** - 创建空文件或更新文件时间戳
   - `touch filename`：创建空文件或更新文件时间戳

10. **cat** - 查看文件内容
    - `cat filename`：显示文件内容

11. **more** / **less** - 分页查看文件内容
    - `more filename`：分页显示文件内容
    - `less filename`：类似 `more`，但功能更强大

12. **head** / **tail** - 查看文件开头或结尾部分
    - `head -n 10 filename`：显示文件前 10 行
    - `tail -n 10 filename`：显示文件最后 10 行

### 文件权限和所有权
1. **chmod** - 修改文件权限
   - `chmod 755 filename`：设置文件权限为 `rwxr-xr-x`

2. **chown** - 修改文件所有者
   - `chown user:group filename`：更改文件所有者和所属组

3. **chgrp** - 修改文件所属组
   - `chgrp groupname filename`：更改文件所属组

### 系统信息
4. **uname** - 显示系统信息
   - `uname -a`：显示所有系统信息

5. **top** / **htop** - 显示系统进程和资源使用情况
   - `top`：实时显示系统进程
   - `htop`：增强版的 `top`

6. **ps** - 显示当前进程状态
   - `ps aux`：显示所有进程

7. **df** - 显示磁盘空间使用情况
   - `df -h`：以人类可读格式显示磁盘空间

8. **du** - 显示目录或文件的磁盘使用情况
   - `du -sh dirname`：显示目录的总大小

9. **free** - 显示内存使用情况
   - `free -h`：以人类可读格式显示内存使用情况

### 网络相关
10. **ping** - 测试网络连接
   - `ping example.com`：测试与 `example.com` 的连接

11. **ifconfig** / **ip** - 显示和配置网络接口
   - `ifconfig`：显示网络接口信息
   - `ip addr show`：显示网络接口信息（`ip` 命令）

12. **netstat** - 显示网络连接、路由表、接口统计信息等
   - `netstat -tuln`：显示所有监听端口

13. **ssh** - 远程登录
   - `ssh user@hostname`：通过 SSH 登录远程主机

14. **scp** - 安全复制文件
   - `scp file user@hostname:/path/to/destination`：复制文件到远程主机

15. **wget** / **curl** - 下载文件
   - `wget http://example.com/file`：下载文件
   - `curl -O http://example.com/file`：下载文件

### 包管理
16. **apt** (Debian/Ubuntu) - 包管理工具
   - `apt update`：更新包列表
   - `apt install package`：安装包
   - `apt remove package`：卸载包

17. **yum** (CentOS/RHEL) - 包管理工具
   - `yum install package`：安装包
   - `yum remove package`：卸载包

18. **dnf** (Fedora) - 包管理工具
   - `dnf install package`：安装包
   - `dnf remove package`：卸载包

### 压缩和解压缩
19. **tar** - 打包和解包文件
   - `tar -cvf archive.tar dirname`：打包目录
   - `tar -xvf archive.tar`：解包文件

20. **gzip** / **gunzip** - 压缩和解压缩文件
   - `gzip filename`：压缩文件
   - `gunzip filename.gz`：解压缩文件

21. **zip** / **unzip** - 压缩和解压缩文件
   - `zip archive.zip file1 file2`：压缩文件
   - `unzip archive.zip`：解压缩文件

### 其他常用命令
22. **find** - 查找文件
   - `find /path -name filename`：在指定路径下查找文件

23. **grep** - 文本搜索
   - `grep "pattern" filename`：在文件中搜索指定模式

24. **sed** - 流编辑器
   - `sed 's/old/new/g' filename`：替换文件中的文本

25. **awk** - 文本处理工具
   - `awk '{print $1}' filename`：打印文件的第一列

26. **man** - 查看命令手册
   - `man command`：查看命令的手册页

27. **history** - 显示命令历史
   - `history`：显示之前执行过的命令

28. **alias** - 创建命令别名
   - `alias ll='ls -la'`：创建 `ll` 作为 `ls -la` 的别名

29. **echo** - 显示文本
   - `echo "Hello, World!"`：显示文本

30. **date** - 显示或设置系统日期和时间
   - `date`：显示当前日期和时间

31. **shutdown** - 关机或重启系统
    - `shutdown now`：立即关机
    - `shutdown -r now`：立即重启

### python
### 1. **运行 Python 脚本**

如果你有一个 Python 脚本文件（例如 `script.py`），可以通过以下命令运行：



```bash
python script.py
```

或者，如果你的系统中安装了多个 Python 版本（如 Python 2 和 Python 3），可以通过指定版本号来运行：



```bash
python3 script.py
```

或者：



```bash
python2 script.py
```
安装特定环境的包
```shell
~\AppData\LocaPrograms\Python\Python39\python.exe -m pip install 包名
```
### 2. **进入 Python 交互式环境**

如果你想进入 Python 的交互式环境（REPL），可以直接运行以下命令：



```bash
python
```

或者指定版本：



```bash
python3
```

或者：



```bash
python2
```

### 3. **使用 Python 脚本作为可执行文件**

你可以将 Python 脚本设置为可执行文件，并通过 `shebang` 行指定解释器路径。例如，在脚本文件 `script.py` 的第一行添加：

Python复制

```python
#!/usr/bin/env python3
```

然后，将脚本文件设置为可执行：

bash复制

```bash
chmod +x script.py
```

之后，可以直接运行脚本：

bash复制

```bash
./script.py
```

### 4. **运行 Python 脚本并传递参数**

你可以通过命令行向 Python 脚本传递参数。例如：

bash复制

```bash
python script.py arg1 arg2
```

在脚本中，可以通过 `sys.argv` 来接收这些参数：

Python复制

```python
import sys
print(sys.argv)  # 输出 ['script.py', 'arg1', 'arg2']
```

### 5. **使用 Python 的 `-m` 选项**

如果你想运行 Python 模块或包，可以使用 `-m` 选项。例如：

bash复制

```bash
python -m http.server 8000
```

这会启动一个简单的 HTTP 服务器。

或者运行模块中的脚本：

bash复制

```bash
python -m mymodule
```

### 6. **运行 Python 脚本并输出到文件**

你可以将 Python 脚本的输出重定向到文件中。例如：

bash复制

```bash
python script.py > output.txt
```

或者同时输出到文件和终端：

bash复制

```bash
python script.py | tee output.txt
```

### 7. **使用虚拟环境运行 Python**

如果你使用了虚拟环境（如 `venv` 或 `conda`），需要先激活环境，然后运行 Python：

#### 使用 `venv`：

bash复制

```bash
source myenv/bin/activate
python script.py
```

#### 使用 `conda`：

bash复制

```bash
conda activate myenv
python script.py
```

### 8. **在后台运行 Python 脚本**

如果你想在后台运行 Python 脚本，可以使用 `nohup` 或 `&`：

bash复制

```bash
nohup python script.py &
```

或者：

bash复制

```bash
python script.py &
```

### 9. **使用 `cron` 定时运行 Python 脚本**

你可以将 Python 脚本添加到 `cron` 任务中，使其定时运行。编辑 `crontab` 文件：

bash复制

```bash
crontab -e
```

然后添加以下内容（每小时运行一次）：

bash复制

```bash
0 * * * * /usr/bin/python3 /path/to/script.py
```

---

这些是 Linux 下运行 Python 的常见方法。根据你的需求选择合适的方式即可。