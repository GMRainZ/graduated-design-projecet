在 Linux 中，将目录递归压缩为 `.tar.gz` 格式（同时保留文件结构），再解压还原的完整操作如下：

---

**一、压缩目录为 `.tar.gz`**

1. **基本命令**
```bash
tar -czvf 压缩包名称.tar.gz 要压缩的目录路径/
```
• 参数解释：

  • `-c` : 创建新压缩包。

  • `-z` : 使用 `gzip` 压缩。

  • `-v` : 显示压缩过程（可选，可删除此参数静默操作）。

  • `-f` : 指定压缩包文件名。


2. **操作示例**
```bash
# 压缩当前目录下的 my_folder 文件夹
tar -czvf my_folder.tar.gz my_folder/

# 压缩绝对路径的目录（如 /home/user/data）
tar -czvf backup.tar.gz /home/user/data/
```

3. **高级用法**
• 排除特定文件/目录：

  ```bash
  tar -czvf archive.tar.gz my_folder/ --exclude="*.log" --exclude="temp/"
  ```
• 压缩时保留权限（适合系统文件备份）：

  ```bash
  tar -czpf backup.tar.gz /etc/  # `-p` 保留文件权限
  ```

---

**二、解压 `.tar.gz` 文件**

1. **解压到当前目录**
```bash
tar -xzvf 压缩包名称.tar.gz
```
• 参数解释：

  • `-x` : 解压。

  • `-z` : 处理 `gzip` 压缩。

  • `-v` : 显示解压过程（可选）。

  • `-f` : 指定压缩包文件。


2. **解压到指定目录**
```bash
tar -xzvf 压缩包名称.tar.gz -C 目标目录路径/
```
• 示例：

  ```bash
  tar -xzvf my_folder.tar.gz -C /opt/  # 解压到 /opt 目录
  ```

---

**三、常见问题解决**

1. **权限不足**
• 如果目录属于其他用户或系统文件，压缩/解压时需加 `sudo`：

  ```bash
  sudo tar -czvf backup.tar.gz /var/log/
  ```

2. **压缩包过大**
• 使用更高压缩率（但速度更慢）：

  ```bash
  tar -cvf - my_folder/ | gzip -9 > archive.tar.gz  # `-9` 最高压缩率
  ```

3. **查看压缩包内容（不解压）**
```bash
tar -tzvf archive.tar.gz
```

---

**四、操作示意图**

```
# 压缩流程
原始目录 my_folder/ → tar -czvf → my_folder.tar.gz

# 解压流程
my_folder.tar.gz → tar -xzvf → 还原为 my_folder/
```

---

**五、总结**
• 压缩：`tar -czvf 包名.tar.gz 目录路径/`

• 解压：`tar -xzvf 包名.tar.gz [-C 目标路径]`

• 排除文件：`--exclude="pattern"`

• 查看内容：`tar -tzvf 包名.tar.gz`


使用此方法可高效处理目录压缩备份及还原，适用于日志归档、数据迁移等场景。