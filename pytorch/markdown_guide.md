# Markdown 使用指南

这份文档用来快速掌握 Markdown 的常用写法。Markdown 的目标是：用普通文本写出结构清晰的文档，再由编辑器或网页渲染成标题、列表、代码块、表格等格式。

在 VS Code 中，常用操作如下：

| 操作 | Windows 快捷键 | 说明 |
| --- | --- | --- |
| 打开 Markdown 预览 | `Ctrl + Shift + V` | 把 `.md` 文件渲染成更容易阅读的样子 |
| 打开侧边预览 | `Ctrl + K` 后按 `V` | 左边编辑，右边预览 |
| 保存文件 | `Ctrl + S` | 保存当前 Markdown 文件 |
| 命令面板 | `Ctrl + Shift + P` | 可以搜索 Markdown 相关命令 |

注意：`Ctrl + Shift + V` 在 VS Code 编辑器里通常是打开 Markdown 预览；在 PowerShell 或终端里通常是粘贴。

## 1. 标题

标题用 `#` 表示。`#` 越多，标题级别越低。

```markdown
# 一级标题
## 二级标题
### 三级标题
#### 四级标题
```

渲染效果大概是：

# 一级标题示例

## 二级标题示例

### 三级标题示例

建议一个文档只有一个一级标题，也就是只写一个 `#`。

## 2. 段落和换行

普通文字直接写就是段落。段落之间空一行。

```markdown
这是第一段。

这是第二段。
```

如果只是按一次回车，很多 Markdown 渲染器会把它当成同一个段落。想明确换行，可以：

```markdown
第一行后面加两个空格  
第二行
```

更推荐多数时候直接空一行，文档更清楚。

## 3. 加粗、斜体、删除线

```markdown
**加粗**
*斜体*
~~删除线~~
```

效果：

**加粗**

*斜体*

~~删除线~~

## 4. 行内代码

行内代码用一对反引号包起来，适合写命令、文件名、变量名。

```markdown
使用 `pip install -r requirements.txt` 安装依赖。
配置文件是 `config/config.yaml`。
```

效果：

使用 `pip install -r requirements.txt` 安装依赖。配置文件是 `config/config.yaml`。

## 5. 代码块

多行代码用三个反引号包起来。最好在开头反引号后写语言名，例如 `python`、`powershell`、`bash`、`yaml`。

````markdown
```python
import torch

print(torch.__version__)
```
````

效果：

```python
import torch

print(torch.__version__)
```

PowerShell 命令可以这样写：

````markdown
```powershell
conda activate rl310
python --version
```
````

效果：

```powershell
conda activate rl310
python --version
```

写代码块时，代码块前后最好各空一行，这样可以避免 markdownlint 的 `MD031` 警告。

## 6. 无序列表

无序列表用 `-`、`*` 或 `+`。建议统一用 `-`。

```markdown
- Python 3.10
- PyTorch
- Gymnasium
- SwanLab
```

效果：

- Python 3.10
- PyTorch
- Gymnasium
- SwanLab

## 7. 有序列表

有顺序的步骤用数字列表。

```markdown
1. 创建环境。
2. 安装依赖。
3. 登录 SwanLab。
4. 开始训练。
```

效果：

1. 创建环境。
2. 安装依赖。
3. 登录 SwanLab。
4. 开始训练。

实际写作时，即使每一项都写成 `1.`，很多渲染器也会自动编号：

```markdown
1. 第一步
1. 第二步
1. 第三步
```

## 8. 嵌套列表

嵌套列表需要缩进。通常用两个或四个空格，保持一致即可。

```markdown
- 环境配置
  - Python 3.10
  - PyTorch
- 日志工具
  - TensorBoard
  - SwanLab
```

效果：

- 环境配置
  - Python 3.10
  - PyTorch
- 日志工具
  - TensorBoard
  - SwanLab

## 9. 链接

链接格式：

```markdown
[显示文字](链接地址)
```

示例：

```markdown
[SwanLab 官网](https://swanlab.cn)
[PyTorch 官网](https://pytorch.org)
```

效果：

[SwanLab 官网](https://swanlab.cn)

[PyTorch 官网](https://pytorch.org)

## 10. 图片

图片比链接多一个感叹号：

```markdown
![图片说明](图片路径或网址)
```

示例：

```markdown
![训练曲线](images/loss_curve.png)
```

如果图片在项目文件夹里，推荐使用相对路径。

## 11. 引用

引用用 `>`。

```markdown
> 这是一段引用，常用来写提示、说明、注意事项。
```

效果：

> 这是一段引用，常用来写提示、说明、注意事项。

也可以写成提示块：

```markdown
> 注意：API Key 不要发给别人。
```

## 12. 表格

表格由表头、分隔线和内容组成。

```markdown
| 名称 | 作用 |
| --- | --- |
| Python | 运行代码 |
| PyTorch | 深度学习框架 |
| SwanLab | 记录训练实验 |
```

效果：

| 名称 | 作用 |
| --- | --- |
| Python | 运行代码 |
| PyTorch | 深度学习框架 |
| SwanLab | 记录训练实验 |

可以用冒号控制对齐：

```markdown
| 左对齐 | 居中 | 右对齐 |
| :--- | :---: | ---: |
| A | B | C |
```

## 13. 分割线

用三个或更多 `-` 可以写分割线。

```markdown
---
```

效果：

---

## 14. 任务列表

任务列表适合写待办事项。

```markdown
- [x] 创建 conda 环境
- [x] 安装 PyTorch
- [ ] 跑通第一个训练脚本
```

效果：

- [x] 创建 conda 环境
- [x] 安装 PyTorch
- [ ] 跑通第一个训练脚本

## 15. 转义字符

如果你想显示 Markdown 语法字符本身，可以在前面加反斜杠 `\`。

```markdown
\# 这不会变成标题
\* 这不会变成列表
```

效果：

\# 这不会变成标题

\* 这不会变成列表

## 16. 常见 markdownlint 提示

VS Code 里如果安装了 markdownlint 插件，它会检查 Markdown 格式。下面是常见提示的含义。

| 规则 | 含义 | 修复方法 |
| --- | --- | --- |
| `MD009` | 行尾有多余空格 | 删除行末空格；如果确实要换行，保留两个空格 |
| `MD022` | 标题前后缺少空行 | 在标题上方和下方各空一行 |
| `MD031` | 代码块前后缺少空行 | 在三个反引号前后各空一行 |
| `MD046` | 代码块风格不统一 | 统一使用三个反引号的 fenced code block |
| `MD047` | 文件结尾没有单独换行 | 文件最后保留一个空行 |

例如，下面这种写法容易触发 `MD031`：

````markdown
说明文字
```python
print("hello")
```
下一段文字
````

推荐改成：

````markdown
说明文字

```python
print("hello")
```

下一段文字
````

## 17. README 的推荐结构

一个项目的 README 可以这样组织：

````markdown
# 项目名称

一句话说明这个项目做什么。

## 环境配置

```powershell
conda create -n rl310 python=3.10
conda activate rl310
pip install -r requirements.txt
```

## 使用方法

```powershell
python run_trpo.py env.name=HalfCheetah-v5
```

## 目录结构

```text
RLfromscratch/
  agent/
  algorithm/
  config/
  run_trpo.py
```

## 常见问题

- 如果 SwanLab 未登录，先执行 `swanlab login`。
- 如果找不到 MuJoCo 环境，确认已经安装 `mujoco`。
````

## 18. 写文档的小习惯

- 标题前后空一行。
- 列表前后空一行。
- 代码块前后空一行。
- 命令、路径、文件名使用行内代码格式。
- 长命令优先放进代码块。
- 不要在普通正文前面随便加四个空格，否则可能被当成代码块。
- 文件结尾保留一个换行。

## 19. Markdown 和普通文本的关系

Markdown 文件本质上还是普通文本文件。你看到的 `.md` 原文是给人和机器看的结构化文本；按 `Ctrl + Shift + V` 打开的预览，是 VS Code 帮你渲染后的阅读版本。

所以写 Markdown 时可以记住一句话：原文要清楚，预览会漂亮。
