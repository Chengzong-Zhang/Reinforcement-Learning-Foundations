# HTML 使用指南

这份文档用来快速了解 HTML 的基本写法。HTML 的目标是描述网页的结构：哪里是标题，哪里是段落，哪里是图片、链接、表格、表单。它通常和 CSS、JavaScript 一起使用。

可以先记住这句话：

> HTML 管结构，CSS 管样式，JavaScript 管交互。

## 1. HTML 文件是什么

HTML 文件通常以 `.html` 结尾，例如：

```text
index.html
about.html
demo.html
```

你可以用 VS Code 编辑 HTML 文件，然后直接用浏览器打开它。

最小的 HTML 文件长这样：

```html
<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8">
    <title>我的第一个网页</title>
  </head>
  <body>
    <h1>你好，HTML</h1>
    <p>这是一个段落。</p>
  </body>
</html>
```

其中：

| 部分 | 作用 |
| --- | --- |
| `<!doctype html>` | 告诉浏览器这是现代 HTML 文档 |
| `<html>` | 整个网页的根元素 |
| `<head>` | 放网页信息，不直接显示在页面正文里 |
| `<meta charset="utf-8">` | 设置字符编码，避免中文乱码 |
| `<title>` | 浏览器标签页标题 |
| `<body>` | 页面真正显示出来的内容 |

## 2. 标签和元素

HTML 由标签组成。大多数标签成对出现：

```html
<p>这是一个段落。</p>
```

这里：

- `<p>` 是开始标签。
- `</p>` 是结束标签。
- 中间的文字是内容。
- 整体叫一个元素。

有些标签不需要结束标签，例如：

```html
<br>
<img src="cat.png" alt="图片说明">
<input type="text">
```

## 3. 常用文本标签

标题标签从 `<h1>` 到 `<h6>`，数字越大级别越低。

```html
<h1>一级标题</h1>
<h2>二级标题</h2>
<h3>三级标题</h3>
```

段落用 `<p>`：

```html
<p>这是一个段落。段落适合写普通正文。</p>
```

加粗和强调：

```html
<strong>重要内容</strong>
<em>强调内容</em>
```

换行：

```html
第一行<br>
第二行
```

一般写文章时优先用 `<p>` 分段，不要频繁用 `<br>` 硬换行。

## 4. 链接

链接用 `<a>` 标签。

```html
<a href="https://pytorch.org">访问 PyTorch 官网</a>
```

如果想在新标签页打开：

```html
<a href="https://pytorch.org" target="_blank" rel="noopener noreferrer">
  访问 PyTorch 官网
</a>
```

其中：

| 属性 | 作用 |
| --- | --- |
| `href` | 链接地址 |
| `target="_blank"` | 在新标签页打开 |
| `rel="noopener noreferrer"` | 提升安全性，常和 `_blank` 一起使用 |

## 5. 图片

图片用 `<img>` 标签。

```html
<img src="images/loss_curve.png" alt="训练曲线">
```

常见属性：

| 属性 | 作用 |
| --- | --- |
| `src` | 图片路径或网址 |
| `alt` | 图片加载失败时显示的说明，也方便无障碍阅读 |
| `width` | 图片宽度 |
| `height` | 图片高度 |

示例：

```html
<img src="images/loss_curve.png" alt="训练曲线" width="600">
```

如果图片就在当前 HTML 文件旁边，可以直接写文件名：

```html
<img src="demo.png" alt="示例图片">
```

## 6. 列表

无序列表用 `<ul>`，列表项用 `<li>`。

```html
<ul>
  <li>Python 3.10</li>
  <li>PyTorch</li>
  <li>SwanLab</li>
</ul>
```

有序列表用 `<ol>`。

```html
<ol>
  <li>创建环境</li>
  <li>安装依赖</li>
  <li>运行训练</li>
</ol>
```

## 7. 表格

表格用 `<table>`。

```html
<table>
  <thead>
    <tr>
      <th>名称</th>
      <th>作用</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>HTML</td>
      <td>描述网页结构</td>
    </tr>
    <tr>
      <td>CSS</td>
      <td>控制网页样式</td>
    </tr>
  </tbody>
</table>
```

其中：

| 标签 | 作用 |
| --- | --- |
| `<table>` | 表格 |
| `<thead>` | 表头区域 |
| `<tbody>` | 表格主体 |
| `<tr>` | 一行 |
| `<th>` | 表头单元格 |
| `<td>` | 普通单元格 |

HTML 表格默认样式很朴素，通常需要 CSS 美化。

## 8. 容器标签

常见容器标签有 `<div>` 和 `<span>`。

`<div>` 是块级容器，常用来包住一块内容：

```html
<div>
  <h2>环境配置</h2>
  <p>这里介绍项目需要的环境。</p>
</div>
```

`<span>` 是行内容器，常用来包住一小段文字：

```html
<p>当前状态：<span>已完成</span></p>
```

简单理解：

| 标签 | 常见用途 |
| --- | --- |
| `<div>` | 包一整块内容 |
| `<span>` | 包一小段文字 |

## 9. 语义化标签

现代 HTML 推荐使用语义化标签，让结构更清楚。

```html
<header>页面头部</header>
<nav>导航栏</nav>
<main>主要内容</main>
<section>一个章节</section>
<article>一篇文章或独立内容</article>
<aside>侧边栏</aside>
<footer>页面底部</footer>
```

一个简单页面可以这样写：

```html
<body>
  <header>
    <h1>强化学习笔记</h1>
  </header>

  <main>
    <section>
      <h2>TRPO</h2>
      <p>这里记录 TRPO 的学习内容。</p>
    </section>
  </main>

  <footer>
    <p>最后更新：2026-05-14</p>
  </footer>
</body>
```

语义化标签的好处是：人更容易读，搜索引擎和辅助工具也更容易理解页面。

## 10. 表单

表单用于输入内容。

```html
<form>
  <label for="username">用户名</label>
  <input id="username" name="username" type="text">

  <label for="password">密码</label>
  <input id="password" name="password" type="password">

  <button type="submit">提交</button>
</form>
```

常见输入类型：

| 类型 | 示例 |
| --- | --- |
| 文本 | `<input type="text">` |
| 密码 | `<input type="password">` |
| 数字 | `<input type="number">` |
| 复选框 | `<input type="checkbox">` |
| 单选框 | `<input type="radio">` |
| 文件 | `<input type="file">` |
| 多行文本 | `<textarea></textarea>` |

`label` 的 `for` 要对应 `input` 的 `id`，这样点击文字也能聚焦输入框。

## 11. 按钮

按钮用 `<button>`。

```html
<button type="button">普通按钮</button>
<button type="submit">提交表单</button>
<button type="reset">重置表单</button>
```

如果按钮在表单里，最好明确写 `type`，避免默认提交表单。

## 12. 注释

HTML 注释不会显示在页面上。

```html
<!-- 这里是注释 -->
```

注释适合说明复杂结构，但不要把密码、API Key 这类敏感信息写进注释。

## 13. 属性

标签可以有属性，属性写在开始标签里。

```html
<a href="https://swanlab.cn">SwanLab</a>
<img src="demo.png" alt="示例图片">
<input type="text" placeholder="请输入用户名">
```

常见属性：

| 属性 | 常见位置 | 作用 |
| --- | --- | --- |
| `id` | 大多数标签 | 唯一标识一个元素 |
| `class` | 大多数标签 | 给元素分组，常用于 CSS |
| `href` | `<a>` | 链接地址 |
| `src` | `<img>`、`script` | 资源路径 |
| `alt` | `<img>` | 图片说明 |
| `type` | `<input>`、`button` | 指定类型 |

## 14. CSS 写在哪里

CSS 用来控制样式。最简单可以写在 `<style>` 标签里。

```html
<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8">
    <title>CSS 示例</title>
    <style>
      body {
        font-family: Arial, sans-serif;
        line-height: 1.6;
      }

      h1 {
        color: #2563eb;
      }
    </style>
  </head>
  <body>
    <h1>带样式的网页</h1>
    <p>这段文字会使用上面定义的样式。</p>
  </body>
</html>
```

也可以把 CSS 单独放到 `.css` 文件里：

```html
<link rel="stylesheet" href="styles.css">
```

推荐稍微正式一点的页面使用单独的 CSS 文件。

## 15. JavaScript 写在哪里

JavaScript 用来做交互。可以写在 `<script>` 标签里。

```html
<button id="helloBtn" type="button">点我</button>

<script>
  const button = document.querySelector("#helloBtn");

  button.addEventListener("click", () => {
    alert("你好，HTML！");
  });
</script>
```

也可以引用单独的 `.js` 文件：

```html
<script src="main.js"></script>
```

通常把 `<script>` 放在 `</body>` 前面，这样页面内容先加载出来。

## 16. 一个完整示例

可以新建一个 `index.html`，写入下面内容，然后用浏览器打开。

```html
<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>强化学习实验记录</title>
    <style>
      body {
        max-width: 800px;
        margin: 40px auto;
        padding: 0 20px;
        font-family: Arial, "Microsoft YaHei", sans-serif;
        line-height: 1.7;
      }

      code {
        background: #f2f4f8;
        padding: 2px 6px;
        border-radius: 4px;
      }
    </style>
  </head>
  <body>
    <header>
      <h1>强化学习实验记录</h1>
      <p>记录环境配置、训练命令和实验结果。</p>
    </header>

    <main>
      <section>
        <h2>环境配置</h2>
        <ul>
          <li>Python 3.10</li>
          <li>PyTorch</li>
          <li>Gymnasium</li>
          <li>SwanLab</li>
        </ul>
      </section>

      <section>
        <h2>训练命令</h2>
        <p>运行 TRPO：</p>
        <pre><code>python run_trpo.py env.name=HalfCheetah-v5</code></pre>
      </section>

      <section>
        <h2>实验链接</h2>
        <p>
          查看实验：
          <a href="https://swanlab.cn" target="_blank" rel="noopener noreferrer">
            SwanLab
          </a>
        </p>
      </section>
    </main>
  </body>
</html>
```

## 17. HTML 和 Markdown 的区别

Markdown 更适合写文档：

```markdown
## 环境配置

- Python 3.10
- PyTorch
- SwanLab
```

HTML 更适合做网页结构：

```html
<section>
  <h2>环境配置</h2>
  <ul>
    <li>Python 3.10</li>
    <li>PyTorch</li>
    <li>SwanLab</li>
  </ul>
</section>
```

简单选择：

| 需求 | 推荐 |
| --- | --- |
| README | Markdown |
| 学习笔记 | Markdown |
| 项目说明 | Markdown |
| 复杂网页 | HTML + CSS |
| 需要精确排版 | HTML + CSS |
| 需要交互 | HTML + CSS + JavaScript |

## 18. 在 VS Code 里怎么写 HTML

常用操作：

| 操作 | 方法 |
| --- | --- |
| 新建 HTML 文件 | 新建文件，命名为 `index.html` |
| 快速生成骨架 | 输入 `!`，然后按 `Tab` |
| 格式化文件 | `Shift + Alt + F` |
| 在浏览器打开 | 在文件资源管理器中双击 HTML 文件，或右键选择打开方式 |

如果安装了 Live Server 插件，可以右键 HTML 文件，选择：

```text
Open with Live Server
```

这样修改代码后浏览器会自动刷新。

## 19. 常见错误

忘记结束标签：

```html
<p>这是一个段落
```

推荐写成：

```html
<p>这是一个段落</p>
```

属性没有加引号：

```html
<img src=demo.png alt=示例图片>
```

推荐写成：

```html
<img src="demo.png" alt="示例图片">
```

中文乱码：

```html
<meta charset="utf-8">
```

把这一行放进 `<head>` 里。

图片路径错误：

```html
<img src="images/demo.png" alt="示例图片">
```

确认 `images` 文件夹和 `demo.png` 是否真的存在，路径是否相对于当前 HTML 文件。

## 20. 学习路线

建议按这个顺序学：

1. 先学 HTML 的基础标签。
2. 再学 CSS 控制颜色、间距、布局。
3. 学会用浏览器开发者工具检查元素。
4. 再学 JavaScript 做按钮点击、数据渲染等交互。
5. 最后再接触 Vue、React 这类前端框架。

对现在的你来说，Markdown 继续用来写 README 和笔记；HTML 可以从一个 `index.html` 小页面开始练。
