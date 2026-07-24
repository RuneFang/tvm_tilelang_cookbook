#!/usr/bin/env python3
"""
把 docs/ 前 5 章 markdown 渲染成两份自包含 HTML：
  - dist/docs.html   标准阅读排版（侧栏目录 + 正文）
  - dist/zsxq.html   知识星球风格（timeline 卡片流，抽取每章精华帖）

零外部依赖：仅使用 Python stdlib；输出 HTML 内联 CSS、不引 CDN。
"""

from __future__ import annotations
import html
import re
from pathlib import Path
from dataclasses import dataclass, field

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "docs"
DIST_DIR = ROOT / "dist"
DIST_DIR.mkdir(exist_ok=True)

CHAPTERS = [
    ("01_hello_tilelang.md",       "第 1 章 · 一个最小的 TileLang 例子"),
    ("02_tvm_tir_basics.md",       "第 2 章 · TVM / TIR 基础概念"),
    ("03_tilelang_dsl.md",         "第 3 章 · TileLang DSL 关键字"),
    ("04_pass_system.md",          "第 4 章 · Pass 系统与 Pipeline"),
    ("05_lowering_pipeline.md",    "第 5 章 · Lowering Pipeline 巡礼"),
]


# ---------------------------------------------------------------------------
# 一个"够用"的 Markdown → HTML 转换器
# ---------------------------------------------------------------------------

@dataclass
class Doc:
    file: str
    title: str
    html: str
    toc: list = field(default_factory=list)   # [(level, anchor, text)]


def slugify(text: str, used: dict) -> str:
    s = re.sub(r"[^\w\u4e00-\u9fff]+", "-", text.lower()).strip("-")
    if not s:
        s = "sec"
    if s in used:
        used[s] += 1
        return f"{s}-{used[s]}"
    used[s] = 0
    return s


def render_inline(text: str) -> str:
    """处理行内标记：转义 → 反引号 code → 加粗 → 链接。"""
    # 先按 `code` 切开，切开的 code 段不再受其它规则影响
    parts = re.split(r"(`[^`\n]+`)", text)
    out = []
    for p in parts:
        if p.startswith("`") and p.endswith("`") and len(p) >= 2:
            out.append(f"<code>{html.escape(p[1:-1])}</code>")
        else:
            esc = html.escape(p)
            # 链接 [text](url)
            esc = re.sub(
                r"\[([^\]]+)\]\(([^)]+)\)",
                lambda m: f'<a href="{m.group(2)}">{m.group(1)}</a>',
                esc,
            )
            # 加粗 **x**
            esc = re.sub(r"\*\*([^*\n]+)\*\*", r"<strong>\1</strong>", esc)
            out.append(esc)
    return "".join(out)


def parse_markdown(md_text: str, file_slug: str) -> Doc:
    """把整份 markdown 转成 HTML 段与 TOC 列表。"""
    lines = md_text.splitlines()
    i, n = 0, len(lines)
    parts = []
    toc = []
    used_slugs: dict = {}
    doc_title = ""

    def collect_paragraph(start_idx: int) -> tuple[int, str]:
        buf = []
        j = start_idx
        while j < n:
            ln = lines[j]
            stripped = ln.strip()
            if not stripped:
                break
            # 遇到新块结构就停
            if stripped.startswith(("#", ">", "```", "|", "- ", "* ", "---")):
                break
            if re.match(r"^\d+\.\s", stripped):
                break
            buf.append(ln)
            j += 1
        return j, " ".join(buf)

    while i < n:
        line = lines[i]
        stripped = line.strip()

        # 空行
        if not stripped:
            i += 1
            continue

        # ATX 标题
        m = re.match(r"^(#{1,6})\s+(.+?)\s*#*$", line)
        if m:
            level = len(m.group(1))
            text = m.group(2).strip()
            anchor = f"{file_slug}--{slugify(text, used_slugs)}"
            html_text = render_inline(text)
            parts.append(f'<h{level} id="{anchor}"><a class="anchor" href="#{anchor}">¶</a>{html_text}</h{level}>')
            toc.append((level, anchor, text))
            if level == 1 and not doc_title:
                doc_title = text
            i += 1
            continue

        # 分割线
        if re.fullmatch(r"-{3,}|\*{3,}|_{3,}", stripped):
            parts.append("<hr>")
            i += 1
            continue

        # 代码块
        if stripped.startswith("```"):
            lang = stripped[3:].strip()
            i += 1
            code_lines = []
            while i < n and not lines[i].lstrip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            i += 1  # 跳过收尾 ```
            code_html = html.escape("\n".join(code_lines))
            cls = f' class="lang-{lang}"' if lang else ""
            parts.append(f'<pre><code{cls}>{code_html}</code></pre>')
            continue

        # 表格
        if stripped.startswith("|") and stripped.endswith("|"):
            table_lines = []
            while i < n and lines[i].strip().startswith("|"):
                table_lines.append(lines[i].strip())
                i += 1
            if len(table_lines) >= 2 and re.match(r"^\|[\s\-|:]+\|$", table_lines[1]):
                header_cells = [c.strip() for c in table_lines[0].strip("|").split("|")]
                rows = []
                for row in table_lines[2:]:
                    rows.append([c.strip() for c in row.strip("|").split("|")])
                thead = "<tr>" + "".join(f"<th>{render_inline(c)}</th>" for c in header_cells) + "</tr>"
                tbody = "".join(
                    "<tr>" + "".join(f"<td>{render_inline(c)}</td>" for c in r) + "</tr>"
                    for r in rows
                )
                parts.append(f'<div class="table-wrap"><table><thead>{thead}</thead><tbody>{tbody}</tbody></table></div>')
                continue
            # 不是标准表格：按段落降级
            parts.append('<p>' + render_inline(" ".join(table_lines)) + '</p>')
            continue

        # 引用块 (支持多行合并、内部含代码块)
        if stripped.startswith(">"):
            quote_lines = []
            while i < n and (lines[i].startswith(">") or lines[i].strip().startswith(">")):
                # 去掉前导 ">"（可能是 "> " 或 ">"）
                ln = lines[i]
                if ln.startswith("> "):
                    quote_lines.append(ln[2:])
                elif ln.startswith(">"):
                    quote_lines.append(ln[1:])
                else:
                    quote_lines.append(ln)
                i += 1
            # 递归处理引用块内的内容（其内部也可能有代码块 / 列表 / 表格）
            inner_doc = parse_markdown("\n".join(quote_lines), file_slug + "-q" + str(len(parts)))
            parts.append(f'<blockquote>{inner_doc.html}</blockquote>')
            continue

        # 无序列表
        if re.match(r"^[-*]\s+", stripped):
            items = []
            while i < n and re.match(r"^[-*]\s+", lines[i].strip()):
                text = re.sub(r"^[-*]\s+", "", lines[i].strip())
                # 支持列表项延续行（缩进 2+ 空格且非新块）
                i += 1
                cont = []
                while i < n and lines[i].startswith(("  ", "\t")) and lines[i].strip() and not re.match(r"^[-*]\s+", lines[i].strip()):
                    cont.append(lines[i].strip())
                    i += 1
                if cont:
                    text = text + " " + " ".join(cont)
                items.append(text)
            body = "".join(f"<li>{render_inline(it)}</li>" for it in items)
            parts.append(f"<ul>{body}</ul>")
            continue

        # 有序列表
        if re.match(r"^\d+\.\s+", stripped):
            items = []
            while i < n and re.match(r"^\d+\.\s+", lines[i].strip()):
                text = re.sub(r"^\d+\.\s+", "", lines[i].strip())
                i += 1
                cont = []
                while i < n and lines[i].startswith(("  ", "\t")) and lines[i].strip() and not re.match(r"^\d+\.\s+", lines[i].strip()):
                    cont.append(lines[i].strip())
                    i += 1
                if cont:
                    text = text + " " + " ".join(cont)
                items.append(text)
            body = "".join(f"<li>{render_inline(it)}</li>" for it in items)
            parts.append(f"<ol>{body}</ol>")
            continue

        # 段落
        j, para = collect_paragraph(i)
        parts.append(f"<p>{render_inline(para)}</p>")
        i = j

    return Doc(file="", title=doc_title, html="\n".join(parts), toc=toc)


# ---------------------------------------------------------------------------
# 精华帖抽取（给知识星球风格用）
# ---------------------------------------------------------------------------

@dataclass
class Post:
    chapter_idx: int
    chapter_title: str
    kind: str        # "tldr" | "concept" | "trap" | "takeaway"
    title: str
    body_md: str


def extract_posts(chapter_idx: int, chapter_title: str, md_text: str) -> list[Post]:
    """从一章 md 里挑 TL;DR / 概念卡 / 陷阱清单 / 本章要带走 这四类精华段。"""
    posts: list[Post] = []
    lines = md_text.splitlines()
    n = len(lines)

    # 1. TL;DR：文件开头 > **TL;DR** 那个 blockquote
    i = 0
    while i < n:
        if lines[i].strip().startswith(">") and "TL;DR" in lines[i]:
            buf = []
            while i < n and lines[i].strip().startswith(">"):
                buf.append(lines[i].lstrip(">").lstrip())
                i += 1
            posts.append(Post(chapter_idx, chapter_title, "tldr", "TL;DR", "\n".join(buf)))
            break
        i += 1

    # 2. 概念卡：所有以 "### 概念卡" 或 "> 📌 " 或 "> 💡 " 开头的段
    i = 0
    while i < n:
        line = lines[i]
        stripped = line.strip()
        # ### 概念卡：xxx
        m = re.match(r"^###\s+(?:概念卡[:：]?|📌)\s*(.*)$", stripped)
        if m:
            title = m.group(1).strip() or "概念卡"
            # 脱掉粗体标记和行内反引号，作为纯文本标题
            title = re.sub(r"\*\*(.+?)\*\*", r"\1", title)
            title = re.sub(r"`([^`]+)`", r"\1", title)
            i += 1
            buf = []
            while i < n and not re.match(r"^#{1,4}\s+", lines[i].strip()) and not re.match(r"^##\s+", lines[i].strip()):
                buf.append(lines[i])
                i += 1
            posts.append(Post(chapter_idx, chapter_title, "concept",
                              title.replace("**", ""), "\n".join(buf).strip()))
            continue
        # blockquote 提示（📌/💡/⚠️ 开头且较长）
        if stripped.startswith(">") and any(t in stripped for t in ["📌", "💡", "⚠️"]):
            buf = []
            while i < n and lines[i].strip().startswith(">"):
                buf.append(lines[i].lstrip(">").lstrip())
                i += 1
            body = "\n".join(buf).strip()
            if len(body) > 80:  # 过滤太短的
                # 从首行提炼标题
                first = buf[0].lstrip("*").strip() if buf else "提示"
                # 去掉粗体标记做标题
                # 去掉粗体标记 + 去掉行内反引号后作为标题
                title = re.sub(r"\*\*(.+?)\*\*.*", r"\1", first)
                title = re.sub(r"`([^`]+)`", r"\1", title)
                title = title[:60] or "提示"
                # 分类：📌 / 💡 归 concept，⚠️ 归 trap
                kind = "trap" if "⚠️" in first else "concept"
                posts.append(Post(chapter_idx, chapter_title, kind, title, body))
            continue
        i += 1

    # 3. 陷阱清单：找 "陷阱清单" 或 "本章要带走" 小节
    i = 0
    while i < n:
        m = re.match(r"^##\s+.*?(陷阱清单|要带走的.*?件事)", lines[i].strip())
        if m:
            heading_text = lines[i].strip().lstrip("#").strip()
            i += 1
            buf = []
            while i < n and not re.match(r"^##\s+", lines[i].strip()):
                buf.append(lines[i])
                i += 1
            kind = "trap" if "陷阱" in heading_text else "takeaway"
            posts.append(Post(chapter_idx, chapter_title, kind, heading_text, "\n".join(buf).strip()))
            continue
        i += 1

    return posts


# ---------------------------------------------------------------------------
# 标准阅读排版 HTML
# ---------------------------------------------------------------------------

DOCS_CSS = r"""
* { box-sizing: border-box; }
:root {
  --fg: #1f2328;
  --fg-soft: #57606a;
  --bg: #ffffff;
  --bg-soft: #f6f8fa;
  --border: #d0d7de;
  --accent: #0969da;
  --accent-soft: #ddf4ff;
  --code-bg: #f6f8fa;
  --code-fg: #24292f;
  --quote-bar: #d0d7de;
  --quote-bg: #f6f8fa;
  --warn: #fff8c5;
  --warn-bar: #d4a72c;
}
html, body { margin: 0; padding: 0; }
body {
  font-family: -apple-system, "Segoe UI", "PingFang SC", "Microsoft YaHei", Helvetica, Arial, sans-serif;
  font-size: 15px;
  line-height: 1.75;
  color: var(--fg);
  background: var(--bg);
}
.layout { display: grid; grid-template-columns: 280px minmax(0,1fr); min-height: 100vh; }
aside.sidebar {
  border-right: 1px solid var(--border);
  background: var(--bg-soft);
  padding: 24px 20px;
  overflow-y: auto;
  position: sticky; top: 0; height: 100vh;
}
aside.sidebar .brand {
  font-weight: 700; font-size: 16px; margin-bottom: 4px; color: var(--fg);
}
aside.sidebar .brand small {
  display: block; font-weight: 400; font-size: 12px; color: var(--fg-soft); margin-top: 4px;
}
aside.sidebar hr {
  border: 0; border-top: 1px solid var(--border); margin: 16px 0;
}
aside.sidebar .toc-chapter {
  font-weight: 600; font-size: 13px; margin: 14px 0 6px; color: var(--fg);
}
aside.sidebar .toc-item {
  display: block; padding: 3px 0 3px 12px; font-size: 13px;
  color: var(--fg-soft); text-decoration: none; border-left: 2px solid transparent;
}
aside.sidebar .toc-item:hover { color: var(--accent); border-left-color: var(--accent); background: #eaf2fb; }
aside.sidebar .toc-item.lvl-2 { padding-left: 12px; }
aside.sidebar .toc-item.lvl-3 { padding-left: 24px; font-size: 12.5px; }
aside.sidebar .toc-item.lvl-4 { padding-left: 36px; font-size: 12px; color: #8b95a1; }
main.content { padding: 40px 56px 120px; max-width: 900px; }
article + article { margin-top: 80px; border-top: 2px solid var(--border); padding-top: 60px; }

h1 { font-size: 30px; font-weight: 700; margin: 24px 0 20px; padding-bottom: 12px; border-bottom: 1px solid var(--border); }
h2 { font-size: 22px; font-weight: 700; margin: 40px 0 14px; padding-bottom: 6px; border-bottom: 1px solid var(--border); }
h3 { font-size: 18px; font-weight: 600; margin: 28px 0 10px; }
h4 { font-size: 15.5px; font-weight: 600; margin: 20px 0 8px; }
h1 .anchor, h2 .anchor, h3 .anchor, h4 .anchor { opacity: 0; margin-left: -18px; padding-right: 4px; color: var(--fg-soft); text-decoration: none; }
h1:hover .anchor, h2:hover .anchor, h3:hover .anchor, h4:hover .anchor { opacity: 1; }

p { margin: 12px 0; }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
strong { color: var(--fg); font-weight: 650; }

code { background: var(--code-bg); color: var(--code-fg); padding: 1px 6px; border-radius: 4px; font-size: 0.88em;
       font-family: "SF Mono", "JetBrains Mono", Menlo, Consolas, monospace; }
pre { background: var(--code-bg); border: 1px solid var(--border); border-radius: 8px; padding: 16px 18px; overflow-x: auto;
      font-size: 13px; line-height: 1.6; margin: 14px 0; }
pre code { background: transparent; padding: 0; font-size: 13px; }

blockquote { border-left: 4px solid var(--quote-bar); background: var(--quote-bg);
             margin: 16px 0; padding: 12px 16px; color: var(--fg); border-radius: 0 6px 6px 0; }
blockquote > *:first-child { margin-top: 0; }
blockquote > *:last-child { margin-bottom: 0; }
blockquote p { margin: 8px 0; }

.table-wrap { overflow-x: auto; margin: 14px 0; }
table { border-collapse: collapse; width: 100%; font-size: 14px; }
th, td { border: 1px solid var(--border); padding: 8px 12px; text-align: left; vertical-align: top; }
th { background: var(--bg-soft); font-weight: 600; }
tr:nth-child(even) td { background: #fafbfc; }

ul, ol { padding-left: 24px; margin: 10px 0; }
li { margin: 4px 0; }

hr { border: 0; border-top: 1px solid var(--border); margin: 30px 0; }

@media (max-width: 900px) {
  .layout { grid-template-columns: 1fr; }
  aside.sidebar { position: static; height: auto; border-right: none; border-bottom: 1px solid var(--border); }
  main.content { padding: 24px 20px 80px; }
}
"""

DOCS_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>TVM & TileLang Cookbook · 前 5 章</title>
<style>{css}</style>
</head>
<body>
<div class="layout">
  <aside class="sidebar">
    <div class="brand">TVM · TileLang Cookbook<small>前 5 章精选 · 阅读模式</small></div>
    <hr>
    {sidebar}
    <hr>
    <a class="toc-item" href="zsxq.html" style="color: var(--accent);">→ 切到知识星球模式</a>
  </aside>
  <main class="content">
{articles}
  </main>
</div>
</body>
</html>
"""


def build_docs_html(docs: list[Doc]) -> str:
    # 侧栏
    sidebar_parts = []
    for d in docs:
        chapter_anchor = d.toc[0][1] if d.toc else ""
        sidebar_parts.append(f'<div class="toc-chapter"><a href="#{chapter_anchor}" style="color:inherit;text-decoration:none;">{html.escape(d.title)}</a></div>')
        for level, anchor, text in d.toc[1:]:
            if level > 4:
                continue
            cls = f"toc-item lvl-{level}"
            sidebar_parts.append(f'<a class="{cls}" href="#{anchor}">{html.escape(text)}</a>')
    sidebar_html = "\n".join(sidebar_parts)

    articles = "\n".join(f'<article>{d.html}</article>' for d in docs)
    return DOCS_TEMPLATE.format(css=DOCS_CSS, sidebar=sidebar_html, articles=articles)


# ---------------------------------------------------------------------------
# 知识星球风格 HTML
# ---------------------------------------------------------------------------

ZSXQ_CSS = r"""
* { box-sizing: border-box; }
:root {
  --primary: #ff8c1a;
  --primary-dark: #e56b00;
  --bg: #f5f5f7;
  --card-bg: #ffffff;
  --text: #1a1a1a;
  --text-soft: #6b7280;
  --text-mute: #9ca3af;
  --border: #ececec;
  --tag-bg: #fff2e6;
  --tag-fg: #d96b00;
  --tldr-bg: linear-gradient(135deg, #ff9944 0%, #ff6b1a 100%);
  --concept-bg: #eff6ff;
  --concept-bar: #3b82f6;
  --trap-bg: #fef3c7;
  --trap-bar: #f59e0b;
  --takeaway-bg: #ecfdf5;
  --takeaway-bar: #10b981;
}
html, body { margin: 0; padding: 0; }
body {
  font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif;
  font-size: 15px; line-height: 1.7; background: var(--bg); color: var(--text);
}

header.hero {
  background: linear-gradient(135deg, #ff6b1a 0%, #ff9944 100%);
  color: #fff; padding: 40px 24px 70px; text-align: center;
  position: relative; overflow: hidden;
}
header.hero::before {
  content: ""; position: absolute; top: -50%; right: -20%;
  width: 400px; height: 400px; border-radius: 50%;
  background: rgba(255,255,255,0.08);
}
header.hero::after {
  content: ""; position: absolute; bottom: -30%; left: -10%;
  width: 300px; height: 300px; border-radius: 50%;
  background: rgba(255,255,255,0.06);
}
header.hero h1 { margin: 0 0 8px; font-size: 26px; font-weight: 700; position: relative; }
header.hero p { margin: 0; opacity: 0.92; font-size: 14px; position: relative; }
header.hero .planet-icon {
  display: inline-block; width: 56px; height: 56px; border-radius: 50%;
  background: rgba(255,255,255,0.25); border: 2px solid rgba(255,255,255,0.6);
  line-height: 52px; text-align: center; font-size: 26px; margin-bottom: 10px;
  position: relative;
}
header.hero .stat-row {
  display: inline-flex; gap: 24px; margin-top: 14px; padding: 8px 20px;
  background: rgba(255,255,255,0.15); border-radius: 20px; font-size: 13px;
  position: relative;
}
header.hero .stat-row span b { font-weight: 700; margin-right: 4px; }

.timeline { max-width: 720px; margin: -40px auto 60px; padding: 0 16px; position: relative; }

.tag-bar {
  display: flex; gap: 8px; flex-wrap: wrap; padding: 14px 6px 22px; overflow-x: auto;
  position: relative; z-index: 2;
}
.tag-bar .tag {
  padding: 6px 14px; background: var(--card-bg); border-radius: 16px;
  font-size: 12.5px; color: var(--text-soft); cursor: pointer;
  border: 1px solid var(--border); white-space: nowrap; user-select: none;
}
.tag-bar .tag.active { background: var(--primary); color: #fff; border-color: var(--primary); }
.tag-bar .tag:hover:not(.active) { background: var(--tag-bg); color: var(--tag-fg); }

.post {
  background: var(--card-bg); border-radius: 14px; padding: 20px 22px;
  margin-bottom: 14px; box-shadow: 0 1px 3px rgba(0,0,0,0.04);
  transition: transform 0.15s, box-shadow 0.15s;
}
.post:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(0,0,0,0.06); }
.post-header { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
.avatar {
  width: 36px; height: 36px; border-radius: 50%; flex-shrink: 0;
  background: linear-gradient(135deg, #ff9944, #ff6b1a); color: #fff;
  display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 14px;
}
.post-meta { flex: 1; min-width: 0; }
.post-meta .author { font-weight: 600; font-size: 14px; }
.post-meta .sub { font-size: 12px; color: var(--text-mute); }
.post-badge {
  font-size: 11.5px; padding: 3px 9px; border-radius: 10px; font-weight: 600;
  background: var(--tag-bg); color: var(--tag-fg); white-space: nowrap;
}
.post-badge.tldr { background: #fff0e6; color: #d64d00; }
.post-badge.concept { background: var(--concept-bg); color: var(--concept-bar); }
.post-badge.trap { background: var(--trap-bg); color: #b45309; }
.post-badge.takeaway { background: var(--takeaway-bg); color: #047857; }

.post-title { font-size: 16px; font-weight: 700; margin: 4px 0 10px; color: var(--text); }
.post-body {
  font-size: 14.5px; line-height: 1.75; color: #2b2f36;
}
.post-body.tldr-body { padding: 14px 16px; background: var(--tldr-bg); color: #fff; border-radius: 10px; }
.post-body.tldr-body strong { color: #fff8e6; }
.post-body.tldr-body code { background: rgba(255,255,255,0.2); color: #fff; }
.post-body.tldr-body a { color: #fff; text-decoration: underline; }
.post-body.concept-body { border-left: 3px solid var(--concept-bar); padding-left: 14px; background: var(--concept-bg); border-radius: 0 8px 8px 0; padding: 12px 14px 12px 16px; }
.post-body.trap-body { border-left: 3px solid var(--trap-bar); padding: 12px 14px 12px 16px; background: var(--trap-bg); border-radius: 0 8px 8px 0; }
.post-body.takeaway-body { border-left: 3px solid var(--takeaway-bar); padding: 12px 14px 12px 16px; background: var(--takeaway-bg); border-radius: 0 8px 8px 0; }

.post-body p { margin: 8px 0; }
.post-body p:first-child { margin-top: 0; }
.post-body p:last-child { margin-bottom: 0; }
.post-body code { background: rgba(0,0,0,0.06); padding: 1px 5px; border-radius: 3px; font-size: 0.9em;
                  font-family: "SF Mono", Menlo, monospace; }
.post-body.tldr-body code { background: rgba(255,255,255,0.2); }
.post-body pre {
  background: rgba(0,0,0,0.05); padding: 12px; border-radius: 8px; overflow-x: auto;
  font-size: 12.5px; line-height: 1.6;
}
.post-body.tldr-body pre { background: rgba(0,0,0,0.2); color: #fff; }
.post-body pre code { background: transparent; padding: 0; }
.post-body ul, .post-body ol { padding-left: 22px; margin: 8px 0; }
.post-body a { color: var(--primary); }
.post-body strong { color: var(--text); font-weight: 650; }
.post-body table { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 13px; }
.post-body th, .post-body td { border: 1px solid rgba(0,0,0,0.08); padding: 6px 10px; text-align: left; }
.post-body th { background: rgba(0,0,0,0.03); font-weight: 600; }
.post-body blockquote { margin: 8px 0; padding: 8px 12px; border-left: 3px solid rgba(0,0,0,0.15); background: rgba(0,0,0,0.03); font-size: 0.95em; }

.post-footer {
  display: flex; gap: 22px; margin-top: 14px; padding-top: 12px;
  border-top: 1px solid var(--border); color: var(--text-mute); font-size: 13px;
}
.post-footer span { display: inline-flex; align-items: center; gap: 5px; cursor: pointer; }
.post-footer span:hover { color: var(--primary); }
.icon { display: inline-block; width: 15px; height: 15px; vertical-align: middle; }

.chapter-sep {
  display: flex; align-items: center; gap: 12px; margin: 30px 4px 14px;
  color: var(--text-soft); font-size: 13px; font-weight: 600;
}
.chapter-sep::before, .chapter-sep::after { content: ""; flex: 1; height: 1px; background: var(--border); }

footer.bottom {
  text-align: center; padding: 30px 20px 40px; color: var(--text-mute); font-size: 13px;
}
footer.bottom a { color: var(--primary); text-decoration: none; }

@media (max-width: 640px) {
  header.hero { padding: 30px 16px 60px; }
  .timeline { padding: 0 10px; }
  .post { padding: 16px 16px; }
}
"""

ZSXQ_JS = r"""
document.addEventListener('DOMContentLoaded', function () {
  var tags = document.querySelectorAll('.tag');
  var posts = document.querySelectorAll('.post');
  tags.forEach(function (t) {
    t.addEventListener('click', function () {
      tags.forEach(function (x) { x.classList.remove('active'); });
      t.classList.add('active');
      var filter = t.getAttribute('data-filter');
      posts.forEach(function (p) {
        if (filter === 'all' || p.getAttribute('data-kind') === filter) {
          p.style.display = '';
        } else {
          p.style.display = 'none';
        }
      });
      // 章节分隔条：如果它下方紧邻的所有 post 全部隐藏，也隐藏它自己
      document.querySelectorAll('.chapter-sep').forEach(function (sep) {
        var next = sep.nextElementSibling;
        var anyVisible = false;
        while (next && !next.classList.contains('chapter-sep')) {
          if (next.classList.contains('post') && next.style.display !== 'none') {
            anyVisible = true; break;
          }
          next = next.nextElementSibling;
        }
        sep.style.display = anyVisible ? '' : 'none';
      });
    });
  });
});
"""

ZSXQ_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Cookbook 星球 · 编译流水线精华</title>
<style>{css}</style>
</head>
<body>
<header class="hero">
  <div class="planet-icon">★</div>
  <h1>TVM · TileLang Cookbook 星球</h1>
  <p>把「编译流水线」讲人话 · 每章精华 · 一图一段读得完</p>
  <div class="stat-row">
    <span><b>{n_posts}</b> 精华帖</span>
    <span><b>5</b> 章主线</span>
    <span><b>2222</b> 行原文</span>
  </div>
</header>

<div class="timeline">
  <div class="tag-bar">
    <span class="tag active" data-filter="all">全部</span>
    <span class="tag" data-filter="tldr">TL;DR</span>
    <span class="tag" data-filter="concept">概念卡</span>
    <span class="tag" data-filter="trap">避坑</span>
    <span class="tag" data-filter="takeaway">要点</span>
  </div>

{posts}

  <footer class="bottom">
    完整正文见 <a href="docs.html">阅读模式</a> · 原文位于 <code>docs/</code> 目录 · 手写渲染
  </footer>
</div>

<script>{js}</script>
</body>
</html>
"""


KIND_LABEL = {
    "tldr":     ("★", "章节速读"),
    "concept":  ("◆", "概念卡"),
    "trap":     ("!", "避坑指南"),
    "takeaway": ("✓", "本章要点"),
}


def build_zsxq_html(all_posts: list[Post]) -> str:
    # 按章节分组
    parts = []
    total = len(all_posts)
    current_chapter = -1
    for p_idx, post in enumerate(all_posts):
        if post.chapter_idx != current_chapter:
            current_chapter = post.chapter_idx
            parts.append(f'<div class="chapter-sep">— {html.escape(post.chapter_title)} —</div>')

        # 渲染 body_md（复用主转换器）
        body_doc = parse_markdown(post.body_md, f"post-{p_idx}")
        body_html = body_doc.html

        icon, label = KIND_LABEL.get(post.kind, ("·", "笔记"))
        # 头像：章号
        avatar_char = f"C{post.chapter_idx}"
        # 时间戳（伪，做视觉用）
        elapsed = ["刚刚", "5 分钟前", "1 小时前", "今天", "昨天"][p_idx % 5]

        body_cls = f"post-body {post.kind}-body"
        title_html = html.escape(post.title)

        parts.append(f'''
<article class="post" data-kind="{post.kind}">
  <div class="post-header">
    <div class="avatar">{avatar_char}</div>
    <div class="post-meta">
      <div class="author">Cookbook · 第 {post.chapter_idx} 章</div>
      <div class="sub">{elapsed} · {html.escape(post.chapter_title)}</div>
    </div>
    <span class="post-badge {post.kind}">{icon} {label}</span>
  </div>
  <div class="post-title">{title_html}</div>
  <div class="{body_cls}">{body_html}</div>
  <div class="post-footer">
    <span>👁 {(p_idx + 1) * 137 % 999 + 100}</span>
    <span>👍 {(p_idx + 1) * 23 % 89 + 12}</span>
    <span>💬 {(p_idx + 1) * 7 % 21 + 3}</span>
    <span style="margin-left:auto;">☆ 收藏</span>
  </div>
</article>''')

    return ZSXQ_TEMPLATE.format(
        css=ZSXQ_CSS, js=ZSXQ_JS,
        n_posts=total, posts="\n".join(parts),
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    docs: list[Doc] = []
    all_posts: list[Post] = []

    for idx, (filename, title) in enumerate(CHAPTERS, start=1):
        path = DOCS_DIR / filename
        md_text = path.read_text(encoding="utf-8")
        file_slug = f"ch{idx:02d}"
        doc = parse_markdown(md_text, file_slug)
        doc.file = filename
        # 若解析得到的标题为空，用手工标题
        if not doc.title:
            doc.title = title
        # 把标题从 md h1 覆盖为章节标题（保证侧栏第一项一致）
        doc.toc = [(1, f"{file_slug}--top", title)] + [t for t in doc.toc if t[0] > 1]
        # 在 html 前面加锚点
        doc.html = f'<h1 id="{file_slug}--top">{html.escape(title)}</h1>\n' + doc.html
        docs.append(doc)

        posts = extract_posts(idx, title, md_text)
        all_posts.extend(posts)

    # 输出
    docs_html = build_docs_html(docs)
    (DIST_DIR / "docs.html").write_text(docs_html, encoding="utf-8")

    zsxq_html = build_zsxq_html(all_posts)
    (DIST_DIR / "zsxq.html").write_text(zsxq_html, encoding="utf-8")

    print(f"[OK] wrote {DIST_DIR / 'docs.html'}  ({len(docs_html):,} bytes)")
    print(f"[OK] wrote {DIST_DIR / 'zsxq.html'}  ({len(zsxq_html):,} bytes, {len(all_posts)} posts)")


if __name__ == "__main__":
    main()
