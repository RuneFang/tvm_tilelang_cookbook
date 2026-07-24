#!/usr/bin/env python3
"""
生成 star history SVG 曲线，数据源为 GitHub 官方 API，不依赖任何第三方服务。

用法：
    export GH_TOKEN=ghp_xxx  # 必须；否则 60 次/小时匿名限流很容易被打爆
    python3 tools/gen_star_history.py \
        --repo RuneFang/tvm_tilelang_cookbook \
        --out  assets/star-history.svg

原理：
    分页拉取 /repos/{owner}/{repo}/stargazers（Accept: star+json 才能拿到 starred_at 时间）
    然后按时间聚合，用纯字符串拼一张自适应尺寸的 SVG 折线图。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone, tzinfo
from typing import List, Tuple

try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except ImportError:  # pragma: no cover
    ZoneInfo = None  # type: ignore


API = "https://api.github.com"
UA = "star-history-generator (github.com actions)"


def gh_get(url: str, token: str | None) -> Tuple[dict, dict]:
    """请求 GitHub API，返回 (json_body, response_headers)。"""
    req = urllib.request.Request(url)
    req.add_header("Accept", "application/vnd.github.v3.star+json")
    req.add_header("User-Agent", UA)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            headers = dict(resp.headers)
            return body, headers
    except urllib.error.HTTPError as e:
        msg = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"GitHub API {e.code} for {url}: {msg}") from e


def parse_next_link(link_header: str) -> str | None:
    """解析 GitHub 的 Link 头，返回 rel=\"next\" 的 URL。"""
    if not link_header:
        return None
    for part in link_header.split(","):
        section = part.strip().split(";")
        if len(section) < 2:
            continue
        url = section[0].strip().lstrip("<").rstrip(">")
        rel = section[1].strip()
        if rel == 'rel="next"':
            return url
    return None


def fetch_all_stars(repo: str, token: str | None) -> List[datetime]:
    """拉取仓库的所有 starred_at 时间，按升序返回。"""
    url: str | None = f"{API}/repos/{repo}/stargazers?per_page=100"
    times: List[datetime] = []
    page = 0
    while url:
        page += 1
        body, headers = gh_get(url, token)
        if not isinstance(body, list):
            raise RuntimeError(f"unexpected response: {body!r}")
        for item in body:
            ts = item.get("starred_at")
            if ts:
                # 形如 "2024-05-01T12:34:56Z"
                times.append(datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc))
        remaining = headers.get("X-RateLimit-Remaining")
        print(f"[page {page:>3}] got {len(body)} records, rate remaining={remaining}", file=sys.stderr)
        url = parse_next_link(headers.get("Link", ""))
    times.sort()
    return times


def build_svg(
    repo: str,
    times: List[datetime],
    width: int = 800,
    height: int = 320,
    tz: tzinfo | None = None,
    tz_label: str = "UTC",
) -> str:
    """把时间序列画成一张自适应 SVG。

    Args:
        tz: 展示用的时区（不影响数据本身，只影响坐标轴刻度和右上角"updated"文案）；
            默认 None 表示 UTC。
        tz_label: 展示在右上角的时区简称，比如 "UTC+8"、"CST"。
    """
    if tz is None:
        tz = timezone.utc
    pad_l, pad_r, pad_t, pad_b = 60, 30, 40, 40
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    total = len(times)
    if total == 0:
        # 空数据：画一个占位卡片
        return _empty_svg(repo, width, height)

    # x: 时间；y: 累计 star 数（1..total）
    t_min = times[0]
    t_max = datetime.now(timezone.utc)
    if t_max <= t_min:
        t_max = t_min

    # y 轴上限：至少留一点顶部空间，避免曲线贴顶
    y_max = max(total, 4)

    def x_of(t: datetime) -> float:
        if t_max == t_min:
            return pad_l
        return pad_l + (t - t_min).total_seconds() / (t_max - t_min).total_seconds() * plot_w

    def y_of(n: int) -> float:
        return pad_t + plot_h - (n / y_max) * plot_h

    # 采样：如果 star 太多，降采样，不然 svg 太大
    step = max(1, total // 400)
    points: List[Tuple[float, float]] = []
    for i in range(0, total, step):
        points.append((x_of(times[i]), y_of(i + 1)))
    # 保证包含最后一个点
    if points[-1] != (x_of(times[-1]), y_of(total)):
        points.append((x_of(times[-1]), y_of(total)))
    # 加一个"从起点垂线到底"的锚，视觉上更完整
    path_d = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in points)

    # ---- x 轴：根据时间跨度自动选粒度，并去重相邻标签 ----
    span_days = max(1.0, (t_max - t_min).total_seconds() / 86400.0)
    if span_days <= 2:
        fmt = "%H:%M"                    # 几小时内
    elif span_days <= 60:
        fmt = "%Y-%m-%d"                 # 两个月内 → 精确到日
    elif span_days <= 365 * 2:
        fmt = "%Y-%m"                    # 两年内 → 精确到月
    else:
        fmt = "%Y"                       # 更长 → 精确到年

    n_ticks = 5
    raw_ticks: List[Tuple[float, str]] = []
    for i in range(n_ticks + 1):
        frac = i / n_ticks
        t = datetime.fromtimestamp(
            t_min.timestamp() + frac * (t_max.timestamp() - t_min.timestamp()),
            tz=tz,   # 用展示时区渲染，不然会一直是 UTC
        )
        raw_ticks.append((pad_l + frac * plot_w, t.strftime(fmt)))

    # 去掉相邻重复：连续几个刻度是同一个标签时只保留第一个
    x_ticks: List[Tuple[float, str]] = []
    last_label = None
    for x, label in raw_ticks:
        if label == last_label:
            x_ticks.append((x, ""))      # 保留刻度线位置，但不写文字
        else:
            x_ticks.append((x, label))
            last_label = label

    # ---- y 轴：整数刻度，避免出现 1.6 / 2.4 之类的小数 ----
    y_ticks: List[Tuple[float, str]] = []
    y_steps = min(5, y_max)
    seen_vals: set[int] = set()
    for i in range(y_steps + 1):
        val = int(round(y_max * i / y_steps))
        if val in seen_vals:
            continue
        seen_vals.add(val)
        y = pad_t + plot_h - (val / y_max) * plot_h
        y_ticks.append((y, str(val)))

    now_label = datetime.now(tz).strftime("%Y-%m-%d")

    svg_lines: List[str] = []
    svg_lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif">'
    )
    # 背景
    svg_lines.append(f'<rect width="{width}" height="{height}" fill="#ffffff"/>')
    # 左上角显示 star 数，右上角显示最后更新时间
    svg_lines.append(
        f'<text x="{pad_l}" y="24" font-size="14" fill="#24292f" font-weight="700">'
        f'{total} stars</text>'
    )
    svg_lines.append(
        f'<text x="{width - pad_r}" y="24" font-size="11" fill="#8c959f" text-anchor="end">'
        f'updated {now_label} ({tz_label})</text>'
    )
    # 网格 + y 轴刻度
    for y, label in y_ticks:
        svg_lines.append(
            f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width - pad_r}" y2="{y:.1f}" '
            f'stroke="#eaeef2" stroke-width="1"/>'
        )
        svg_lines.append(
            f'<text x="{pad_l - 8}" y="{y + 4:.1f}" font-size="11" fill="#57606a" text-anchor="end">{label}</text>'
        )
    # x 轴刻度
    for x, label in x_ticks:
        svg_lines.append(
            f'<text x="{x:.1f}" y="{height - pad_b + 20}" font-size="11" fill="#57606a" text-anchor="middle">{label}</text>'
        )
    # 面积填充（可选，让曲线更好看）
    if len(points) >= 2:
        area_d = (
            f"M {pad_l},{pad_t + plot_h} "
            + "L "
            + " L ".join(f"{x:.1f},{y:.1f}" for x, y in points)
            + f" L {points[-1][0]:.1f},{pad_t + plot_h} Z"
        )
        svg_lines.append(
            f'<path d="{area_d}" fill="#dbedff" fill-opacity="0.6"/>'
        )
    # 主曲线
    svg_lines.append(
        f'<path d="{path_d}" fill="none" stroke="#0969da" stroke-width="2" '
        f'stroke-linejoin="round" stroke-linecap="round"/>'
    )
    svg_lines.append("</svg>")
    return "\n".join(svg_lines)


def _empty_svg(repo: str, width: int, height: int) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
        f'<rect width="{width}" height="{height}" fill="#f6f8fa" stroke="#d0d7de" rx="8"/>'
        f'<text x="{width // 2}" y="{height // 2 - 6}" font-family="-apple-system,BlinkMacSystemFont,sans-serif" '
        f'font-size="16" fill="#57606a" text-anchor="middle">no stars yet for {repo}</text>'
        f'<text x="{width // 2}" y="{height // 2 + 16}" font-family="-apple-system,BlinkMacSystemFont,sans-serif" '
        f'font-size="12" fill="#8c959f" text-anchor="middle">be the first to ⭐</text>'
        f'</svg>'
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="owner/name, e.g. RuneFang/tvm_tilelang_cookbook")
    parser.add_argument("--out", required=True, help="output svg path")
    parser.add_argument(
        "--tz",
        default="Asia/Shanghai",
        help='IANA timezone for axis labels, e.g. "Asia/Shanghai" (default), "UTC", "America/New_York".',
    )
    parser.add_argument(
        "--tz-label",
        default=None,
        help='Short label shown next to the "updated" timestamp; auto-derived from --tz if omitted.',
    )
    args = parser.parse_args()

    # 解析时区
    if args.tz.upper() == "UTC":
        tz = timezone.utc
    else:
        if ZoneInfo is None:
            print("::error::Python 3.9+ with zoneinfo is required for --tz", file=sys.stderr)
            return 2
        try:
            tz = ZoneInfo(args.tz)
        except Exception as e:
            print(f"::error::unknown tz '{args.tz}': {e}", file=sys.stderr)
            return 2

    # 自动推导展示标签，比如 Asia/Shanghai → UTC+8
    if args.tz_label:
        tz_label = args.tz_label
    elif tz is timezone.utc:
        tz_label = "UTC"
    else:
        offset = datetime.now(tz).utcoffset() or timezone.utc.utcoffset(datetime.now())
        hours = int(offset.total_seconds() // 3600) if offset else 0
        tz_label = f"UTC{hours:+d}"

    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        print("warning: no GH_TOKEN/GITHUB_TOKEN, subject to 60/hour rate limit", file=sys.stderr)

    times = fetch_all_stars(args.repo, token)
    print(f"total stars fetched: {len(times)}", file=sys.stderr)

    svg = build_svg(args.repo, times, tz=tz, tz_label=tz_label)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(svg)
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
