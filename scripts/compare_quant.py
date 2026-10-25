#!/usr/bin/env python3
"""把 bf16 / GPTQ / AWQ 三份 metrics.json 汇总成一张 markdown 对比表。

三份结果都由 scripts/eval_vllm.py 在同一批 150 条验证样本（seed=0）上跑出，
所以除了「体积 / 速度 / 精度」之外不需要再做任何归一化。

用法：
  python scripts/compare_quant.py \
      --runs bf16=output/eval_quant/bf16/metrics.json \
             gptq-w4a16=output/eval_quant/gptq/metrics.json \
             awq-w4a16=output/eval_quant/awq/metrics.json \
      --out output/eval_quant
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="*", default=[], help="名字=GiB，例如 bf16=17.53 gptq-w4a16=6.73")
    ap.add_argument("--fig", default=None, help="输出图路径（不给则只写 table.md）")
    ap.add_argument("--runs", nargs="+", required=True, help="名字=metrics.json 路径")
    ap.add_argument("--out", required=True, help="输出目录（写 table.md）")
    args = ap.parse_args()

    rows = []
    for item in args.runs:
        name, _, path = item.partition("=")
        m = load(Path(path))
        rows.append({
            "name": name,
            "f1": m.get("token_f1"),
            "rouge": m.get("rouge_l"),
            "bleu1": m.get("bleu_1"),
            "bleu4": m.get("bleu_4"),
            "em": m.get("exact_match"),
            "pred_len": m.get("pred_len_mean"),
            "lat": m.get("latency_median_s"),
            "size": next((float(kv.split("=", 1)[1]) for kv in args.sizes
                          if kv.split("=", 1)[0] == name), None),
        })

    base = next((r for r in rows if r["name"].startswith("bf16")), rows[0])
    # 单条耗时用 median：mean 会被前几条预热（模型首次编译/CUDA graph 捕获）抬高
    head = "| 模型 | Token-F1 | ΔF1 | ROUGE-L | BLEU-1 | BLEU-4 | 输出长度 | 单条耗时(median) | 相对速度 |\n"
    sep = "|---|---|---|---|---|---|---|---|---|\n"
    lines = [head, sep]
    for r in rows:
        d = (r["f1"] - base["f1"]) if r["f1"] is not None and base["f1"] is not None else 0.0
        speed = (base["lat"] / r["lat"]) if r["lat"] and base["lat"] else 1.0
        lines.append(
            f"| {r['name']} | {r['f1']:.4f} | {d:+.4f} | {r['rouge']:.4f} | {r['bleu1']:.4f} | "
            f"{r['bleu4']:.4f} | {r['pred_len']:.1f} | {r['lat']:.2f} s | {speed:.2f}× |\n"
        )
    table = "".join(lines)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "table.md").write_text(table)
    print(table)

    if args.fig:
        plot(rows, Path(args.fig))


def plot(rows: list[dict], path: Path) -> None:
    """三面板：精度 / 体积 / 单条耗时（median）。图中文字用英文，机器无中文字体。"""
    import os
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplcache")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [r["name"] for r in rows]
    colors = ["#6e6e6e"] + ["#2171b5", "#d94801", "#41ab5d", "#756bb1"][: len(names) - 1]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))

    ax = axes[0]
    ax.bar(names, [r["f1"] for r in rows], color=colors, width=0.62)
    for i, r in enumerate(rows):
        ax.text(i, r["f1"] + 0.004, f'{r["f1"]:.4f}', ha="center", fontsize=7.5)
    ax.set_ylim(min(r["f1"] for r in rows) - 0.02, max(r["f1"] for r in rows) + 0.02)
    ax.set_ylabel("Token-F1 (n=150)")
    ax.set_title("Accuracy")
    ax.tick_params(axis="x", labelrotation=15, labelsize=8)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    sizes = [r["size"] if r["size"] else 0 for r in rows]
    ax.bar(names, sizes, color=colors, width=0.62)
    for i, s in enumerate(sizes):
        ax.text(i, s + 0.2, f"{s:.2f}", ha="center", fontsize=7.5)
    ax.set_ylabel("size (GiB)")
    ax.set_title("Size")
    ax.tick_params(axis="x", labelrotation=15, labelsize=8)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[2]
    lats = [r["lat"] or 0 for r in rows]
    ax.bar(names, lats, color=colors, width=0.62)
    for i, v in enumerate(lats):
        ax.text(i, v + 0.01, f"{v:.2f}s", ha="center", fontsize=7.5)
    ax.set_ylabel("latency per sample (s, median)")
    ax.set_title("Latency")
    ax.tick_params(axis="x", labelrotation=15, labelsize=8)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("GPTQ / AWQ vs bf16 (same 150 val samples)", fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    print("->", path)


if __name__ == "__main__":
    main()
