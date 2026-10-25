#!/usr/bin/env python3
"""从 trainer_state.json 画训练曲线（纯 CPU，训练中途也能跑）。

输出到 <output_dir>/plots/：01_loss / 02_lr / 03_grad_norm / 04_epoch_loss /
05_eval_loss / 06_overview + log_history.csv + summary.txt

用法:
    python src/plot_train.py --output-dir output/qwen3vl8b_medpix_lora
    python src/plot_train.py --output-dir output/.../checkpoint-150   # 直接指向某个 checkpoint

找日志的顺序：主目录的 trainer_state.json → 最新 checkpoint-*/trainer_state.json。
（checkpoint 是 save_steps 一步一个，训练中途没有主目录文件也能画。）
"""
import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

# matplotlib 默认往 ~/.config 写缓存，容器里常没权限 → 指到 /tmp
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplcache")
os.makedirs("/tmp/mplcache", exist_ok=True)
import matplotlib
matplotlib.use("Agg")          # 无显示环境，必须用 Agg 后端
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 130, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 10})


def ckpt_step(p: Path) -> int:
    m = re.search(r"checkpoint-(\d+)", p.name)
    return int(m.group(1)) if m else -1


def training_alive() -> bool:
    """看 /proc 里有没有 train_sft.py 在跑（只看不改，用来给提示）。"""
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            cmd = (d / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "ignore")
        except OSError:
            continue
        if "train_sft.py" in cmd:
            return True
    return False


def find_state(root: Path):
    """返回 (trainer_state 路径, 来源说明)。"""
    main = root / "trainer_state.json"
    if main.exists():
        return main, "主目录"
    # 训练中途：主目录还没写，退到最新的 checkpoint
    ckpts = sorted([p for p in root.glob("checkpoint-*") if (p / "trainer_state.json").exists()],
                   key=ckpt_step)
    if ckpts:
        return ckpts[-1] / "trainer_state.json", f"最新 checkpoint {ckpts[-1].name}"
    # 兜底：目录里任何地方（比如 output_dir 指到了上一层）
    any_state = sorted(root.rglob("trainer_state.json"), key=lambda p: p.stat().st_mtime)
    if any_state:
        return any_state[-1], "递归搜到"
    return None, ""


def explain_missing(root: Path):
    """没日志时给一屏可执行的排查提示，而不是干巴巴一句报错。"""
    print(f"[error] 在 {root} 下没找到 trainer_state.json\n")
    if not root.exists():
        print("  · 这个目录还不存在 —— 训练脚本还没跑过，或者 --output_dir 写错了。")
    else:
        items = sorted(p.name for p in root.iterdir())[:10]
        print(f"  · 目录现在是空的/只有这些：{items or '(空)'}")
    print("\n  常见原因，按顺序对一下：")
    print("  1) 训练还没启动：先跑 `bash src/train_lora.sh`；")
    print("  2) 训练刚启动不到第一个存档点：日志在 checkpoint-*/ 里，通常 50~200 步后出现；")
    print("  3) 训练启动就崩了：把训练日志往上翻，找 Traceback / OOM；")
    print("  4) 训练用了别的输出目录：`find . -name trainer_state.json` 看一下真实位置。")
    if training_alive():
        print("\n  → 检测到 train_sft.py 进程在跑，等存档点出来再画（或先看 tensorboard）。")


def ema(v, a=0.9):
    """指数滑动平均，让 loss 曲线好看一点。"""
    out, p = [], None
    for x in v:
        p = x if p is None else a * p + (1 - a) * x
        out.append(p)
    return out


def series(hist, key):
    """从 log_history 里抽 (step, value) 序列，缺的 key 直接跳过。"""
    r = [(h["step"], h[key]) for h in hist if h.get(key) is not None]
    return [x[0] for x in r], [x[1] for x in r]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True, help="训练输出目录（或某个 checkpoint 目录）")
    ap.add_argument("--from-checkpoint", action="store_true", help="（已自动，保留兼容）")
    ap.add_argument("--smooth", type=float, default=0.9)
    args = ap.parse_args()

    out = Path(args.output_dir).expanduser().resolve()
    state_path, source = find_state(out)
    if state_path is None:
        explain_missing(out)
        sys.exit(1)
    print(f"[info] 读 {state_path}  ({source})")
    state = json.loads(state_path.read_text())
    hist = state.get("log_history", [])
    if not hist:
        print(f"[error] {state_path} 里 log_history 是空的，训练可能还没写日志")
        sys.exit(1)

    pdir = out / "plots"          # 图始终放在 --output-dir/plots，方便找
    pdir.mkdir(parents=True, exist_ok=True)

    steps, loss = series(hist, "loss")
    s_lr, lr = series(hist, "learning_rate")
    s_gn, gn = series(hist, "grad_norm")
    s_ev, ev = series(hist, "eval_loss")

    # 顺手导出一份 csv，方便自己丢进 excel / pandas
    keys = list(dict.fromkeys(k for h in hist for k in h))
    with open(pdir / "log_history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(hist)

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(pdir / name)
        plt.close(fig)
        print("  ", name)

    # 1) loss：原始 + 平滑
    if loss:
        fig, ax = plt.subplots(figsize=(7.5, 4))
        ax.plot(steps, loss, lw=0.9, color="#9ecae1", label="raw")
        ax.plot(steps, ema(loss, args.smooth), lw=1.9, color="#08519c", label=f"EMA {args.smooth}")
        ax.set(xlabel="step", ylabel="loss",
               title=f"Train loss ({len(loss)} pts, epoch {state.get('epoch', 0):.2f})")
        ax.legend(frameon=False)
        save(fig, "01_loss.png")

    # 2) 学习率
    if lr:
        fig, ax = plt.subplots(figsize=(7.5, 3))
        ax.plot(s_lr, lr, color="#31a354", lw=1.5)
        ax.set(xlabel="step", ylabel="lr", title="Learning rate")
        save(fig, "02_lr.png")

    # 3) 梯度范数（突然飙升 = lr 太大 / 该 batch 数据有问题）
    if gn:
        fig, ax = plt.subplots(figsize=(7.5, 3))
        ax.plot(s_gn, gn, color="#e6550d", lw=1)
        ax.set(xlabel="step", ylabel="grad norm", title="Gradient norm")
        save(fig, "03_grad_norm.png")

    # 4) 每个 epoch 的平均 loss
    by_ep = {}
    for h in hist:
        if h.get("loss") is not None and h.get("epoch") is not None:
            by_ep.setdefault(round(h["epoch"], 2), []).append(h["loss"])
    if by_ep:
        eps = sorted(by_ep)
        means = [sum(by_ep[e]) / len(by_ep[e]) for e in eps]
        fig, ax = plt.subplots(figsize=(4.6, 3.4))
        bars = ax.bar([str(e) for e in eps], means, color="#4292c6", width=0.55)
        for b, m in zip(bars, means):
            ax.text(b.get_x() + b.get_width() / 2, m, f"{m:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set(xlabel="epoch", ylabel="mean loss", title="Epoch mean loss")
        save(fig, "04_epoch_loss.png")

    # 5) 验证集 loss（抬头说明开始过拟合）
    if ev:
        fig, ax = plt.subplots(figsize=(5.4, 3.4))
        ax.plot(s_ev, ev, "o-", color="#756bb1", lw=1.8, ms=5)
        ax.set(xlabel="step", ylabel="eval loss", title="Validation loss")
        save(fig, "05_eval_loss.png")

    # 6) 四联总览（贴报告用这张）
    panels = [(steps, loss, "train loss", "#08519c"), (s_lr, lr, "learning rate", "#31a354"),
              (s_gn, gn, "grad norm", "#e6550d"), (s_ev, ev, "eval loss", "#756bb1")]
    panels = [p for p in panels if p[1]]
    if panels:
        fig, axes = plt.subplots(2, 2, figsize=(11, 6))
        for ax, (xs, ys, t, c) in zip(axes.ravel(), panels):
            ax.plot(xs, ys, color=c, lw=1.2)
            ax.set(title=t, xlabel="step")
        for ax in axes.ravel()[len(panels):]:
            ax.axis("off")
        fig.suptitle(f"Training overview (max_steps={state.get('max_steps')})")
        save(fig, "06_overview.png")

    total = state.get("max_steps") or 0
    done = max(steps) if steps else 0
    txt = [f"max_steps={total} done={done} ({done / total:.1%})" if total else f"done={done}",
           f"epoch={state.get('epoch')} logged={len(loss)}"]
    if loss:
        txt.append(f"loss {loss[0]:.4f} -> {loss[-1]:.4f} (min {min(loss):.4f})")
    if ev:
        txt.append("eval_loss=" + ", ".join(f"{x:.4f}" for x in ev))
    (pdir / "summary.txt").write_text("\n".join(txt) + "\n")
    print("\n".join(txt))
    print(f"[ok] -> {pdir}")


if __name__ == "__main__":
    main()
