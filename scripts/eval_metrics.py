#!/usr/bin/env python3
"""验证集生成式评估：跑 N 条 → 算指标 → 出图。

指标：EM（归一化完全匹配）、Token-F1（SQuAD 式）、ROUGE-L、BLEU-1、BLEU-4。
输出：<out>/metrics.json、samples.jsonl、01_metrics.png、02_lengths.png、03_by_mode.png

用法（在仓库根目录）:
    PYTHONPATH=src python src/eval_metrics.py \
        --adapter output/qwen3vl8b_medpix_lora \
        --data data/medpix/val.json --images data/medpix/images --n 100 \
        --out output/qwen3vl8b_medpix_lora/eval
    python src/eval_metrics.py --selftest --out /tmp/eval_demo   # 不加载模型，只测指标+画图
"""
import argparse
import json
import os
import random
import string
import sys
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplcache")
os.makedirs("/tmp/mplcache", exist_ok=True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 130, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 10})

PUNCT = str.maketrans("", "", string.punctuation)
CKPT = "/models/weight/Qwen3-VL-8B-Instruct"


# ============================ 指标实现（纯 python，无需额外依赖）============================
def norm(s: str) -> str:
    """小写、去标点、压空格 —— 所有指标都基于这个归一化。"""
    return " ".join(s.lower().translate(PUNCT).split())


def tok_f1(pred: str, ref: str) -> float:
    """词级 F1，对长答案比 EM 更有区分度。"""
    p, r = norm(pred).split(), norm(ref).split()
    if not p or not r:
        return float(p == r)
    same = sum((Counter(p) & Counter(r)).values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(r)
    return 2 * prec * rec / (prec + rec)


def rouge_l(pred: str, ref: str) -> float:
    """ROUGE-L：最长公共子序列的 F1，衡量表达顺序/结构。"""
    a, b = norm(pred).split(), norm(ref).split()
    if not a or not b:
        return float(a == b)
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i, x in enumerate(a, 1):
        for j, y in enumerate(b, 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if x == y else max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[-1][-1]
    if lcs == 0:
        return 0.0
    prec, rec = lcs / len(a), lcs / len(b)
    return 2 * prec * rec / (prec + rec)


def bleu(pred: str, ref: str, max_n: int = 4) -> float:
    """单参考 BLEU，带 brevity penalty；加 eps 防止 log(0)。"""
    p, r = norm(pred).split(), norm(ref).split()
    if not p:
        return 0.0
    scores = []
    for n in range(1, max_n + 1):
        pc = Counter(tuple(p[i:i + n]) for i in range(len(p) - n + 1))
        rc = Counter(tuple(r[i:i + n]) for i in range(len(r) - n + 1))
        num = sum(min(c, rc[g]) for g, c in pc.items())
        scores.append((num + 1e-9) / max(1, len(p) - n + 1) if rc else 1e-9)
    bp = 1.0 if len(p) >= len(r) else 2.718281828 ** (1 - len(r) / max(1, len(p)))
    prod = 1.0
    for s in scores:
        prod *= s
    return bp * prod ** (1.0 / max_n)


def exact_match(pred: str, ref: str) -> float:
    return float(norm(pred) == norm(ref))


def compute_metrics(pairs):
    """pairs = [(pred, ref, mode), ...] -> dict（含按 mode 分组的分数）"""
    if not pairs:
        return {}
    n = len(pairs)
    out = {
        "n": n,
        "exact_match": sum(exact_match(p, r) for p, r, _ in pairs) / n,
        "token_f1": sum(tok_f1(p, r) for p, r, _ in pairs) / n,
        "rouge_l": sum(rouge_l(p, r) for p, r, _ in pairs) / n,
        "bleu_1": sum(bleu(p, r, 1) for p, r, _ in pairs) / n,
        "bleu_4": sum(bleu(p, r, 4) for p, r, _ in pairs) / n,
        "pred_len_mean": sum(len(norm(p).split()) for p, _, _ in pairs) / n,
        "ref_len_mean": sum(len(norm(r).split()) for _, r, _ in pairs) / n,
    }
    modes = {}
    for p, r, m in pairs:
        modes.setdefault(m or "all", []).append((p, r))
    if len(modes) > 1:
        out["by_mode"] = {m: {"n": len(v),
                              "token_f1": sum(tok_f1(p, r) for p, r in v) / len(v),
                              "rouge_l": sum(rouge_l(p, r) for p, r in v) / len(v)}
                          for m, v in modes.items()}
    return out


# ============================ 画图 ============================
def plots(pairs, metrics, out_dir: Path):
    made = []

    # 1) 五个指标柱状图
    names = [("exact_match", "EM"), ("token_f1", "Token-F1"), ("rouge_l", "ROUGE-L"),
             ("bleu_1", "BLEU-1"), ("bleu_4", "BLEU-4")]
    vals = [metrics[k] for k, _ in names]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    bars = ax.bar([n for _, n in names], vals, color="#4292c6", width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, max(vals) * 1.25 + 0.02)
    ax.set(ylabel="score", title=f"Validation metrics (n={metrics['n']})")
    fig.tight_layout(); fig.savefig(out_dir / "01_metrics.png"); plt.close(fig)
    made.append("01_metrics.png")

    # 2) 答案长度分布 + 长度对齐散点（看模型是不是话痨/答太短）
    rl = [len(norm(r).split()) for _, r, _ in pairs]
    pl = [len(norm(p).split()) for p, _, _ in pairs]
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
    ax[0].hist(rl, bins=30, alpha=0.6, label="reference", color="#31a354")
    ax[0].hist(pl, bins=30, alpha=0.6, label="prediction", color="#e6550d")
    ax[0].set(xlabel="answer length (tokens)", ylabel="count", title="Answer length distribution")
    ax[0].legend(frameon=False)
    ax[1].scatter(rl, pl, s=10, alpha=0.5, color="#08519c")
    lim = max(max(rl), max(pl)) * 1.05
    ax[1].plot([0, lim], [0, lim], "r--", lw=1, label="y = x")
    ax[1].set(xlabel="reference length", ylabel="prediction length", title="Length alignment")
    ax[1].legend(frameon=False)
    fig.tight_layout(); fig.savefig(out_dir / "02_lengths.png"); plt.close(fig)
    made.append("02_lengths.png")

    # 3) 按数据 mode 分组（caption 生成的 QA 通常比 alignment 的容易）
    if "by_mode" in metrics:
        modes = list(metrics["by_mode"])
        fig, ax = plt.subplots(figsize=(max(5.5, 1.2 * len(modes)), 3.6))
        w = 0.38
        for i, (k, lbl, c) in enumerate([("token_f1", "Token-F1", "#4292c6"),
                                        ("rouge_l", "ROUGE-L", "#756bb1")]):
            xs = [j + (i - 0.5) * w for j in range(len(modes))]
            ax.bar(xs, [metrics["by_mode"][m][k] for m in modes], width=w, label=lbl, color=c)
        ax.set_xticks(range(len(modes)))
        ax.set_xticklabels([f"{m}\n(n={metrics['by_mode'][m]['n']})" for m in modes], fontsize=8)
        ax.set(ylabel="score", title="Metrics by data mode")
        ax.legend(frameon=False)
        fig.tight_layout(); fig.savefig(out_dir / "03_by_mode.png"); plt.close(fig)
        made.append("03_by_mode.png")
    return made


# ============================ 推理 ============================
def find_code_root(start: Path) -> Path:
    """代码目录可能被改名/挪动（比如放到 wenjian/），认含 src/params.py 的那一层。"""
    cands = [start] + sorted(p for p in start.glob("*") if p.is_dir()) + [start.parent]
    for c in cands:
        if (c / "src" / "params.py").is_file() and (c / "src" / "train" / "train_sft.py").is_file():
            return c
    return start


def run_inference(args, samples):
    """用仓库自己的 load_pretrained_model 加载 adapter + 基座（它会顺带 merge 进内存）。"""
    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo / "src"))

    import torch
    from qwen_vl_utils import process_vision_info
    from utils import load_pretrained_model

    if args.base_only:                        # 「微调前」基线：只加载基座，不挂 adapter
        print(f"评估基座模型（不加载 LoRA）: {args.base}")
        processor, model = load_pretrained_model(
            model_path=args.base, model_base=None, model_name="base",
            device_map="cuda", load_4bit=args.load_4bit)
    else:
        processor, model = load_pretrained_model(
            model_path=args.adapter, model_base=args.base, model_name="medpix-lora",
            device_map="cuda", load_4bit=args.load_4bit)
    model.eval()

    rows = []
    t_start = time.time()
    for i, s in enumerate(samples):
        # JSON 里的 image 是相对 <data 目录> 的路径（如 images/xx.png）
        img = Path(args.images) / s["image"]
        if not img.exists():
            img = Path(args.data).parent / s["image"]
        q = s["conversations"][0]["value"].replace("<image>\n", "").replace("<image>", "").strip()

        messages = [{"role": "user", "content": [
            {"type": "image", "image": str(img),
             "min_pixels": args.min_pixels, "max_pixels": args.max_pixels},
            {"type": "text", "text": q}]}]
        text = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info([messages], image_patch_size=16)
        inputs = processor(text=[text], images=images, videos=videos, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        pred = processor.tokenizer.decode(
            gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        rows.append({"image": s["image"], "question": q,
                     "ref": s["conversations"][1]["value"], "pred": pred,
                     "mode": s.get("mode", "")})
        el = time.time() - t_start
        print(f"[{i + 1}/{len(samples)}] f1={tok_f1(pred, rows[-1]['ref']):.3f} "
              f"{el / (i + 1):.2f}s/条 pred={pred[:70]!r}")
        del inputs, gen
        torch.cuda.empty_cache()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", help="LoRA 输出目录（含 adapter_model.safetensors）")
    ap.add_argument("--base", default=CKPT, help="基座权重目录")
    ap.add_argument("--data", default="data/medpix/val.json")
    ap.add_argument("--images", default="data/medpix/images")
    ap.add_argument("--repo", default=None, help="代码根目录（含 src/params.py 的那层），默认自动找")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--min-pixels", type=int, default=128 * 32 * 32)   # 与训练一致
    ap.add_argument("--max-pixels", type=int, default=1024 * 32 * 32)
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--base-only", action="store_true", help="只评估基座模型（微调前基线）")
    ap.add_argument("--out", default="eval_out")
    ap.add_argument("--selftest", action="store_true", help="不加载模型，验证指标与画图")
    args = ap.parse_args()
    if args.repo is None:
        args.repo = str(find_code_root(Path(__file__).resolve().parent.parent))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 自测：用假预测验证指标/画图链路（不需要 GPU）----
    if args.selftest:
        rng = random.Random(args.seed)
        refs = ["the imaging shows a large right-sided pleural effusion with compressive atelectasis",
                "contrast enhanced ct reveals a hypodense lesion in the left hepatic lobe, 3.2 cm",
                "there is moderate degenerative change of the lumbar spine without fracture",
                "no evidence of intracranial hemorrhage or mass effect"]
        rows = []
        for i in range(120):
            ref = rng.choice(refs)
            words = ref.split()
            pred = " ".join(words[:rng.randint(max(1, len(words) - 5), len(words))])
            rows.append({"image": f"fake_{i}.png", "question": "What do the imaging results show?",
                         "ref": ref, "pred": pred, "mode": rng.choice(["from caption", "alignment"])})
        pairs = [(r["pred"], r["ref"], r["mode"]) for r in rows]
        metrics = compute_metrics(pairs)
        made = plots(pairs, metrics, out_dir)
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        print(json.dumps({k: round(v, 4) for k, v in metrics.items() if isinstance(v, float)}, indent=2))
        print(f"[selftest ok] {made} -> {out_dir}")
        return

    assert args.base_only or args.adapter, "--adapter 必填（LoRA 输出目录；基线评测用 --base-only）"
    data = json.loads(Path(args.data).read_text())
    samples = random.Random(args.seed).sample(data, min(args.n, len(data)))  # 固定种子，可复现
    rows = run_inference(args, samples)

    pairs = [(r["pred"], r["ref"], r["mode"]) for r in rows]
    metrics = compute_metrics(pairs)
    metrics.update({"dataset": str(args.data),
                    "model": "base (no LoRA)" if args.base_only else str(args.adapter)})
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    with open(out_dir / "samples.jsonl", "w") as f:      # 逐条样本，方便肉眼挑错
        for r in rows:
            f.write(json.dumps({**r, "f1": tok_f1(r["pred"], r["ref"])}, ensure_ascii=False) + "\n")

    made = plots(pairs, metrics, out_dir)
    print("\n" + json.dumps({k: round(v, 4) for k, v in metrics.items() if isinstance(v, float)}, indent=2))
    print(f"[ok] {out_dir} -> {made}")


if __name__ == "__main__":
    main()
