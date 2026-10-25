#!/usr/bin/env python3
"""用 vLLM 起一个 OpenAI 兼容服务，对 val 集打分，和微调后（bf16）用同一套指标。

GPTQ / AWQ 的 W4A16 权重都是 compressed-tensors 格式，vLLM 直接能加载：
    vllm serve output/quant/gptq-w4a16 --max-model-len 4096 ...
本脚本负责「起服务 → 逐条请求 → 算指标 → 存盘」，指标函数直接复用
scripts/eval_metrics.py，保证量化前后是同一把尺子、同一批样本。

用法：
  # 自己起服务（默认）
  /root/autodl-tmp/venv_quant/bin/python scripts/eval_vllm.py \
      --model output/quant/gptq-w4a16 --n 150 --out output/eval_quant/gptq

  # 已经有服务在跑（--no-serve + --port）
  ... scripts/eval_vllm.py --model gptq-w4a16 --no-serve --port 8100 --out ...
"""
from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parent.parent


def load_metrics_module():
    spec = importlib.util.spec_from_file_location("em", REPO / "scripts" / "eval_metrics.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def wait_health(port: int, timeout: float = 900) -> bool:
    """vLLM 启动要几十秒（加载权重 + torch.compile + CUDA graph），给足 15 分钟。"""
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = requests.get(f"http://127.0.0.1:{port}/health", timeout=3)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(3)
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="量化后的模型目录（或 --no-serve 时的 served name）")
    ap.add_argument("--served-name", default="qwen3vl-medpix")
    ap.add_argument("--data", default=str(REPO / "data/medpix/val.json"))
    ap.add_argument("--images", default=str(REPO / "data/medpix"))
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--tokenizer", default=None,
                    help="单独指定 tokenizer 目录（合并/量化目录的 tokenizer_config 可能带 "
                         "list 形式的 extra_special_tokens，transformers 4.57 会报 AttributeError）")
    ap.add_argument("--no-serve", action="store_true", help="连已有服务，不自己起")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    em = load_metrics_module()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 与 eval_metrics.py / eval150 完全一致的抽样（同种子、同 sample()），保证是同一批样本
    data = json.loads(Path(args.data).read_text())
    samples = random.Random(args.seed).sample(data, min(args.n, len(data)))

    proc = None
    if not args.no_serve:
        log = open(out / "server.log", "w")
        cmd = [
            str(Path(sys.executable).parent / "vllm"), "serve", str(args.model),
            "--served-model-name", args.served_name,
            "--host", "127.0.0.1", "--port", str(args.port),
            "--max-model-len", str(args.max_model_len),
            "--gpu-memory-utilization", str(args.gpu_util),
            "--limit-mm-per-prompt", '{"image": 1}',
            "--disable-log-requests",
        ]
        if args.tokenizer:
            cmd += ["--tokenizer", str(args.tokenizer)]
        print("[serve]", " ".join(cmd), flush=True)
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)

    try:
        if not wait_health(args.port):
            print("vLLM 起不来，看", out / "server.log")
            sys.exit(1)
        print("[serve] /health OK", flush=True)

        rows = []
        t0 = time.time()
        for i, s in enumerate(samples):
            img = (Path(args.images) / s["image"]).read_bytes()
            b64 = base64.b64encode(img).decode()
            q = s["conversations"][0]["value"].replace("<image>", "").strip()
            ref = s["conversations"][1]["value"].strip()
            payload = {
                "model": args.served_name,
                "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": q}]}],
                "max_tokens": args.max_new_tokens,
                "temperature": 0.0,
                "seed": args.seed,
            }
            t1 = time.time()
            r = requests.post(f"http://127.0.0.1:{args.port}/v1/chat/completions", json=payload, timeout=600)
            r.raise_for_status()
            dt = time.time() - t1
            pred = r.json()["choices"][0]["message"]["content"].strip()
            rows.append({"image": s["image"], "question": q, "ref": ref, "pred": pred,
                         "mode": s.get("mode", ""), "latency_s": round(dt, 3)})
            if (i + 1) % 10 == 0:
                el = time.time() - t0
                print(f"  {i+1}/{len(samples)}  {el:.0f}s  {el/(i+1):.2f}s/条", flush=True)
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
            log.close()

    pairs = [(r["pred"], r["ref"], r["mode"]) for r in rows]
    metrics = em.compute_metrics(pairs)
    lat = sorted(r["latency_s"] for r in rows)
    metrics.update({
        "model": str(args.model),
        "quant": "W4A16",
        "n": len(rows),
        "latency_mean_s": round(sum(lat) / len(lat), 3),
        "latency_median_s": round(lat[len(lat) // 2], 3),
    })
    (out / "predictions.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    em.plots(pairs, metrics, out)
    print(json.dumps(metrics, ensure_ascii=False, indent=2)[:800])
    print("结果写在", out)


if __name__ == "__main__":
    main()
