#!/usr/bin/env python3
"""配对比较两个 metrics 目录：同批样本逐条算指标差，给 bootstrap 95% 置信区间。

两份 predictions.jsonl 必须来自同一批样本、同一顺序（eval_vllm.py 用同一个
random.Random(seed).sample()，所以行序天然一致）。

用法：
  python scripts/paired_ci.py \
      --a output/eval_quant/bf16 --b output/eval_quant/gptq --out output/eval_quant
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def load_metrics_module():
    spec = importlib.util.spec_from_file_location("em", REPO / "scripts" / "eval_metrics.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def per_sample(em, rows: list[dict]) -> list[float]:
    return [em.tok_f1(r["pred"], r["ref"]) for r in rows]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="参照组目录（如 bf16）")
    ap.add_argument("--b", required=True, help="比较组目录（如 gptq）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=2000)
    args = ap.parse_args()

    em = load_metrics_module()
    a_rows = [json.loads(l) for l in (Path(args.a) / "predictions.jsonl").read_text().splitlines() if l.strip()]
    b_rows = [json.loads(l) for l in (Path(args.b) / "predictions.jsonl").read_text().splitlines() if l.strip()]
    assert len(a_rows) == len(b_rows), "两份预测的样本数不一致"
    assert [r["image"] for r in a_rows] == [r["image"] for r in b_rows], "两份预测的样本顺序不一致"

    fa, fb = per_sample(em, a_rows), per_sample(em, b_rows)
    diffs = [b - a for a, b in zip(fa, fb)]
    n = len(diffs)
    mean = sum(diffs) / n

    rng = random.Random(0)
    boots = []
    for _ in range(args.rounds):
        s = sum(diffs[rng.randrange(n)] for _ in range(n)) / n
        boots.append(s)
    boots.sort()
    lo, hi = boots[int(0.025 * args.rounds)], boots[int(0.975 * args.rounds) - 1]

    same = sum(1 for x, y in zip(a_rows, b_rows) if x["pred"].strip() == y["pred"].strip())
    out = {
        "n": n,
        "delta_token_f1": round(mean, 4),
        "ci95": [round(lo, 4), round(hi, 4)],
        "identical_outputs": f"{same}/{n}",
    }
    Path(args.out).mkdir(parents=True, exist_ok=True)
    name = f"paired_{Path(args.a).name}_vs_{Path(args.b).name}.json"
    (Path(args.out) / name).write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
