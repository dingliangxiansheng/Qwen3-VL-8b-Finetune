#!/usr/bin/env python3
"""MedPix-VQA parquet → 本仓库要的 LLaVA/ShareGPT JSON（+ 图片落盘去重）。

为什么需要这一步：仓库的 SFT 数据集读的是 JSON 列表
    [{"image": "images/xx.png",
      "conversations": [{"from": "human", "value": "<image>\\n问题"},
                        {"from": "gpt",   "value": "答案"}]}]
而 /models/data 是 HF parquet，图片以二进制塞在 image_id 列里，所以要先落盘。

用法:
    python code/convert.py                       # 默认 /models/data → <仓库>/data/medpix
    python code/convert.py --limit 50            # 只转 50 条，冒烟测试
    python code/convert.py --out_dir /大盘/medpix # 磁盘紧张时换目录
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

import pyarrow.parquet as pq

# 仓库根目录 = 本文件的上一级（脚本放 src/ 或 code/ 都能算对），产物默认放仓库里
REPO = Path(__file__).resolve().parent.parent


def image_bytes(v):
    """HF 的 Image 列有三种存法：bytes / {"bytes":..,"path":None} / {"path": 文件路径}。"""
    if isinstance(v, (bytes, bytearray)):
        return bytes(v)
    if isinstance(v, dict):
        if v.get("bytes"):
            return v["bytes"]
        if v.get("path") and Path(v["path"]).is_file():
            return Path(v["path"]).read_bytes()
    return None


def ext_of(raw: bytes) -> str:
    """按文件头猜扩展名，别用 parquet 里写的格式（不可靠）。"""
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    return ".jpg" if raw[:3] == b"\xff\xd8\xff" else ".png"


def rows(files, limit):
    """流式读 parquet，避免一次性把 2.4GB 读进内存。"""
    n = 0
    for f in files:
        for batch in pq.ParquetFile(f).iter_batches(batch_size=32):
            for r in batch.to_pylist():
                yield r
                n += 1
                if limit and n >= limit:
                    return


def build(files, out_dir, seen, limit, split):
    """把一个 split 转成 records；seen = {图片 sha1: 相对路径}，跨 split 共享以去重。"""
    recs, st = [], dict(rows=0, kept=0, no_image=0, empty=0, new_img=0, reuse=0)
    for r in rows(files, limit):
        st["rows"] += 1
        raw = image_bytes(r.get("image_id"))
        q, a = (r.get("question") or "").strip(), (r.get("answer") or "").strip()
        if not raw:
            st["no_image"] += 1
            continue
        if not q or not a:
            st["empty"] += 1
            continue

        # 同一张图会有多条 QA（按内容 sha1 去重，磁盘省 ~30%）
        h = hashlib.sha1(raw).hexdigest()
        if h in seen:
            rel = seen[h]
            st["reuse"] += 1
        else:
            cid = re.sub(r"[^A-Za-z0-9_.-]", "_", str(r.get("case_id") or "case"))
            rel = f"images/{cid}_{h[:10]}{ext_of(raw)}"
            p = out_dir / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(raw)
            seen[h] = rel
            st["new_img"] += 1

        recs.append({
            "image": rel,  # 相对 out_dir，训练时用 --image_folder <out_dir>/images 拼
            "conversations": [
                # <image> 会被 sft_dataset.replace_image_tokens() 换成
                # <|vision_start|><|image_pad|><|vision_end|>，并吞掉这个换行
                {"from": "human", "value": f"<image>\n{q}"},
                {"from": "gpt", "value": a},
            ],
            # 下面三个字段仓库读取时会忽略，留着方便你排查和按 mode 分组评估
            "case_id": r.get("case_id"), "mode": r.get("mode"), "split": split,
        })
        st["kept"] += 1
    return recs, st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/models/data")
    ap.add_argument("--out_dir", default=f"{REPO}/data/medpix")
    ap.add_argument("--limit", type=int, default=0, help="每个 split 只转前 N 条（0=全部）")
    args = ap.parse_args()

    src, out = Path(args.data_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_files = sorted(src.glob("train-*.parquet"))
    val_files = sorted(src.glob("validation-*.parquet"))
    assert train_files, f"{src} 下没有 train-*.parquet"

    seen = {}  # 全局去重表：train/val 共用
    tr, tr_st = build(train_files, out, seen, args.limit, "train")
    (out / "train.json").write_text(json.dumps(tr, ensure_ascii=False), "utf-8")
    va, va_st = build(val_files, out, seen, args.limit, "val")
    (out / "val.json").write_text(json.dumps(va, ensure_ascii=False), "utf-8")

    for name, st in (("train", tr_st), ("val", va_st)):
        print(f"[{name}] rows={st['rows']} kept={st['kept']} no_image={st['no_image']} "
              f"empty={st['empty']} new_img={st['new_img']} reuse={st['reuse']}")

    # 自检：抽查图片是否真的落盘、train/val 有没有泄漏同一张图
    missing = sum(1 for r in (tr[:100] + va[:100]) if not (out / r["image"]).is_file())
    overlap = len({r["image"] for r in va} & {r["image"] for r in tr})
    print(f"[check] 抽查缺图 {missing} 张；train/val 重叠 {overlap} 张")
    print(f"[done] {out}/train.json  {out}/val.json  {out}/images")


if __name__ == "__main__":
    main()
