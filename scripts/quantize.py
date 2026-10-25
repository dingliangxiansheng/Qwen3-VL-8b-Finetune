#!/usr/bin/env python3
"""GPTQ / AWQ 权重量化（W4A16，compressed-tensors 格式，vLLM 可直接 serve）。

这里只做「权重 4bit、激活 bf16」的训练后量化（PTQ），两种算法思路不同：
    GPTQ：逐层用 Hessian 逆矩阵补偿量化误差（二阶信息，闭式解）
    AWQ ：先看激活找出 ~1% 的重要通道，对这些通道做等价缩放再量化（一阶信息）

产物是 compressed-tensors 的 W4A16 checkpoint（4bit、group_size=128、对称），
`vllm serve <输出目录>` 可以直接加载，精度/速度评测走 scripts/eval_vllm.py。

输入模型是「LoRA 已合并」的完整 HF checkpoint（scripts/quantize.sh 里负责合并）：
    python scripts/quantize.py --algo gptq --model /dev/shm/hf_merged --out output/quant/gptq-w4a16
    python scripts/quantize.py --algo awq  --model /dev/shm/hf_merged --out output/quant/awq-w4a16

校准数据只用训练集（默认 256 条，seed 固定），不碰验证集，避免评测泄漏。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

# llmcompressor 0.9 的 API（0.12+ 的 API 不同，别混用）
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.modifiers.awq.mappings import AWQMapping
from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier
from llmcompressor import oneshot

# 视觉塔只占 0.42B，量化收益小、掉点风险大；lm_head 在 vLLM 里本来就按 bf16 跑。
# 这两条也是社区 VLM AWQ/GPTQ 权重的通行做法。
IGNORE = ["lm_head", "re:.*visual.*"]


class CalibDataset:
    """惰性预处理校准样本：只在被取到时才解码图片、跑 processor。

    llmcompressor 会用 len() 和 [i] 访问它，并且要求它带
    `column_names = ["input_ids"]`（这样它才跳过自己的 tokenize 流程）。
    返回的张量不带 batch 维，真正的 batched 化交给 passthrough collate。
    """

    column_names = ["input_ids"]

    def __init__(self, records: list[dict], images_root: Path, processor, max_len: int):
        self.records = records
        self.images_root = images_root
        self.processor = processor
        self.max_len = max_len
        self._cache: dict[int, dict] = {}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx):
        # llmcompressor 的 LengthAwareSampler 会做 dataset["input_ids"] 取整列长度，
        # 所以这里同时支持「按列取」和「按行取」两种访问方式。
        if isinstance(idx, str):
            return [self[i][idx] for i in range(len(self))]
        if idx not in self._cache:  # 采样器会二次访问（取 token 长度），必须缓存
            self._cache[idx] = self._prepare(idx)
        return self._cache[idx]

    def _prepare(self, idx: int) -> dict:
        rec = self.records[idx]
        image = Image.open(self.images_root / rec["image"]).convert("RGB")
        question = rec["conversations"][0]["value"].replace("<image>", "").strip()
        messages = [{
            "role": "user",
            "content": [{"type": "image", "image": image}, {"type": "text", "text": question}],
        }]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text], images=[image], return_tensors="pt",
            truncation=True, max_length=self.max_len,
        )
        return {
            "input_ids": inputs["input_ids"][0],
            "attention_mask": inputs["attention_mask"][0],
            "pixel_values": inputs["pixel_values"],          # [num_patches, dim]，2 维
            "image_grid_thw": inputs["image_grid_thw"],      # [1, 3]
        }


def passthrough_collate(batch: list[dict]) -> dict:
    """batch=1：输入图像尺寸不同，pixel_values 形状也就不同，只能按 patch 维拼接。"""
    out = {
        "input_ids": torch.stack([b["input_ids"] for b in batch], dim=0),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch], dim=0),
        "pixel_values": torch.cat([b["pixel_values"] for b in batch], dim=0),
        "image_grid_thw": torch.cat([b["image_grid_thw"] for b in batch], dim=0),
    }
    return {k: v.cuda() for k, v in out.items()}


def build_calib_dataset(args, processor) -> CalibDataset:
    data = json.loads(Path(args.data).read_text())
    n = min(args.samples, len(data))
    records = random.Random(args.seed).sample(data, n)
    print(f"[calib] 训练集 {len(data)} 条 → 取 {n} 条（seed={args.seed}），图片根 {args.images}")
    return CalibDataset(records, Path(args.images), processor, args.max_len)


def build_recipe(algo: str, duo: bool = True):
    if algo == "awq-rtn":
        # AWQ 的平滑由本脚本自己完成（见 awq_smooth），量化本身用 RTN：
        # AWQ 原论文的量化步骤就是「平滑后按 group 取 min/max 做 RTN」，
        # 而 llmcompressor 0.9 的 AWQModifier 在前向中做网格搜索，
        # 会污染权重观测器的 min/max（实测 gate/up 的 scale 被放大 47~203 倍，
        # 整层被量化成 0），因此这里把两步拆开自己做。
        return [QuantizationModifier(targets="Linear", scheme="W4A16", ignore=IGNORE)]
    if algo == "gptq":
        # block_size=128：每次处理 128 行权重，显存与精度的折中
        # dampening_frac=0.01：Hessian 加阻尼，避免病态矩阵导致的数值爆炸
        return [GPTQModifier(
            targets="Linear", scheme="W4A16", ignore=IGNORE,
            block_size=128, dampening_frac=0.01,
        )]
    # AWQModifier 自己就带量化（QuantizationMixin），不需要再挂 QuantizationModifier。
    #
    # 这里必须显式给 mappings，不能用 llmcompressor 的默认映射。原因：
    # 默认映射里有一条 `v_proj → o_proj`，用于在把 input_layernorm 的输出按通道
    # 缩放 1/s 之后，把 v_proj 的这一份缩放补偿到 o_proj 的输入上。
    # 但 Qwen3-VL 是 GQA（q 24 头 / kv 4 头）：v_proj 输出 4×256=1024 维，
    # 而 o_proj 输入是 24×256=6144 维，两者形状对不上，llmcompressor 会
    # `skipping AWQ for ... v_proj ... because found incompatible balance layers`，
    # 只跳过补偿、却不撤销前面的缩放 —— 结果 v 的输出被放大而 o_proj 没有补偿，
    # 注意力输出全错，模型直接输出乱码（实测 Token-F1 从 0.34 掉到 0.003）。
    #
    # 所以注意力这一段整体不平滑，只做 MLP 的两条映射（这两条两端形状都一致）：
    #   post_attention_layernorm → gate_proj / up_proj
    #   up_proj → down_proj
    # 代价是 AWQ 的收益少一部分，但结果在数值上等价、模型可用。
    print("  [!] 注意：llmcompressor 0.9 的 AWQ 在本模型上会产出不可用权重"
          "（GQA 的 v_proj→o_proj 补偿被跳过、观测器被网格搜索污染），"
          "部署请用 --algo awq-rtn；此处仅保留用于对照实验。")
    mappings = [
        AWQMapping("re:.*post_attention_layernorm$", ["re:.*gate_proj$", "re:.*up_proj$"]),
        AWQMapping("re:.*up_proj$", ["re:.*down_proj$"]),
    ]
    return [AWQModifier(targets="Linear", scheme="W4A16", ignore=IGNORE,
                        mappings=mappings, duo_scaling=duo)]


@torch.no_grad()
def awq_smooth(model, dataset, n_samples: int = 32, alpha: float = 0.5) -> None:
    """AWQ 通道平滑：数学上等价的变换，把量化误差挪到更容易压的位置。

    对每个 decoder 层（`<layer>.mlp` 与同层的 `post_attention_layernorm`）：

        post_attention_layernorm.weight *= s          # 激活按通道放大 s
        gate_proj.weight /= s, up_proj.weight /= s    # 权重按输入通道缩小 s

    缩放系数取输入激活与权重幅度比的 alpha 次幂：s = (mean|W| / mean|X|)^alpha，
    之后归一化到均值 1，避免整体幅度漂移。
    """
    n = min(n_samples, len(dataset))
    print(f"[awq] 统计激活幅度（{n} 条样本）")

    stats: dict[str, list] = {}  # name -> [sum(|x|) per channel, 计数]
    hooks = []
    for name, module in model.named_modules():
        if not name.endswith(("mlp.gate_proj", "mlp.up_proj")):
            continue
        stats[name] = [torch.zeros(module.in_features, dtype=torch.float64, device="cuda"), 0]

        def hook(mod, args, name=name):
            x = args[0].detach().abs()
            stats[name][0] += x.reshape(-1, x.shape[-1]).sum(0).double()
            stats[name][1] += x.numel() // x.shape[-1]

        hooks.append(module.register_forward_pre_hook(hook))

    try:
        for i in range(n):
            # input_ids/attention_mask 补上 batch 维；pixel_values 与 image_grid_thw
            # 由 processor 直接给出（前者是 [num_patch, dim]，后者是 [1, 3]），不能再 unsqueeze
            batch = {}
            for k, v in dataset[i].items():
                v = v.to("cuda")
                batch[k] = v.unsqueeze(0) if k in ("input_ids", "attention_mask") else v
            model(**batch)
            if (i + 1) % 8 == 0:
                print(f"  {i + 1}/{n}")
    finally:
        for h in hooks:
            h.remove()

    print("[awq] 逐层施加平滑")
    modules = dict(model.named_modules())
    applied = 0
    for name, mlp in modules.items():
        if not name.endswith(".mlp") or not hasattr(mlp, "gate_proj"):
            continue
        prefix = name[: -len(".mlp")]
        norm = modules.get(f"{prefix}.post_attention_layernorm")
        g_key, u_key = f"{name}.gate_proj", f"{name}.up_proj"
        if norm is None or g_key not in stats or stats[g_key][1] == 0:
            continue

        mean_x = (stats[g_key][0] / stats[g_key][1]) + (stats[u_key][0] / stats[u_key][1])
        mean_x = (mean_x / 2).float()
        mean_w = (mlp.gate_proj.weight.abs().float().mean(0)
                  + mlp.up_proj.weight.abs().float().mean(0)) / 2
        s = (mean_w.clamp_min(1e-8) / mean_x.clamp_min(1e-8)).pow(alpha).clamp(0.01, 100.0)
        s = s / s.mean()

        s_cast = s.to(norm.weight.dtype)
        norm.weight.mul_(s_cast)
        mlp.gate_proj.weight.div_(s_cast.unsqueeze(0))
        mlp.up_proj.weight.div_(s_cast.unsqueeze(0))
        applied += 1

    print(f"[awq] 平滑完成，共 {applied} 层")


def dir_gib(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1024**3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["gptq", "awq", "awq-rtn"], required=True)
    ap.add_argument("--model", required=True, help="已合并 LoRA 的 HF checkpoint 目录")
    ap.add_argument("--processor", default=None,
                    help="processor/tokenizer 目录（默认用 --model；合并目录的 tokenizer_config "
                         "里 extra_special_tokens 是 list，transformers 4.57 会报错，故常传基座目录）")
    ap.add_argument("--out", required=True, help="量化产物输出目录")
    ap.add_argument("--data", default="data/medpix/train.json", help="LLaVA 格式训练集（做校准）")
    ap.add_argument("--images", default="data/medpix", help="image 字段的相对路径根目录")
    ap.add_argument("--duo-scaling", default="true", choices=["true", "false"],
                    help="AWQ 的缩放通道选择：true 同时用激活与权重幅度，false 只用权重幅度。"
                         "实测 true 时本模型的 gate/up 观测范围被网格搜索放大，量化后整层塌成 0")
    ap.add_argument("--samples", type=int, default=256, help="校准样本数")
    ap.add_argument("--max-len", type=int, default=2048, help="校准样本的最大 token 数")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t_load = time.time()
    print(f"[load] {args.model}（bf16 → cuda）")
    proc_dir = args.processor or args.model
    processor = AutoProcessor.from_pretrained(proc_dir)
    config = AutoConfig.from_pretrained(args.model)
    # LoRA 合并出来的 config.json 里 text_config.rope_scaling 是 null，
    # transformers 4.57 的 qwen3_vl 会在 `rope_scaling.get(...)` 处直接 AttributeError，
    # 所以这里从基座补回来（mrope_section 随模型而定，不能写死成别的值）。
    if getattr(config.text_config, "rope_scaling", None) is None:
        fallback = {"mrope_interleaved": True, "mrope_section": [24, 20, 20], "rope_type": "default"}
        base_cfg = Path(proc_dir) / "config.json"
        if base_cfg.exists():
            fallback = json.loads(base_cfg.read_text()).get("text_config", {}).get("rope_scaling") or fallback
        config.text_config.rope_scaling = fallback
        print(f"[config] rope_scaling 缺失，按基座补回 {fallback}")
    # 先落在 CPU 再整体搬到 GPU：transformers 4.57 里 device_map="cuda" 会让
    # 视觉塔的 pos_embed 停在 meta device，前向时抛
    # `NotImplementedError: Cannot copy out of meta tensor; no data!`
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, config=config, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    model.to("cuda")
    model.eval()
    print(f"[load] 完成 {time.time() - t_load:.1f}s，显存 {torch.cuda.memory_allocated() / 1024**3:.1f} GiB")

    dataset = build_calib_dataset(args, processor)
    if args.algo == "awq-rtn":
        awq_smooth(model, dataset, n_samples=min(32, args.samples))

    recipe = build_recipe(args.algo, duo=(args.duo_scaling == "true"))

    # llmcompressor 的顺序流水线默认把权重 offload 到 CPU、用 meta 张量占位，
    # 再按子图逐个 onload。Qwen3-VL 的视觉塔是靠 get_image_features() 调用的，
    # 不会被 trace 进任何子图，于是永远停在 meta 上，前向时报
    # `Cannot copy out of meta tensor; no data!`。我们的模型 17GB 能整块放进 32GB 显存，
    # 不需要 offload，所以直接把这一步屏蔽掉。
    import llmcompressor.pipelines.sequential.helpers as _seq_helpers
    _seq_helpers.offloaded_dispatch = (
        lambda module, execution_device, offload_device=None: module
    )

    print(f"[oneshot] algo={args.algo} recipe={[type(m).__name__ for m in recipe]}")
    t0 = time.time()
    oneshot(
        model=model,
        processor=processor,
        dataset=dataset,
        splits="calibration",        # 0.9 必须显式指定，否则找不到 calibration 分割
        data_collator=passthrough_collate,
        batch_size=1,
        num_calibration_samples=len(dataset),
        max_seq_length=args.max_len,
        recipe=recipe,
        output_dir=str(out),
    )
    elapsed = time.time() - t0
    del model
    torch.cuda.empty_cache()
    print(f"[oneshot] 完成 {elapsed:.1f}s")

    size = dir_gib(out)
    print(f"[save] {out} 共 {size:.2f} GiB")
    (out / "quant_report.json").write_text(json.dumps({
        "algo": args.algo,
        "scheme": "W4A16",
        "group_size": 128,
        "samples": len(dataset),
        "max_len": args.max_len,
        "seed": args.seed,
        "ignore": IGNORE,
        "size_gib": round(size, 2),
        "seconds": round(elapsed, 1),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
