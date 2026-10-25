# Qwen3-VL-8B × MedPix-VQA：LoRA 微调与 4bit 量化

在单张 32 GB 显卡上完成 Qwen3-VL-8B-Instruct 的领域微调（LoRA）与 4bit 权重量化（GPTQ / AWQ）的完整工程实现，包含数据转换、训练、评测与量化脚本。

- **模型**：`Qwen3-VL-8B-Instruct`（8.77B 参数，bf16 约 17.5 GB）
- **数据**：`MedPix-VQA`（train 17,420 条 / val 3,080 条，2,050 张医学影像，每图 10 条问答）
- **微调**：只训练 247.8M 参数（2.83%）——LLM 内 LoRA（87.3M）+ 视觉 Merger 与 DeepStack（160.5M），ViT 与词嵌入冻结
- **量化**：GPTQ / AWQ 的 W4A16 组量化，产物为 compressed-tensors 格式，6.73 GiB，vLLM 可直接加载

## 主要结果

| 阶段 | 验证集 Token-F1 | 模型体积 | 单条延迟（median） |
|---|---|---|---|
| 基座（未微调，bf16） | 0.155 | 17.53 GiB | 2.90 s |
| LoRA 微调后（bf16） | **0.345** | 17.53 GiB | 0.52 s |
| GPTQ W4A16 | 0.343 | **6.73 GiB** | **0.28 s** |
| AWQ W4A16（通道平滑 + RTN） | 0.313 | 6.73 GiB | 0.26 s |

评测条件：同一批 150 条验证样本（seed 0）、greedy 解码、`max_new_tokens=256`、vLLM 0.11.2 服务端推理。GPTQ 与 bf16 的配对差为 −0.0016（95% CI [−0.0159, +0.0125]，包含 0）。

> 完整实验记录（配置推导、损失曲线、逐案例对比、量化问题排查）约 3.7 万字，位于 `docs/`，按仓库约定不纳入版本库。

## 环境要求

| 组件 | 版本 / 规格 |
|---|---|
| GPU | 单卡 ≥ 32 GB（实测 RTX 5090 32GB）；多卡可选，脚本走 DeepSpeed ZeRO-2 |
| 驱动 / CUDA | driver 595.80 / CUDA 13.2（运行时使用 torch 自带的 cu128） |
| Python | 3.12 |
| 训练栈 | torch 2.8.0 · transformers 5.3.0 · trl 0.25.0 · peft 0.15.2 |
| 量化栈 | torch 2.9.0 · transformers 4.57.3 · llmcompressor 0.9.0.5 |
| 评测栈 | vllm 0.11.2（配 compressed-tensors 0.12.2） |

三套依赖的版本互相冲突，必须分环境安装：

```bash
# 训练环境
python -m venv --system-site-packages .venv-train
. .venv-train/bin/activate && pip install -r requirements.txt

# 量化环境（不要加 --system-site-packages，避免 torch C 扩展 ABI 冲突）
python -m venv .venv-quant
. .venv-quant/bin/activate && pip install -r requirements-quant.txt

# 评测环境
python -m venv .venv-eval
. .venv-eval/bin/activate && pip install -r requirements-eval.txt
```

若机器上已有同版本 torch，可改用 `bash scripts/setup_env.sh` 只补装差异包。

## 快速开始

```bash
# 1) 数据转换：parquet → LLaVA 格式 JSON + 图片
python data_process/convert.py --data_dir /models/data --out_dir data/medpix

# 2) LoRA 微调（单卡约 1 小时/轮，默认 2 轮）
bash scripts/train_lora.sh

# 3) 训练曲线
python scripts/plot_train.py --output-dir output/qwen3vl8b_medpix_lora

# 4) 验证集评测（transformers 逐条推理）
python scripts/eval_metrics.py --adapter output/qwen3vl8b_medpix_lora --n 150

# 5) 量化：合并 LoRA → GPTQ / AWQ（中间产物写入 /dev/shm）
bash scripts/quantize.sh                 # ALGOS=gptq,awq-rtn SAMPLES=128

# 6) vLLM 评测（精度 + 延迟，同一批样本）
python scripts/eval_vllm.py --model /dev/shm/quant/gptq-w4a16 --n 150

# 7) 汇总对比表、图与配对检验
python scripts/compare_quant.py --runs base=... bf16=... gptq=... awq=... --out output/eval_quant
python scripts/paired_ci.py --a output/eval_quant/bf16 --b output/eval_quant/gptq
```

## 目录结构

```
Qwen3-VL-8b-Finetune/
├── data_process/
│   └── convert.py              # parquet → LLaVA json + 图片（按内容 sha1 去重）
├── src/                        # 训练代码
│   ├── params.py               # 全部超参定义
│   ├── train/train_sft.py      # 训练入口（DeepSpeed / 单卡均可）
│   ├── train/monkey_patch_forward.py
│   ├── model/{load_model,modeling_cls}.py
│   ├── dataset/{sft_dataset,data_utils}.py
│   ├── trainer/sft_trainer.py  # 支持 vision_lr / merger_lr 分组
│   ├── loss/  serve/
│   └── merge_lora_weights.py   # LoRA 合并为完整 HF 权重
├── scripts/
│   ├── setup_env.sh            # 建训练 venv
│   ├── train_lora.sh           # 训练脚本（单卡/多卡自适应，环境变量可覆盖）
│   ├── plot_train.py           # 训练曲线（loss / lr / grad_norm / eval）
│   ├── eval_metrics.py         # 验证集生成式评测：EM / F1 / ROUGE-L / BLEU + 绘图
│   ├── quantize.py             # 量化入口（gptq / awq / awq-rtn）
│   ├── quantize.sh             # 一键：合并 LoRA → 量化 → 汇总
│   ├── eval_vllm.py            # 通过 vLLM 评测（精度 + 延迟）
│   ├── compare_quant.py        # 多份 metrics.json → 对比表 + 三面板图
│   ├── paired_ci.py            # 同批样本配对 bootstrap 95% CI
│   ├── zero{2,3}*.json         # DeepSpeed 配置
│   └── finetune*.sh merge_lora.sh   # 上游原始脚本（DPO / GRPO / 分类等，未改动）
└── requirements*.txt           # 三套环境依赖
```

运行期目录（不入库）：`data/medpix/`（304 MB 转换产物）、`output/`（adapter、日志、评测结果）、`docs/`（实验记录）。

## 训练配置

关键参数均可在 `scripts/train_lora.sh` 顶部通过环境变量覆盖：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `MODEL_PATH` | `/models/weight/Qwen3-VL-8B-Instruct` | 基座权重目录 |
| `DATA_DIR` | `data/medpix` | 转换后的数据目录（含 `train.json` / `val.json` / `images/`） |
| `NUM_DEVICES` | `1` | 1 卡走纯 Python，>1 卡走 DeepSpeed ZeRO-2 |
| `BATCH_PER_DEVICE` | 2（单卡）/ 4（多卡） | 实测 batch 4 峰值 30 GB，贴 32 GB 上限 |
| `GLOBAL_BATCH_SIZE` | 32（单卡）/ 128（多卡） | 梯度累积步数由脚本推导 |
| `EPOCHS` | 2（单卡）/ 3（多卡） | 17,420 条、每图 10 问答，1 轮即可学到答题风格 |
| `LR` | `2e-4` | LoRA 常规学习率；`merger_lr` 固定 1e-5 |
| `MAX_SEQ_LEN` | 4096 | 样本实际约 350~1000 token |
| `IMG_MIN` / `IMG_MAX` | `64*32*32` / `512*32*32` | Qwen3-VL patch=16、merge=2，像素须为 32 的倍数 |
| `FREEZE_MERGER` | `False` | 默认训练 Merger（160.5M）；显存紧张时设 `True` |
| `FLASH_ATTN` | `0` | 实测 flash-attn2 与 sdpa 吞吐一致，默认 sdpa |
| `ENABLE_EVAL` | `1` | 每轮记录一次验证 loss |
| `DRY_RUN` | `0` | 设 1 只打印命令不执行 |

LoRA 相关参数在 `src/params.py`：`lora_rank=32`、`lora_alpha=64`、`lora_dropout=0.05`、`lora_namespan_exclude=['lm_head','embed_tokens']`、`vision_lora=False`。

实测吞吐 0.90 样本/s（约 1.1 s/样本），显存峰值 27 GB（batch 2），17,420 条样本约 545 步/轮。

## 量化配置

| 项目 | GPTQ | AWQ（平滑 + RTN） |
|---|---|---|
| 方案 | Hessian 近似误差补偿，`block_size=128` | 按激活幅度做通道平滑，再按组 RTN |
| 校准集 | 训练集随机 128 条（seed 1234，`max_seq_len=1024`） | 同左 |
| 耗时 | 约 32 min | 约 40 s |
| 产物 | 6.73 GiB，compressed-tensors W4A16（group 128，symmetric） | 同左 |
| 与 bf16 的 F1 差 | −0.0016（95% CI 含 0） | −0.0319（95% CI 不含 0） |

VIT（0.42B）、`lm_head` 与 embedding 保持 bf16，量化部分为 252 个线性层。

`--algo awq` 保留 llmcompressor 原生 AWQ 路径用于对照，但该实现在 GQA 结构上会产出不可用权重（`v_proj → o_proj` 补偿映射因形状不匹配被跳过，权重观测器又被网格搜索污染），部署请使用 `awq-rtn`。


## 参考

- Qwen3-VL：<https://github.com/QwenLM/Qwen3-VL>
- MedPix-VQA：NLM MedPix 影像教学库
- 数据格式沿用 LLaVA / ShareGPT 的 `conversations` 约定
- 量化基于 llmcompressor 与 compressed-tensors

## 许可

代码遵循 Apache-2.0；数据集与基座模型分别遵循其原始许可。仅供研究与教学使用，不构成医疗建议。
