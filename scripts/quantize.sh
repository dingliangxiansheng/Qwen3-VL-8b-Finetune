#!/usr/bin/env bash
# GPTQ / AWQ 量化一条龙：合并 LoRA → 4bit 权重量化（W4A16）→ 打印产物大小
#
# 环境（独立 venv，不动 llm_serving 的 .venv-vllm，也不动系统环境）：
#   python3 -m venv /root/autodl-tmp/venv_lc          # 注意：不要加 --system-site-packages
#   TMPDIR=/root/autodl-tmp/tmp_pip /root/autodl-tmp/venv_lc/bin/pip install --no-cache-dir \
#       -i https://pypi.tuna.tsinghua.edu.cn/simple \
#       torch==2.9.0 torchvision==0.24.0 transformers==4.57.3 "datasets==4.8.5" \
#       "llmcompressor==0.9.0.5" sentencepiece protobuf accelerate pillow
#
# 用法：bash scripts/quantize.sh            # GPTQ + AWQ 都做
#       ALGOS=gptq bash scripts/quantize.sh
#       SAMPLES=256 bash scripts/quantize.sh
set -euo pipefail

cd "$(dirname "$0")/.."
REPO_DIR=$(pwd)

PY=${PY:-/root/autodl-tmp/venv_lc/bin/python}
BASE=${BASE:-/models/weight/Qwen3-VL-8B-Instruct}
ADAPTER=${ADAPTER:-$REPO_DIR/output/qwen3vl8b_medpix_lora}
MERGED=${MERGED:-/dev/shm/hf_merged}     # 17GB，放内存盘不占磁盘
ALGOS=${ALGOS:-gptq,awq-rtn}
SAMPLES=${SAMPLES:-128}
# 量化产物：默认写内存盘，磁盘只剩几 G 时也能跑完；要留档再拷回 output/
OUT_ROOT=${OUT_ROOT:-/dev/shm/quant}

export TMPDIR=${TMPDIR:-/root/autodl-tmp/tmp_pip}   # /tmp 在 overlay 上，装包/写中间文件会 ENOSPC
export HF_HUB_OFFLINE=1

# 1) 合并 LoRA：量化的是「微调后的模型」，不是基座
if [[ ! -f "$MERGED/config.json" ]]; then
  echo "== 合并 LoRA -> $MERGED"
  PYTHONPATH="$REPO_DIR/src" "$PY" src/merge_lora_weights.py \
    --model-path "$ADAPTER" \
    --model-base "$BASE" \
    --save-model-path "$MERGED" \
    --safe-serialization
else
  echo "== 复用已合并模型 $MERGED"
fi

# 2) 逐种算法量化（--processor 传基座目录：合并目录的 tokenizer_config 里
#    extra_special_tokens 是 list，transformers 4.57 加载时会 AttributeError）
mkdir -p "$OUT_ROOT"
IFS=',' read -ra LIST <<< "$ALGOS"
for algo in "${LIST[@]}"; do
  case "$algo" in
    gptq)    out="$OUT_ROOT/gptq-w4a16" ;;
    awq)     out="$OUT_ROOT/awq-w4a16" ;;
    awq-rtn) out="$OUT_ROOT/awq-w4a16" ;;   # 自实现通道平滑 + RTN，见 docs1 4.3 问题 6
    *)       echo "未知算法 $algo（支持 gptq / awq / awq-rtn）"; exit 1 ;;
  esac
  echo "== 量化 $algo -> $out"
  "$PY" scripts/quantize.py \
    --algo "$algo" --model "$MERGED" --processor "$BASE" \
    --out "$out" --samples "$SAMPLES" --max-len 1024
done

# 3) 汇总
echo "== 产物"
du -sh "$OUT_ROOT"/* 2>/dev/null
cat "$OUT_ROOT"/*/quant_report.json 2>/dev/null
echo "== 下一步：用 vLLM 评测（需要装了 vllm 的环境，见 scripts/eval_vllm.py 头部说明）"
echo "  $PY scripts/eval_vllm.py --model $OUT_ROOT/gptq-w4a16 --n 150 --out output/eval_quant/gptq"
echo "  $PY scripts/eval_vllm.py --model $OUT_ROOT/awq-w4a16  --n 150 --out output/eval_quant/awq"
