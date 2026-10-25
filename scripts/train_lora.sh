#!/bin/bash
# =============================================================================
# Qwen3-VL-8B-Instruct × MedPix-VQA  LoRA 微调
#
# 单卡 5090（32G）：直接 bash src/train_lora.sh        ← 不用 deepspeed，纯 torch
# 多卡（A100 等）：NUM_DEVICES=8 bash src/train_lora.sh ← 自动用 deepspeed ZeRO-2
#
# 常用覆盖：
#   NUM_DEVICES=1 BATCH_PER_DEVICE=1 bash src/train_lora.sh   # 显存紧
#   EPOCHS=1 bash src/train_lora.sh                           # 先跑 1 轮看效果
#   FREEZE_MERGER=True bash src/train_lora.sh                 # 只训 LLM 里的 LoRA
#
# 本数据实测：train 17420 / val 3080 / 唯一图 2050 张；答案平均 67 词、图片约 450×470
#            单条样本 ~350~400 token
# 只保存微调部分（lora_enable=True 时 train_sft.py 的行为）：
#   adapter_model.safetensors (~189MB) + non_lora_state_dict.bin (merger ~306MB) + config/processor
# 不存 17GB 基座，也不需要 merge。
# =============================================================================
set -e

export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1          # 权重/数据都在本地，禁止联网，避免加载时卡住
export HF_DATASETS_OFFLINE=1

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_DIR=$(dirname "$SCRIPT_DIR")

# ---- 找代码根目录：仓库可能被挪成 wenjian/ 之类，认「有 src/train/train_sft.py」的那层 ----
CODE_DIR=""
for cand in "$REPO_DIR" "$REPO_DIR"/* "$REPO_DIR"/../*; do
  if [ -f "$cand/src/train/train_sft.py" ] && [ -f "$cand/src/params.py" ]; then
    CODE_DIR=$(cd "$cand" && pwd); break
  fi
done
[ -n "$CODE_DIR" ] || { echo "!! 没找到代码目录（要含 src/train/train_sft.py 和 src/params.py）"; exit 1; }
cd "$CODE_DIR"
export PYTHONPATH="$CODE_DIR/src:$PYTHONPATH"

MODEL_PATH=${MODEL_PATH:-/models/weight/Qwen3-VL-8B-Instruct}
DATA_DIR=${DATA_DIR:-$REPO_DIR/data/medpix}                          # convert.py 的默认输出
OUTPUT_DIR=${OUTPUT_DIR:-$REPO_DIR/output/qwen3vl8b_medpix_lora}

NUM_DEVICES=${NUM_DEVICES:-1}
# 默认值按卡数分两套：单卡有效 batch 32，多卡 128
if [ "$NUM_DEVICES" -eq 1 ]; then
  BATCH_PER_DEVICE=${BATCH_PER_DEVICE:-2}
  GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-32}
  EPOCHS=${EPOCHS:-2}
  SAVE_STEPS=${SAVE_STEPS:-100}
else
  BATCH_PER_DEVICE=${BATCH_PER_DEVICE:-4}
  GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-128}
  EPOCHS=${EPOCHS:-3}
  SAVE_STEPS=${SAVE_STEPS:-50}
fi
LR=${LR:-2e-4}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-4096}
IMG_MIN=${IMG_MIN:-$((64 * 32 * 32))}     # 65536：图本来就小，别硬拉大
IMG_MAX=${IMG_MAX:-$((512 * 32 * 32))}    # 524288：本数据最大 0.47M 像素，基本不缩放
FREEZE_MERGER=${FREEZE_MERGER:-False}
FLASH_ATTN=${FLASH_ATTN:-0}               # 1=用 flash-attn2（装了就更快更省显存）；0=用 sdpa
ENABLE_EVAL=${ENABLE_EVAL:-1}             # 1=每 epoch 记一次 val loss（05_eval_loss.png）

GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))
[ "$GRAD_ACCUM_STEPS" -ge 1 ] || { echo "!! GLOBAL_BATCH_SIZE 太小，至少要 $((BATCH_PER_DEVICE * NUM_DEVICES))"; exit 1; }
mkdir -p "$OUTPUT_DIR"

# 解释器：PYTHON_BIN > setup_env.sh 建的 venv > 系统 python
if [ -z "$PYTHON_BIN" ] && [ -x /root/autodl-tmp/venv5/bin/python ]; then
  PYTHON_BIN=/root/autodl-tmp/venv5/bin/python
fi

# 单卡不走 deepspeed（省掉 JIT 编译，5090 上更省事）；多卡走 ZeRO-2
if [ "$NUM_DEVICES" -gt 1 ]; then
  LAUNCH="deepspeed --num_gpus=$NUM_DEVICES"
  ZERO_ARGS="--deepspeed scripts/zero2.json"
else
  LAUNCH="${PYTHON_BIN:-python}"      # 想用别的解释器：PYTHON_BIN=/path/to/python bash scripts/train_lora.sh
  ZERO_ARGS=""
fi

echo "=============================================================="
echo " 代码 : $CODE_DIR"
echo " 模型 : $MODEL_PATH"
echo " 数据 : $DATA_DIR/train.json"
echo " 输出 : $OUTPUT_DIR"
echo " 卡数 : $NUM_DEVICES   单卡batch: $BATCH_PER_DEVICE   累积: $GRAD_ACCUM_STEPS   有效batch: $((BATCH_PER_DEVICE * NUM_DEVICES * GRAD_ACCUM_STEPS))"
echo " 轮数 : $EPOCHS   lr: $LR   序列: $MAX_SEQ_LEN   存档: 每 $SAVE_STEPS 步"
echo " 启动 : $LAUNCH"
echo "=============================================================="
[ -f "$DATA_DIR/train.json" ] || { echo "!! 先跑 python data_process/convert.py 生成数据"; exit 1; }
[ -z "$(ls -A "$OUTPUT_DIR" 2>/dev/null | grep -v '^plots$')" ] || \
  echo "!! 注意：$OUTPUT_DIR 里已有内容，train_sft.py 发现有 checkpoint-* 会自动续训；想从头训就 rm -rf 它"

if [ "${DRY_RUN:-0}" = "1" ]; then      # DRY_RUN=1 bash scripts/train_lora.sh 只打印命令，不真跑
  echo "[dry-run] 命令如下："
  echo "  cd $CODE_DIR && PYTHONPATH=src $LAUNCH src/train/train_sft.py $ZERO_ARGS --model_id $MODEL_PATH ..."
  echo "[dry-run] 有效 batch = $((BATCH_PER_DEVICE * NUM_DEVICES * GRAD_ACCUM_STEPS))，存档每 $SAVE_STEPS 步"
  command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv || true
  exit 0
fi

EVAL_ARGS=""
if [ "$ENABLE_EVAL" = "1" ]; then
  EVAL_ARGS="--eval_path $DATA_DIR/val.json
             --eval_strategy epoch
             --per_device_eval_batch_size 4
             --prediction_loss_only True"
fi

# 注意：json 里的 image 字段是 "images/xxx.png"，相对 DATA_DIR，所以 image_folder 传 DATA_DIR 而不是 DATA_DIR/images
$LAUNCH src/train/train_sft.py \
    $ZERO_ARGS \
    --model_id "$MODEL_PATH" \
    --data_path "$DATA_DIR/train.json" \
    --image_folder "$DATA_DIR" \
    $EVAL_ARGS \
    --remove_unused_columns False \
    --dataloader_num_workers 8 \
    --dataloader_pin_memory True \
    --lazy_preprocess True \
    --output_dir "$OUTPUT_DIR" \
    --bf16 True --fp16 False --tf32 True \
    --num_train_epochs "$EPOCHS" \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --gradient_checkpointing True \
    --learning_rate "$LR" \
    --merger_lr 1e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 5 \
    --logging_first_step True \
    --save_strategy "steps" \
    --save_steps $SAVE_STEPS \
    --save_total_limit 1 \
    --save_only_model True \
    --report_to tensorboard \
    --run_name "qwen3vl8b_medpix_lora" \
    --max_seq_length $MAX_SEQ_LEN \
    --image_min_pixels $IMG_MIN \
    --image_max_pixels $IMG_MAX \
    --freeze_llm True \
    --freeze_vision_tower True \
    --freeze_merger "$FREEZE_MERGER" \
    --lora_enable True \
    --vision_lora False \
    --use_dora False \
    --lora_rank 32 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --lora_namespan_exclude "['lm_head', 'embed_tokens']" \
    --num_lora_modules -1 \
    --disable_flash_attn2 $( [ "$FLASH_ATTN" = "1" ] && echo False || echo True ) \
    --use_liger_kernel False

echo "训练结束:"
ls -lh "$OUTPUT_DIR"
echo
echo "下一步："
echo "  python scripts/plot_train.py --output-dir $OUTPUT_DIR"
echo "  python scripts/eval_metrics.py --adapter $OUTPUT_DIR \\"
echo "      --data $DATA_DIR/val.json --images $DATA_DIR/images --n 100 --out $OUTPUT_DIR/eval"
echo "不续训就删掉优化器状态: rm -f $OUTPUT_DIR/optimizer.pt"
