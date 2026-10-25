#!/bin/bash
# =============================================================================
# 建一个只给本项目用的 Python 环境（不改你的全局环境）
#
# 为什么要单独建：仓库的 src/model/modeling_cls.py、src/train/monkey_patch_forward.py
# 依赖 transformers 5.x 里的 qwen3_5 模块，requirements.txt 也钉的是 transformers==5.3.0；
# 而你系统里是 4.57.1，直接跑会在 import 阶段挂掉（ModuleNotFoundError: qwen3_5）。
# 这里用 --system-site-packages，torch / flash-attn 这些照旧复用系统里的，只多装 3 个包。
#
# 用法： bash scripts/setup_env.sh
# 建好后 scripts/train_lora.sh 会自动用它（也可以用 PYTHON_BIN=... 指定别的）
# =============================================================================
set -e

VENV=${VENV:-/root/autodl-tmp/venv5}
PIP_INDEX=${PIP_INDEX:-https://mirrors.aliyun.com/pypi/simple/}

echo "==> 创建 venv: $VENV"
python -m venv --system-site-packages "$VENV"

echo "==> 装 transformers / hub / peft（版本与 requirements.txt 一致）"
"$VENV/bin/pip" install -q -i "$PIP_INDEX" \
    "transformers==5.3.0" "huggingface-hub==1.6.0" "peft==0.15.2"

echo "==> 检查 trl / ujson / datasets（这三个在系统环境里就该有）"
"$VENV/bin/python" - <<'PY'
import importlib
need = {"trl": "0.25.0", "ujson": None, "datasets": "3.5.1"}
for mod, ver in need.items():
    try:
        m = importlib.import_module(mod)
        got = getattr(m, "__version__", "?")
        flag = "" if (ver is None or got == ver) else f"  <-- 期望 {ver}"
        print(f"  {mod:9s} {got}{flag}")
    except ImportError:
        print(f"  {mod:9s} 缺失  <-- pip install -i {PIP_INDEX} {mod}{('==' + ver) if ver else ''}")
PY

echo "==> 验证关键依赖"
"$VENV/bin/python" - <<'PY'
import transformers, huggingface_hub, peft, torch
print(f"  transformers {transformers.__version__} | hub {huggingface_hub.__version__} | peft {peft.__version__}")
print(f"  torch {torch.__version__} | cuda {torch.cuda.is_available()}")
import transformers.models.qwen3_5.modeling_qwen3_5  # 4.x 里没有这个模块
print("  qwen3_5 模块 OK -> 训练链能 import")
PY

echo
echo "完成。直接跑： bash scripts/train_lora.sh"
