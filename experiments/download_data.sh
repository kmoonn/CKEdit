#!/usr/bin/env bash
# 阶段 0 数据下载：模型 + SafeEdit 数据集 + 安全判定器（全部走 hf-mirror）
# 用法：在 EasyEdit 根目录下执行  bash ../CKEdit/experiments/download_data.sh
# 前置：已 conda activate EasyEdit
set -e

export HF_ENDPOINT=https://hf-mirror.com
mkdir -p ./hugging_cache ./data

# ── 1. Llama-2-7b-chat（gated，需 HF 授权 + token）────────────────
HF_TOKEN="${HF_TOKEN:-}"          # 先 export HF_TOKEN=hf_xxx 填入你的 token
if [ ! -d ./hugging_cache/llama-2-7b-chat ]; then
  if [ -n "$HF_TOKEN" ]; then
    pip install -U "huggingface_hub[cli]" -q
    hf download meta-llama/Llama-2-7b-chat-hf \
      --token "$HF_TOKEN" \
      --local-dir ./hugging_cache/llama-2-7b-chat
  else
    echo "!! 未设置 HF_TOKEN。两个替代方案："
    echo "   a) 到 https://huggingface.co/meta-llama/Llama-2-7b-chat-hf 申请授权，"
    echo "      然后 export HF_TOKEN=hf_xxx 重跑本脚本"
    echo "   b) ModelScope（无需授权）："
    echo "      pip install modelscope && modelscope download --model AI-ModelScope/llama-2-7b-chat-hf --local_dir ./hugging_cache/llama-2-7b-chat"
    exit 1
  fi
fi

# ── 2. SafeEdit 数据集（9 类有害知识，1350 测试条目）────────────────
if [ ! -f ./data/SafeEdit_test.json ]; then
  git clone https://hf-mirror.com/datasets/zjunlp/SafeEdit /tmp/safeedit_data
  cp /tmp/safeedit_data/SafeEdit_test.json ./data/ && rm -rf /tmp/safeedit_data
fi

# ── 3. SafeEdit 安全判定器（RoBERTa）────────────────────────────────
if [ ! -d ./hugging_cache/SafeEdit-Safety-Classifier ]; then
  pip install -U "huggingface_hub[cli]" -q
  hf download zjunlp/SafeEdit-Safety-Classifier \
    --local-dir ./hugging_cache/SafeEdit-Safety-Classifier
fi

echo "==== 下载完成 ===="
ls -lh ./hugging_cache ./data
