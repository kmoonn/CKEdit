#!/usr/bin/env bash
# 阶段 0 环境搭建：conda + PyTorch + EasyEdit 依赖
# 用法：在 EasyEdit 根目录下执行  bash ../CKEdit/experiments/setup_env.sh
set -e

# 1. conda 环境（EasyEdit 官方推荐 python 3.9.7）
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh
conda create -n EasyEdit python=3.9.7 -y
conda activate EasyEdit

# 2. PyTorch（CUDA 11.8 版；服务器为 CUDA 12.x 时把 cu118 换成 cu121）
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu118

# 3. EasyEdit 依赖
pip install -r requirements.txt

# 4. 微调依赖（E2 用）：trl + peft
pip install trl==0.7.11 peft==0.6.2 datasets accelerate

echo "==== 环境就绪 ===="
python -c "import torch; print('torch:', torch.__version__, '| cuda:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO-GPU')"
