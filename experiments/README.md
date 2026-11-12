# CKEdit 阶段 0 部署包——服务器环境搭建与冒烟验证

目标：在实验室 GPU 服务器上，从裸机到跑通「模型加载 → ROME 编辑 → SafeEdit 判定」全链路。
对应实验设计文档的阶段 0：**过关判据 = smoke_test.py 四步全绿**。

## 0. 服务器要求

| 项目 | 要求 |
|---|---|
| GPU | 单卡 ≥ 24GB（4090 / A100 / A800 均可） |
| CUDA | ≥ 11.8 |
| 磁盘 | ≥ 150GB 空闲（模型 13G ×2 + wikipedia 统计缓存 + 环境） |
| 系统 | Linux（Ubuntu 20.04/22.04 为佳） |
| 网络 | 能访问 huggingface 镜像 hf-mirror.com（或配置代理） |

## 1. 代码上传

把以下两个目录打包上传到服务器（如 `~/ckedit/`）：

```
~/ckedit/
├── EasyEdit/        # 编辑框架（本地 C:\Users\Kmoon\Desktop\Master\EasyEdit）
└── CKEdit/
    └── experiments/ # 本部署包（脚本）
```

## 2. 环境搭建

```bash
cd ~/ckedit/EasyEdit
bash ../CKEdit/experiments/setup_env.sh
```

脚本做三件事：创建 conda 环境（python 3.9.7，EasyEdit 官方版本）→ 安装 PyTorch（CUDA 11.8 版）→ 安装 EasyEdit 依赖。

## 3. 数据与模型下载

```bash
bash ../CKEdit/experiments/download_data.sh
```

下载三样东西（全部走 hf-mirror 镜像，无需代理）：
1. **Llama-2-7b-chat** 模型权重 → `./hugging_cache/llama-2-7b-chat`
2. **SafeEdit 数据集**（9 类有害知识，1350 条测试）→ `./data/SafeEdit_test.json`
3. **SafeEdit-Safety-Classifier** 安全判定器 → `./hugging_cache/SafeEdit-Safety-Classifier`

> ⚠️ Llama-2 是 gated model：先在 https://huggingface.co/meta-llama/Llama-2-7b-chat-hf 用注册邮箱申请访问授权（通常几分钟到几小时通过），下载脚本里填入你的 HF token。
> 嫌授权麻烦的替代方案：用 ModelScope 下载同名镜像（脚本注释里有命令），或先用 Mistral-7B-Instruct-v0.2 跑通（无需授权），llama 授权下来再补。

## 4. 冒烟测试

```bash
conda activate EasyEdit
cd ~/ckedit/EasyEdit
python ../CKEdit/experiments/smoke_test.py
```

四步依次验证，全绿即阶段 0 过关：

```
[1/4] GPU 可用：NVIDIA GeForce RTX 4090
[2/4] SafeEdit 安全判定器加载 OK
[3/4] ROME 编辑管线 OK（含编辑成功指标）
[4/4] 判定器输出 OK → 阶段 0 完成，可进入 E1
```

## 5. 常见问题

| 症状 | 处理 |
|---|---|
| 下载卡住/超时 | 确认 `HF_ENDPOINT=https://hf-mirror.com` 已生效；或换 ModelScope |
| CUDA out of memory | 冒烟阶段不应出现；若出现检查是否混入其他进程（`nvidia-smi`） |
| transformers 版本报错 | 用 Mistral 时需 `pip install transformers==4.34.0`（SafeEdit.md 明确要求） |
| mom2/wikipedia 统计缓存首次运行慢 | 正常，ROME 首次运行会下载 wikipedia 二阶矩统计，第二次起有缓存 |
| ROME 编辑成功率为 0 | 检查 `hparams/ROME/llama-7b.yaml` 的 model_name 是否指向实际存在的权重路径 |

## 6. 过关之后的路标

阶段 0 全绿后，下一步是 **E1 上界实验**（编辑 + 不微调 → 全文基线数字）：
基于 `examples/run_safety_editing.py`（DINM/MEND）改造为 ROME/MEMIT 主线，
把 SafeEdit 有害条目转为 (prompt, target_new) 编辑对。脚本我准备好后放本目录。
