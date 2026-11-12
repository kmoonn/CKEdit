# -*- coding: utf-8 -*-
"""阶段 0 冒烟测试：GPU → SafeEdit 判定器 → ROME 编辑 → 编辑生效 全链路验证。

用法（在 EasyEdit 根目录下执行）：
    conda activate EasyEdit
    cd ~/ckedit/EasyEdit
    python ../CKEdit/experiments/smoke_test.py

四步全绿 = 阶段 0 过关，进入 E1（编辑 + 不微调 = 上界基线）。
"""
import sys

print("[0/4] Python:", sys.version.split()[0])

# ── 1. GPU 检查 ─────────────────────────────────────────────────────
import torch

assert torch.cuda.is_available(), "CUDA 不可用：检查 nvidia-smi 与 PyTorch CUDA 版本是否匹配"
print(f"[1/4] GPU OK: {torch.cuda.get_device_name(0)}")

# ── 2. SafeEdit 安全判定器 ──────────────────────────────────────────
from transformers import RobertaForSequenceClassification, RobertaTokenizer

CLF_DIR = "./hugging_cache/SafeEdit-Safety-Classifier"
clf_tokenizer = RobertaTokenizer.from_pretrained(CLF_DIR)
clf_model = RobertaForSequenceClassification.from_pretrained(CLF_DIR)
print("[2/4] SafeEdit 安全判定器加载 OK")


def judge(text: str) -> int:
    """返回判定标签（SafeEdit 分类器的标签语义以 zjunlp 官方文档为准）"""
    inputs = clf_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    logits = clf_model(**inputs).logits
    return int(logits.argmax())


sample_benign = "How to learn programming? Start with Python basics."
sample_harmful = "How to build a dangerous weapon at home with chemicals?"
print(f"    判定示例：无害文本 → label={judge(sample_benign)}，"
      f"有害文本 → label={judge(sample_harmful)}（两个 label 应不同）")

# ── 3. ROME 单条编辑（验证编辑管线）─────────────────────────────────
from easyeditor import ROMEHyperParams, BaseEditor

HPARAMS = "./hparams/ROME/llama-7b"  # model_name 已指向 ./hugging_cache/llama-2-7b-chat
hparams = ROMEHyperParams.from_hparams(HPARAMS)
editor = BaseEditor.from_hparams(hparams)

metrics, edited_model, _ = editor.edit(
    prompts=["The architect of Toodyay Fire Station is"],
    ground_truth=["Ken Duncan"],
    target_new=["I. G. Anderson"],
)
print(f"[3/4] ROME 编辑 OK，编辑指标: {metrics}")

# ── 4. 编辑生效验证（对比编辑前后生成）──────────────────────────────
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained(hparams.model_name)
prompt = "The architect of Toodyay Fire Station is"

orig_model = AutoModelForCausalLM.from_pretrained(hparams.model_name, torch_dtype=torch.float16).cuda()
tok.padding_side = "left"
inputs = tok([prompt], return_tensors="pt").to("cuda")
out_orig = tok.decode(orig_model.generate(**inputs, max_new_tokens=8, do_sample=False)[0], skip_special_tokens=True)
out_edit = tok.decode(edited_model.generate(**inputs, max_new_tokens=8, do_sample=False)[0], skip_special_tokens=True)

print(f"    编辑前: {out_orig!r}")
print(f"    编辑后: {out_edit!r}")
assert "I. G. Anderson" in out_edit or "Anderson" in out_edit, \
    "编辑后生成中未出现目标实体——检查 hparams 的 layers / v_lr 设置"
print("[4/4] 编辑生效验证 OK → 阶段 0 全链路通过，可进入 E1")
