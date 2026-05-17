# lab6_ablation_and_packaging.py
import json
import textwrap
from pathlib import Path
from typing import List, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
try:
    from ..lab5.lab5 import evaluate_one
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from lab5.lab5 import evaluate_one

_LAB6_DIR = Path(__file__).resolve().parent
_WORKDIR = _LAB6_DIR / "workdir"

def load_model_for_inference(base_model_id: str, adapter_dir: str):
    tok_src = adapter_dir if Path(adapter_dir).is_dir() else base_model_id
    tokenizer = AutoTokenizer.from_pretrained(tok_src, use_fast=True, local_files_only=Path(tok_src).is_dir())

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_cfg,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return tokenizer, model

def generate_correct(tokenizer, model, messages) -> str:
    """
    正確模板：使用 apply_chat_template + system。
    """
    # TODO 1: 用與 Lab5 相同方式實作
    # 移除 assistant 回覆（測試資料含答案），只保留 system + user
    messages = [m for m in messages if m["role"] != "assistant"]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )
    gen_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

def generate_wrong(tokenizer, model, messages) -> str:
    """
    錯誤模板：故意不用 chat template，只拼接 user 的內容。
    """
    # TODO 2: 將所有 user content 串起來，手寫 prompt，例如：
    # prompt = f"請用繁體中文回答：\n{user_text}\n回答："
    messages = [m for m in messages if m["role"] != "assistant"]
    user_text = "\n".join(
        m["content"] for m in messages if m.get("role") == "user"
    )
    prompt = f"請用繁體中文回答：\n{user_text}\n回答："
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )
    gen_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

def run_template_ablation(test_examples: List[Dict[str, Any]], tokenizer, model, max_samples: int = 5):
    """
    對前幾筆測試資料做正確模板 vs 錯誤模板的比較。
    """
    for ex in test_examples[:max_samples]:
        good = generate_correct(tokenizer, model, ex["messages"])
        bad = generate_wrong(tokenizer, model, ex["messages"])

        eval_good = evaluate_one(ex, good)
        eval_bad = evaluate_one(ex, bad)

        print(f"\n[{ex['id']}] 正確模板 分數={eval_good['score']:.2f}, 錯誤模板 分數={eval_bad['score']:.2f}")
        print("正確模板回覆：", textwrap.shorten(good, width=200, placeholder=" ..."))
        print("錯誤模板回覆：", textwrap.shorten(bad, width=200, placeholder=" ..."))

def write_inference_script(base_model_id: str, adapter_dir: str, path: str):
    """
    產生一個簡單的推理腳本 inference.py，以方便 CLI 使用。
    """
    # TODO 3: 寫入一段 Python 程式碼到 path
    # 內容可包含 argparse 參數：--model_id, --adapter_dir, --system, --user
    code = f'''\
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

def main():
    parser = argparse.ArgumentParser(description="LoRA 推理腳本")
    parser.add_argument("--model_id", default="{base_model_id}")
    parser.add_argument("--adapter_dir", default="{adapter_dir}")
    parser.add_argument("--system", default="你是專業客服助理，請用繁體中文，語氣禮貌。")
    parser.add_argument("--user", required=True, help="使用者輸入")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir or args.model_id, use_fast=True)
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_cfg,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model.eval()

    messages = [
        {{"role": "system", "content": args.system}},
        {{"role": "user", "content": args.user}},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )
    gen_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    reply = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    print(reply)

if __name__ == "__main__":
    main()
'''
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)

def write_readme(base_model_id: str, path: str):
    """
    產生一個 README_delivery.txt，說明如何使用模型。
    """
    # TODO 4: 說明基礎模型、LoRA 路徑、推理指令範例、注意事項
    readme = f"""\
專案說明（SFT 指令微調交付）
=====================================
基礎模型   : {base_model_id}
LoRA 權重路徑: workdir/adapter
訓練資料   : lab3/train.json（450 筆繁中客服對話）
評估結果   : lab5/eval_results.json（50 筆，平均分 0.700）

使用方式
-------------------------------------
1. 在專案根目錄安裝依賴：
   uv sync

2. 單筆推理（CLI）：
   python workdir/inference.py \\
     --model_id "{base_model_id}" \\
     --adapter_dir "workdir/adapter" \\
     --system "你是專業客服助理，請用繁體中文，語氣禮貌。" \\
     --user "我要查詢出貨進度"

3. 以模組方式執行 Lab6 消融實驗：
   uv run python -m lab6.lab6

注意事項
-------------------------------------
- 推理時必須使用與訓練相同的 chat template（tokenizer.apply_chat_template）。
- 本模型以 4-bit（NF4 QLoRA）量化載入，需要支援 CUDA 的 GPU（建議 8GB+）。
- 若顯存不足，請降低 --max_new_tokens 或確認 bitsandbytes 版本相容。
- 推理 template 不一致（不套用 apply_chat_template）會導致輸出品質明顯下降，
  詳見 Lab6 消融實驗結果。
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(readme)

def main():
    _WORKDIR.mkdir(parents=True, exist_ok=True)
    adapter_dir = str(_LAB6_DIR.parent / "adapter")
    tokenizer, model = load_model_for_inference(BASE_MODEL_ID, adapter_dir)

    # 讀取測試資料（支援 .jsonl 與 .json 兩種格式）
    test_json = _LAB6_DIR.parent / "lab3" / "test.json"
    test_examples = []
    with open(test_json, "r", encoding="utf-8") as f:
        test_examples = json.load(f)

    # TODO 5: 執行模板消融實驗
    run_template_ablation(test_examples, tokenizer, model, max_samples=50)

    # TODO 6: 寫出 inference.py 與 README_delivery.txt
    write_inference_script(BASE_MODEL_ID, adapter_dir, str(_WORKDIR / "inference.py"))
    write_readme(BASE_MODEL_ID, str(_WORKDIR / "README_delivery.txt"))

    print("已輸出 workdir/inference.py 與 workdir/README_delivery.txt")

if __name__ == "__main__":
    main()