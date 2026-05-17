# lab5_inference_and_eval.py
import json
import re
from collections import Counter
from typing import Any, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

#BASE_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
BASE_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

def load_base_and_adapter(base_model_id: str, adapter_dir: str):
    """
    載入基礎模型 + LoRA 權重。
    """
    #tokenizer = AutoTokenizer.from_pretrained(adapter_dir or base_model_id, use_fast=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
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

def generate_reply(tokenizer, model, messages, max_new_tokens: int = 256):
    """
    使用正確的 chat template 做推理。
    """
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )

    gen_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    reply_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return reply_text

CJK_RE = re.compile(r"[\u4e00-\u9fff]")

def evaluate_one(example: Dict[str, Any], reply: str) -> Dict[str, Any]:
    """
    啟發式評估一條回覆：
    - 有沒有禮貌用語
    - 有沒有步驟/結構字樣
    - 中文比例是否夠高
    - 是否包含 topic/關鍵詞
    """
    # TODO 3: 自行設計幾個簡單的規則打分
    polite_words = ["您好", "請", "感謝", "謝謝", "麻煩您"]
    structure_words = ["1.", "2.", "3.", "步驟", "首先", "接著", "最後"]
    topic = str(example.get("topic", "")).strip()

    polite = any(w in reply for w in polite_words)
    structured = any(w in reply for w in structure_words)

    non_space_chars = [ch for ch in reply if not ch.isspace()]
    zh_count = sum(1 for ch in non_space_chars if CJK_RE.match(ch))
    zh_ratio = (zh_count / len(non_space_chars)) if non_space_chars else 0.0
    zh_ok = zh_ratio >= 0.5

    topical = topic in reply if topic else False


    score = sum([polite, structured, zh_ok, topical]) / 4.0
    errors = []
    if not polite:
        errors.append("缺少禮貌用語")
    if not structured:
        errors.append("缺少結構化步驟")
    if not zh_ok:
        errors.append("語言非中文為主")
    if not topical:
        errors.append("主題相關性不足")

    return {"score": score, "errors": errors, "reply": reply}

def main():
    #tokenizer, model = load_base_and_adapter(BASE_MODEL_ID, "workdir/adapter")
    tokenizer, model = load_base_and_adapter(BASE_MODEL_ID, "adapter")
    # TODO 4: 讀取 test.jsonl
    test_examples = []
    with open("lab3/test.json", "r", encoding="utf-8") as f:
        test_examples = json.load(f)

    results = []
    for ex in test_examples:
        messages = [m for m in ex["messages"] if m["role"] != "assistant"]
        reply = generate_reply(tokenizer, model, messages, max_new_tokens=256)
        eval_res = evaluate_one(ex, reply)
        eval_res["id"] = ex["id"]
        results.append(eval_res)

    # TODO 5: 計算平均分數與錯誤統計
    avg_score = sum(r["score"] for r in results) / len(results) if results else 0.0
    err_counter = Counter()
    for r in results:
        err_counter.update(r["errors"])

    print(f"平均分數: {avg_score:.3f}")
    print("錯誤統計：", dict(err_counter))

    # TODO 6: 輸出 eval_results.json
    with open("lab5/eval_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()