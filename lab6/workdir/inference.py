import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

def main():
    parser = argparse.ArgumentParser(description="LoRA 推理腳本")
    parser.add_argument("--model_id", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--adapter_dir", default="/home/jimmy/2026_ai_rookie_course4/adapter")
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
        {"role": "system", "content": args.system},
        {"role": "user", "content": args.user},
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
