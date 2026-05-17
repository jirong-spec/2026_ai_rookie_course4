# lab4_sft_training.py
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from datasets import DatasetDict, load_dataset
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
import torch
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_PATH = BASE_DIR / "lab3" / "train.json"
TEST_PATH = BASE_DIR / "lab3" / "test.json"

BASE_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

def formatting_samples(example, tokenizer):
    """
    將一筆 example["messages"] 轉成一段訓練用文字。
    """
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}

    # return {"input_ids": ..., "attention_mask": ..., "labels": ...}



def main():
    # TODO 2: 載入 Lab3 產生的 train / val JSONL

    
    ds = DatasetDict(
        {
            "train": load_dataset("json", data_files=str(TRAIN_PATH))["train"],
            "test": load_dataset("json", data_files=str(TEST_PATH))["train"],
        }
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # TODO 3: 對資料集做 map，轉成 {"text": "..."} 格式
    ds_proc = ds.map(
        lambda ex: formatting_samples(ex, tokenizer),
        remove_columns=ds["train"].column_names,
    )

    # TODO 4: 載入 base model，啟用 gradient checkpointing
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    base_model.config.use_cache = False
    _ = base_model.gradient_checkpointing_enable()

    # TODO 5: 設定 LoRA 參數（r, alpha, dropout, target_modules）
    lora_cfg = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    peft_model = get_peft_model(base_model, lora_cfg)

    # TODO 6: 建立 num_train_epochs, batch_size等）
    sft_cfg = SFTConfig(
        output_dir="adapter",
        num_train_epochs=10,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        logging_steps=4,
        save_strategy="epoch",
        learning_rate=2e-4,
        warmup_ratio=0.05,
        max_length=1024,
        bf16=True,
        report_to=[],
        dataset_text_field="text",
    )

    # TODO 7: 建立 SFTTrainer 並呼叫 train()
    trainer = SFTTrainer(
        model=peft_model,
        train_dataset=ds_proc["train"],
        processing_class=tokenizer,
        args=sft_cfg,
    )


    trainer.train()

    # TODO 8: 儲存 LoRA 權重與 tokenizer
    trainer.save_model("adapter")
    tokenizer.save_pretrained("adapter")
    print("訓練完成，LoRA 權重已儲存。")

if __name__ == "__main__":
    main()