# lab1_tokenizer_and_budget.py
import json
from typing import Dict, List

from transformers import AutoTokenizer

CANDIDATE_TOKENIZERS = [
    "Qwen/Qwen2.5-3B-Instruct",
    # 可以視環境加入其他模型:
    # "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    # "mistralai/Mistral-7B-Instruct-v0.2",
]

SAMPLES = {
    "繁中": "您好，我想了解退貨流程與退款時程，另外能否提供取件時間的選擇？",
    "英文": "Please provide a concise step-by-step refund process and expected timeline.",
    "程式碼": "def add(a, b):\n    return a + b\n\nprint(add(3, 5))",
}

def token_count_report(models: List[str], texts: Dict[str, str]) -> Dict:
    """
    對每個模型，計算以下資訊：
    - raw_tokens: 對原始文本 encode 後的 token 數
    - chat_tokens: 將文本放入 system+user 之後，apply_chat_template 再 encode 的 token 數
    回傳結構：
    {
      model_id: {
        text_name: {
          "raw_tokens": int,
          "chat_tokens": Optional[int]
        },
        ...
      }
    }
    """
    report = {}

    # TODO 1: 迴圈遍歷每個模型 id
    for model_id in models:
        try:
            # TODO 1-1: 載入 tokenizer
            tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        except Exception as e:
            print(f"[警告] 無法載入 {model_id} 的 tokenizer: {e}")
            continue

        model_result = {}

        # TODO 1-2: 對每個樣本文本計算 raw_tokens / chat_tokens
        for name, text in texts.items():
            raw_tokens = len(tok.encode(text))
         

            # chat_tokens: 需要組成 messages 後，用 apply_chat_template
            messages = [{"role": "system", "content": "..."},
                        {"role": "user", "content": text}]
            try:
                chat_ids = tok.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=False,
                )
                chat_tokens = len(chat_ids)
            except Exception:
                chat_tokens = None

            model_result[name] = {
                "raw_tokens": raw_tokens,
                "chat_tokens": chat_tokens,
            }

        report[model_id] = model_result

    return report

def estimate_training_budget(
    num_samples: int,
    avg_prompt_tokens: int,
    avg_resp_tokens: int,
    epochs: int = 1,
    tokens_per_sec: float = 15000.0,
) -> Dict:
    """
    根據輸入參數估算訓練 token 數與大致時間。
    回傳：
    {
      "total_tokens": int,
      "train_seconds": float,
      "train_hours": float
    }
    """
    # TODO 2: 計算 total_tokens = num_samples * (avg_prompt_tokens + avg_resp_tokens) * epochs
    total_tokens = num_samples * (avg_prompt_tokens + avg_resp_tokens) * epochs

    # TODO 3: 計算 train_seconds = total_tokens / tokens_per_sec
    train_seconds = total_tokens / tokens_per_sec

    # TODO 4: train_hours = train_seconds / 3600
    train_hours = train_seconds / 3600

    return {
        "total_tokens": total_tokens,
        "train_seconds": train_seconds,
        "train_hours": train_hours,
    }

def main():
    report = token_count_report(CANDIDATE_TOKENIZERS, SAMPLES)
    print(json.dumps(report, ensure_ascii=False, indent=2))

    # TODO 5: 假設一個場景，呼叫 estimate_training_budget 做估算
    # 例如：1 萬筆資料，每筆 prompt 200 tokens，回覆 300 tokens，1 個 epoch
    budget = estimate_training_budget(
        num_samples=10000,
        avg_prompt_tokens=200,
        avg_resp_tokens=300,
        epochs=1,
        tokens_per_sec=20000.0,
    )
    print("預算估算：", budget)

if __name__ == "__main__":
    main()