# lab3_data_cleaning.py
import json
import random
import hashlib
import re
import unicodedata
from urllib import response
import requests
from typing import Any, Dict, List
from tqdm import tqdm
from transformers import AutoTokenizer

BASE_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

# TODO 0: 嘗試匯入 opencc（如無則 fallback）
try:
    from opencc import OpenCC
    _cc_s2t = OpenCC("s2t")
    def to_trad(s: str) -> str:
        return _cc_s2t.convert(s)
except Exception:
    def to_trad(s: str) -> str:
        return s

BADWORDS = ("垃圾", "白癡", "去死")  # 示意
DEFAULT_SYSTEM = "你是專業客服助理，請用繁體中文，語氣禮貌、清楚且具體。"


def normalize_text(s: str) -> str:
    s = to_trad(s)
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def is_toxic(s: str) -> bool:
    return any(bad in s for bad in BADWORDS)

def call_llm(messages):
    data = {
        "model": "Qwen2.5-3B-Instruct",
        "messages": messages,
        "temperature": 0.8,
        "top_p": 0.6,
        "stream": False,
        "max_tokens": 1024
    }
    
    response = requests.post("http://127.0.0.1:8299/v1/chat/completions", json=data)
    response = response.json()
    response_text = response["choices"][0]["message"]["content"]
    return response_text

from more_topics import more_topics
def build_synthetic_examples(n: int = 60) -> List[Dict[str, Any]]:
    """
    建立一個模擬的客服訓練資料集，
    每筆資料格式：
    {
      "id": "ex0001",
      "messages": [ {role, content}, ... ],
      "topic": "退貨流程",
      "language": "zh"
    }
    """
    topics = [
        ("查詢出貨", "請幫我查詢訂單出貨進度，訂單號碼是 ABC123。"),
        ("退貨流程", "我想辦退貨，該怎麼做？"),
        ("退款時程", "退款通常需要幾天會到帳？"),
        ("修改地址", "下單後可以修改收件地址嗎？"),
        ("維修保固", "產品有保固嗎？如何送修？"),
        ("發票補發", "可以補發發票嗎？流程是？"),
    ]

    def synth_assistant(topic: str, user: str) -> str:
        templates = {
            "查詢出貨": "您好，已收到您的需求。請提供訂單編號與收件資訊，我們會協助您確認目前出貨進度，並回覆預計配送時間。",
            "退貨流程": "您好，若您需要辦理退貨，請先確認商品仍在退貨期限內且包裝完整，接著提供訂單資訊與退貨原因，我們將協助您安排後續流程。",
            "退款時程": "您好，退款時間通常會依付款方式而有所不同。一般在確認退貨或取消成功後，約需數個工作天入帳，我們也會同步通知您。",
            "修改地址": "您好，若訂單尚未出貨，通常可以協助修改收件地址。請提供訂單編號與正確地址，我們會盡快為您確認。",
            "維修保固": "您好，產品是否適用保固需依品項與購買時間判定。請提供訂單編號、商品名稱與問題描述，我們會協助您確認送修方式。",
            "發票補發": "您好，可以協助您申請發票補發。請提供訂單資訊與需求說明，我們會確認處理方式後回覆您。",
        }
        return templates.get(
            topic,
            f"您好，關於「{topic}」的問題，我們已收到您的需求。請提供相關訂單或商品資訊，我們會盡快協助您處理。"
        )

    data = []
    for i in tqdm(range(n)):
        topic, user = random.choice(more_topics)
        # TODO 4: 生成單輪對話
        assistant = synth_assistant(topic, user)

        messages = [
            {"role": "system", "content": DEFAULT_SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
        data.append(
            {
                "id": f"ex{i:04d}",
                "messages": messages,
                "topic": topic,
                "language": "zh-Hant",
            }
        )
    return data

def clean_dataset(
    examples: List[Dict[str, Any]],
    tokenizer,
    max_user_len: int = 512,
    max_total_tokens: int = 2048,
) -> List[Dict[str, Any]]:
    """
    - 對 messages 的 content 做 normalize / 繁簡轉換 / 毒性過濾
    - 過短/過長樣本刪除
    - 以 user 內容的 hash 去重
    - 用 chat template 估計 token 長度，超過 max_total_tokens 的刪除
    """
    cleaned = []
    seen_keys = set()

    for ex in examples:
        toxic_found = False
        messages = []

        for msg in ex.get("messages", []):
            role = msg.get("role", "")
            content = normalize_text(str(msg.get("content", "")))

            if not content:
                continue

            if is_toxic(content):
                toxic_found = True
                break

            messages.append({"role": role, "content": content})

        if toxic_found or not messages:
            continue

        user_concat = " ".join(
            msg["content"] for msg in messages if msg.get("role") == "user"
        ).strip()

        if len(user_concat) < 3 or len(user_concat) > max_user_len:
            continue

        try:
            chat_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
            )
            chat_tokens = len(chat_ids)
        except Exception:
            continue

        if chat_tokens > max_total_tokens:
            continue

        key = hashlib.md5(user_concat.encode("utf-8")).hexdigest()
        if key in seen_keys:
            continue
        seen_keys.add(key)

        new_ex = dict(ex)
        new_ex["messages"] = messages
        cleaned.append(new_ex)

    return cleaned

def save_json(path: str, items: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=4)

def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=True)

    raw_examples = build_synthetic_examples(n=1)
    cleaned = clean_dataset(raw_examples, tokenizer, max_user_len=300, max_total_tokens=1024)
    random.shuffle(cleaned)

    n = len(cleaned)
    train = cleaned[: int(0.9 * n)]
    test = cleaned[int(0.9 * n) : n]

    save_json("train.json", train)
    save_json("test.json", test)

    print(f"資料筆數: train={len(train)}, test={len(test)}")

if __name__ == "__main__":
    main()