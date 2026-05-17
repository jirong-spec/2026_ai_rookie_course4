# lab2_chat_template.py
"""
【Lab1 目的】
模型在訓練和推理時，吃的不是純字串，而是有固定格式的 token 序列。
這個格式叫做 Chat Template，由每個模型的 tokenizer 定義。

本 Lab 做三件事：
  1. ensure_system_message      — 確保每筆資料都有 system 角色（沒有就自動補）
  2. to_chat_template_text      — 把 messages 轉成模型看得懂的格式化字串
  3. check_template_consistency — 檢查 messages 結構是否合法，避免訓練踩雷

【原始資料 → 輸出範例】

輸入（JSON messages）：
  [{"role": "system",  "content": "你是專業客服助理..."},
   {"role": "user",    "content": "我想取消訂單，流程是什麼？"}]

輸出（訓練用，add_generation_prompt=False）：
  <|system|>
  你是專業客服助理...</s>
  <|user|>
  我想取消訂單，流程是什麼？</s>

輸出（推理用，add_generation_prompt=True）：
  <|system|>
  你是專業客服助理...</s>
  <|user|>
  我想取消訂單，流程是什麼？</s>
  <|assistant|>
  ← 模型從這行開始生成，這就是 True 和 False 的唯一差別
"""

from typing import Any, Dict, List

from transformers import AutoTokenizer

BASE_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

"""
範例原始資料（實務中可從檔案載入）
ex1：完整格式（含 system）
ex2：故意省略 system，用來測試 ensure_system_message 會自動補上
"""
RAW_EXAMPLES = [
    {
        "id": "ex1",
        "messages": [
            {"role": "system", "content": "你是專業客服助理，請用繁體中文，語氣禮貌。"},
            {"role": "user", "content": "我想取消訂單，流程是什麼？"},
        ],
    },
    {
        "id": "ex2",
        "messages": [
            # 故意省略 system，測試自動補上
            {"role": "user", "content": "請問有沒有學生優惠？"},
        ],
    },
]

DEFAULT_SYSTEM_MESSAGE = "你是專業客服助理，請用繁體中文，語氣禮貌。"

def ensure_system_message(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    確保 messages 的第一則是 system；若沒有就自動補上一則預設 system。

    為什麼需要這步？
    - SFT 訓練時，所有樣本的格式必須一致，system 存在與否會影響 token 序列長度和位置。
    - 如果有些資料有 system、有些沒有，模型學到的格式會不穩定。
    """
    # TODO 1: 如果第一則不是 system，就插入一則預設 system
    # 預設內容可以是： "你是專業客服助理，請用繁體中文，語氣禮貌。"
    if not messages or messages[0].get("role") != "system":
        return [{"role": "system", "content": DEFAULT_SYSTEM_MESSAGE}] + list(messages)
    return list(messages)


def to_chat_template_text(example: Dict[str, Any], tokenizer) -> str:
    """
    將一筆 example（含 messages）轉換成 chat template 的文字結果。
    不加 generation prompt（訓練用）。

    【add_generation_prompt 的差別】
      False（訓練用）：輸出完整對話，包含 assistant 回覆，讓模型學「正確答案長什麼樣」
      True（推理用） ：輸出到 <|assistant|> 為止，模型從這裡接著生成，不能包含答案

    【tokenize=False 的意思】
      只回傳格式化後的字串，不轉成 token id，方便人類閱讀和除錯。
    """
    # TODO 2: 取得 messages 並呼叫 ensure_system_message
    messages = ensure_system_message(list(example["messages"]))
    # TODO 3: 用 tokenizer.apply_chat_template 轉成文字
    chat_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    return chat_text

def check_template_consistency(
    chat_text: str, messages: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    對即將／已經餵給 apply_chat_template 的 messages 做簡單一致性檢查（建議傳入與
    apply_chat_template 相同的列表，例如已經過 ensure_system_message）：
    - 每則是否資料完整（必要欄位是否存在）
    - content 是否為空
    - role 順序是否合理（第一則為 system，之後 user／assistant 交替）

    為什麼要做一致性檢查？
    - apply_chat_template 本身不會報錯，它會盡量硬轉，錯誤格式會悄悄變成錯誤的訓練資料。
    - 訓練完才發現資料有問題，代價極高，事前檢查成本低很多。

    role 順序規則：
      system(0) → user(1) → assistant(2) → user(3) → assistant(4) → ...
      索引為奇數應是 user，偶數（非 0）應是 assistant
    """
    issues: List[str] = []

    # TODO 4: 資料完整性與 content 是否為空
    #   - 每則皆為 dict，且含 role、content
    #   - role、content 經 strip 後不應為空字串

    # TODO 5: role 順序
    #   - 第一則 role 必須為 system
    #   - 之後索引 1,3,5,... 應為 user；2,4,6,... 應為 assistant
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            issues.append(f"messages[{i}] 不是 dict")
            continue

        if "role" not in msg:
            issues.append(f"messages[{i}] 缺少 role")
        if "content" not in msg:
            issues.append(f"messages[{i}] 缺少 content")

        role = str(msg.get("role", "")).strip()
        content = str(msg.get("content", "")).strip()

        if role == "":
            issues.append(f"messages[{i}].role 為空")
        if content == "":
            issues.append(f"messages[{i}].content 為空")

    if messages:
        first_role = str(messages[0].get("role", "")).strip()
        if first_role != "system":
            issues.append("第一則訊息必須為 system")

    for i in range(1, len(messages)):
        role = str(messages[i].get("role", "")).strip()
        expected_role = "user" if i % 2 == 1 else "assistant"
        if role != expected_role:
            issues.append(
                f"messages[{i}] 的 role 應為 {expected_role}，實際為 {role or '空字串'}"
            )
    return {
        "issues": issues,
        "length": len(chat_text),
    }

def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=True)

    for ex in RAW_EXAMPLES:
        messages_for_check = ensure_system_message(list(ex["messages"]))
        chat_text = to_chat_template_text(ex, tokenizer)
        report = check_template_consistency(chat_text, messages_for_check)
        print(f"ID: {ex['id']}, 長度={report['length']}, 問題={report['issues']}")
        # 可以視需要印出 chat_text 片段
        # print(chat_text)

if __name__ == "__main__":
    main()
