專案說明（SFT 指令微調交付）
=====================================
基礎模型   : TinyLlama/TinyLlama-1.1B-Chat-v1.0
LoRA 權重路徑: workdir/adapter
訓練資料   : lab3/train.json（450 筆繁中客服對話）
評估結果   : lab5/eval_results.json（50 筆，平均分 0.700）

使用方式
-------------------------------------
1. 在專案根目錄安裝依賴：
   uv sync

2. 單筆推理（CLI）：
   python workdir/inference.py \
     --model_id "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
     --adapter_dir "workdir/adapter" \
     --system "你是專業客服助理，請用繁體中文，語氣禮貌。" \
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
