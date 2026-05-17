# 2026 AI Rookie Course — Lab 0–6 SFT Pipeline

繁中客服機器人完整 SFT（Supervised Fine-Tuning）流程，涵蓋資料準備、QLoRA 訓練、推理評估、消融實驗與交付封裝。

---

## 成果總覽

| Lab | 主題 | 結果 |
|-----|------|------|
| Lab 0 | 環境建置 | uv + pyproject.toml |
| Lab 1 | Chat Template 格式化 | 450 筆訓練格式驗證通過 |
| Lab 2 | 資料清洗 | 去除雜訊、格式標準化 |
| Lab 3 | SFT 資料集 | train 450 筆 / test 50 筆（繁中客服） |
| Lab 4 | QLoRA 訓練 | TinyLlama-1.1B，loss ≈ 0.02，token acc ≈ 99.3% |
| Lab 5 | 推理 & 批次評估 | **平均分 0.975 / 1.0**（50 筆，90% 滿分） |
| Lab 6 | 消融實驗 & 交付封裝 | 正確模板 0.975 vs 錯誤模板 0.945，勝出 5 筆、0 敗 |

---

## Lab 4 訓練設定

| 參數 | 值 |
|------|----|
| 基礎模型 | TinyLlama/TinyLlama-1.1B-Chat-v1.0 |
| LoRA rank | 32 |
| LoRA alpha | 64 |
| Epochs | 10 |
| Learning rate | 2e-4 |
| Batch size | 2（grad accum 4） |
| 量化 | 4-bit NF4 QLoRA |
| 最終 loss | ≈ 0.02 |

---

## Lab 5 評估結果（50 筆）

規則評分：禮貌用語 / 結構化步驟 / 中文比例 ≥ 50% / 主題關鍵字

| 指標 | 值 |
|------|----|
| 平均分 | **0.975 / 1.0** |
| 滿分筆數 | 45 / 50（90%） |
| 0.75 分筆數 | 5 / 50（唯一弱點：主題關鍵字） |
| 0.5 分以下 | 0 |

---

## Lab 6 消融實驗：Chat Template 重要性

同樣的模型，對 50 筆測試資料分別用**正確模板**（`apply_chat_template`）與**錯誤模板**（手拼字串）推理：

| | 正確模板 | 錯誤模板 |
|--|---------|---------|
| 平均分 | **0.975** | 0.945 |
| 滿分筆數 | 45 / 50 | 40 / 50 |
| 最低分 | 0.75 | 0.50（英文亂碼） |
| 勝 / 平 / 負 | **5 / 45 / 0** | — |

> **結論**：錯誤模板讓模型在 10% 的測試案例失分，其中一筆（ex0102）完全格式崩潰，混入英文 email 殘碼，得 0.50 分。正確模板 50 筆無一低於 0.75，驗證「訓練與推理 template 必須一致」。

---

## 執行方式

```bash
# 安裝依賴
uv sync

# Lab 5 推理評估
cd 專案根目錄
uv run python -m lab5.lab5

# Lab 6 消融實驗
uv run python -m lab6.lab6

# 單筆 CLI 推理
python lab6/workdir/inference.py \
  --model_id "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
  --adapter_dir "adapter" \
  --system "你是專業客服助理，請用繁體中文，語氣禮貌。" \
  --user "我要查詢出貨進度"
```

---

## 報告與投影片

- [report.html](report.html) — 完整實驗報告（各 Lab 說明 + 數據）
- [slides.html](slides.html) — 19 張投影片簡報（可瀏覽器開啟）
- [lab6/workdir/README_delivery.txt](lab6/workdir/README_delivery.txt) — 交付說明文件
