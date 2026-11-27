# AI 指令與使用紀錄

依課程規定（Git 專案須包含「AI 指令」），本資料夾保存本專案使用 AI 輔助的指令與紀錄。

| 檔案 | 內容 |
|---|---|
| [`01_handoff_prompt_V1-V17.md`](01_handoff_prompt_V1-V17.md) | 把 V1–V17 專案交給 Claude 的交接指令（原文） |
| [`02_github_rules.md`](02_github_rules.md) | 給 Claude 的 GitHub 操作規範（原文） |
| [`03_user_preferences.md`](03_user_preferences.md) | 使用者在過程中提出的操作偏好 |
| [`04_claude_session_log.md`](04_claude_session_log.md) | V17 → V22、報告與繳交期間，每一則主要指令與 AI 的對應工作、結果、錯誤與修正 |

## 使用的 AI
| 階段 | AI | 用途 |
|---|---|---|
| V1–V17 | ChatGPT | 早期版本的程式開發與實驗（細節見 `01_handoff_prompt_V1-V17.md` 與 `experiments/` 各版 README） |
| 交接與規範 | ChatGPT | 產生 `01`、`02` 兩份給 Claude 的指令 |
| V17–V22、報告、繳交 | Claude Opus 5.5（claude.ai） | 實驗設計、程式撰寫、結果分析、文件與 PDF 報告 |

## 原則
- AI 沒有存取研究室 Server 或 Kaggle 的權限：所有訓練、推論都由使用者在 Server 上執行，Kaggle 由使用者手動提交；文件中的數字皆為實際執行結果。
- 最終方法（V20–V22）只使用訓練集標籤；V18 的測試集偽標籤因可能違反作業規則而不採用（見 `../report/` 與 PDF 第 11.2 節）。
