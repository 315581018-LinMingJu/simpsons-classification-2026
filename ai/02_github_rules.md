# GitHub 操作規範（給 Claude 的固定規則）

> 來源：使用者請另一個 AI 助理 ChatGPT 整理的規則，第 1 行是使用者的指令。2026-11-21 由使用者原文提供給 Claude，之後所有 commit、push、PR 都依此規範進行。內容未修改。

---

我想要你把連接GUTHUB帳號要做什麼事情的規矩整理好 我要丟給CLAUDE
我幫你整理成一份可以直接貼給 Claude Project 專案指示的規則。

這份專門規範 Claude 連接 GitHub 帳號後，如何查看專案、修改程式、執行測試、Commit、Push，以及如何與研究室 Server 配合。

重點是：Claude 必須先了解現有專案，不能亂改、不能覆蓋舊實驗，也不能沒有驗證就宣稱完成。

Claude GitHub 連接與專案操作規範
一、GitHub 專案資訊
我的 GitHub Repository：

https://github.com/315581018-LinMingJu/simpsons-classification-2026

主要分支：main

這是正在進行中的機器學習 Kaggle 競賽專案，已有多個版本的實驗、模型程式及測試。

GitHub 是程式碼與版本管理的主要來源，研究室 Linux Server 是實際執行 GPU 訓練及推論的環境。

請嚴格遵守以下規則。

二、連接 GitHub 帳號的規則
優先使用 Claude 已授權的 GitHub Connector、整合工具或可用的 GitHub 操作環境。

如果尚未連接 GitHub，先明確告訴我需要透過哪個介面授權。

不要要求我直接提供 GitHub 密碼、Personal Access Token、SSH 私鑰或其他秘密憑證。

連接後，先確認：

能否讀取指定 Repository。

能否讀取 main。

是否具備建立 Commit、分支與推送的權限。

目前最新 Commit SHA。

是否有尚未合併或尚未提交的工作。

如果只有讀取權限，就明確說明無法直接修改或 Push，不要假裝已經更新 GitHub。

不要擅自修改 Repository 的公開狀態、Collaborators、Secrets、Actions 設定、分支保護或其他管理權限。

不要將 Kaggle API Key、Server 登入資訊、存取憑證、私人資料或 .env 秘密檔案提交到 GitHub。

三、每次開始工作前，必須先檢查專案
在修改任何程式碼之前，先閱讀與本次任務有關的現有程式、README、測試、相關實驗紀錄及最近 Commit。

必須確認：

目前使用哪個版本的模型。

相關程式碼是否已存在。

之前是否已經做過相同實驗。

已有功能能否直接重用。

修改是否會影響其他版本。

新工作依賴哪些資料、checkpoint 或輸出檔案。

禁止在沒有檢查既有程式的情況下，直接重新寫一套功能。

若 GitHub 中的資訊與我提供的 Server 執行紀錄不同，請先指出差異，不能自行假定某一邊一定正確。

四、修改程式的原則
優先在現有架構上做必要修改，不要過度工程化。

盡量重用已有的資料處理、模型載入、Validation、Inference 和 Submission 功能。

不要為了小功能建立大量沒有必要的抽象層或檔案。

保留原本程式風格，註解簡潔、使用繁體中文。

新實驗應使用獨立版本，例如 classification_v18.py、v18/、tests/test_classification_v18.py。

修正既有版本的錯誤時，可以修改原始程式，但必須說明修改範圍及相容性影響。

不可任意刪除或重新命名既有的重要檔案。

不可在沒有明確需求時修改舊模型的訓練設定或推論邏輯。

修改前要理解原有的 CLI 參數、資料來源與輸出格式。

避免把實驗結果寫入與舊版本共用的檔案路徑。

最重要的原則：不能為了做新版本，破壞已經成功的舊版本。

五、Git Branch、Commit、Push 規則
1. 修改前
先取得 Repository 最新狀態。

記錄目前的 Base Commit SHA。

確認工作目錄沒有不相關或未保存的變更。

若使用本地 Git 工作區，先檢查 git status。

若使用 GitHub Connector，確認對應的分支及檔案版本。

2. 修改時
每次只處理目前指定的任務。

不要順便修改其他無關功能。

不要覆蓋其他人已經提交的程式。

遇到版本衝突時先檢查差異，不要強制覆寫。

3. Commit 規範
Commit 訊息要能清楚說明改了什麼。

例如：

feat(v18): add checkpoint ensemble inference

fix(v17): correct checkpoint loading validation

test(v18): add submission integrity checks

不要使用沒有資訊量的訊息，例如 update、fix、done。

4. Push 規範
一般功能優先使用新分支與 Pull Request。

如果我已明確授權這個專案的特定任務可以直接推送 main，才可以直接 Push。

不得自行 Force Push。

不得自行刪除遠端分支或重寫歷史。

Push 前必須先通過適用測試。

Push 後必須核對 GitHub 遠端 Commit 確實存在。

5. 完成後必須提供
Repository 名稱。

Branch 名稱。

完整 40 字元 Commit SHA。

GitHub Commit 或 Pull Request 連結。

這次新增／修改的檔案。

測試結果。

尚未完成或尚未驗證的部分。

建立 Commit 不等於成功 Push；成功 Push 也不等於 Server 已完成實驗。這三者必須清楚區分。

六、測試與品質檢查
在推送程式前，必須盡可能完成適用的測試：

Python 語法檢查。

新功能單元測試。

既有功能回歸測試。

CLI 參數及輸出路徑檢查。

Checkpoint 格式與 SHA 驗證。

資料切分與類別順序檢查。

Submission CSV 欄位、筆數及 ID 順序檢查。

Cache 是否能安全重用。

中斷後重新執行是否會覆蓋既有產物。

CPU 可測試部分的整合測試。

不能只因為程式通過語法檢查，就宣稱模型推論成功。

如果沒有實際 GPU 環境，必須寫明：

「CPU 測試已完成，GPU Validation／Predict 尚未執行。」

不要虛構 GPU 執行結果、Validation 分數、Kaggle Score 或模型性能。

七、GitHub 與研究室 Server 的關係
研究室 Server 專案根目錄：

/usr1/Aa315581018/simpsons-classification-2026

Python 環境：

.venv

GPU：

NVIDIA RTX 5090。

但 GitHub 連接權限不代表 Claude 自動取得研究室 Server 的 SSH 或執行權限。

如果沒有 Server 存取權限，Claude 應該提供可以讓我直接執行的 Linux 指令，並根據我回傳的 Terminal 輸出繼續工作。

不能宣稱自己已經在 Server 執行程式，除非實際使用了有權限的工具並取得執行結果。

八、Server 程式部署方式
我的 Server 不一定有 Git checkout。

因此，不要直接假設：

git pull

可以使用。

優先使用固定 Commit SHA 下載整份 GitHub 原始碼，放入獨立資料夾，例如：

runs/v18/code-<完整SHA>/

實驗結果則另外存放：

runs/v18/

固定版本下載必須：

使用完整 Commit SHA。

確認下載成功才開始執行。

避免覆蓋舊的程式版本。

從原專案根目錄執行，維持既有資料相對路徑。

保留舊模型和 checkpoint。

確認新程式與現有 .venv 相容。

需要時提供下載檔案完整性驗證。

如果之前已經下載相同 Commit，應檢查是否能重用，而不是每次重新下載。

九、模型及實驗產物保護
以下檔案或資料不可任意刪除或覆蓋：

原始 Kaggle 資料。

V8 模型及相關記錄。

V14-A、V14-B checkpoint。

V14 的訓練歷史。

V16 模型推論機率快取。

V16 最佳 Submission。

V17 平均模型與 Validation。

其他已完成實驗的 Summary、Log、CSV。

新實驗一律使用新版本資料夾。

每次執行前應先確認目標輸出是否已存在，不能直接清空舊目錄。

若要清除大型快取、重跑模型或刪除 checkpoint，必須先說明影響並取得我的明確同意。

十、GPU 實驗規則
每個 GPU 工作階段都要提供：

使用目的。

是否需要 tmux。

使用哪張 GPU。

預估執行時間。

VRAM 需求。

完成後產出檔案。

需要回傳的最少資訊。

長時間訓練和推論必須有即時進度條，包含：

已完成圖片數。

總圖片數。

百分比。

Elapsed。

ETA。

進度必須根據實際完成的 Batch 更新。

如果任務中斷，優先確認是否有完整且可驗證的快取，避免重新運算已完成的部分。

禁止未經確認同時啟動多個大量占用 GPU 的實驗。

十一、Kaggle Submission 命名
正式提交檔必須依照流水號命名：

submission_vN.csv

例如：

submission_v16.csv

submission_v17.csv

submission_v18.csv

新版本使用下一個尚未使用的號碼，不能覆蓋舊 Submission。

不得自行把正式提交檔命名成：

submission_v18_epoch3_ensemble_test.csv

這類包含大量附加描述的檔名。

實驗細節應寫入 summary.json 或 README，而不是塞進正式 CSV 檔名。

Claude 不得在未經我的授權下替我提交 Kaggle，也不能把本地 Accuracy 當成 Kaggle Public Score。

十二、每次工作完成的回報格式
完成 GitHub 任務後，請按以下格式回報：

1. 本次任務

簡短說明完成什麼功能。

2. GitHub

Repository：

Branch：

完整 Commit SHA：

Commit／PR 連結：

是否已成功 Push：

3. 程式修改

說明新增、修改的主要檔案，以及修改原因。

4. 測試

通過的測試數量。

測試使用的環境。

是否完成 CPU 測試。

是否完成 GPU 測試。

尚未驗證的項目。

5. Server 執行

提供適用於無 Git checkout 環境的 pinned 下載指令。

按照順序提供：

Prepare → GPU Probe → Validation → Predict。

不要一次要求我全部執行，除非我明確希望自動跑完整流程。

6. 最少回傳內容

提供容易複製的指令，輸出使用：

===== 請貼這段開始 =====

===== 請貼這段結束 =====

我只需要回傳這兩段標記之間的內容。

十三、遇到問題時的處理原則
如果發現錯誤：

先確認目前執行到哪個階段。

優先閱讀實際 Traceback、Log 和相關原始碼。

找到具體原因，不要盲目修改多個地方。

使用最小必要變更修正。

加入對應回歸測試。

重新執行適用測試。

推送新的 Commit。

提供新的固定版本下載指令。

不修改舊實驗的原始紀錄來偽造通過結果。

不要為了讓測試通過而降低安全檢查，除非已證明原檢查與實際資料格式不相容，而且有合理且可驗證的修正方式。

十四、授權與自主工作範圍
當我明確交代一個程式開發任務時，你可以自主：

閱讀 GitHub 專案。

分析相關程式碼。

設計必要的實作方式。

修改授權範圍內的程式碼。

撰寫與執行 CPU 測試。

建立有意義的 Commit。

依照既有授權政策建立 PR 或推送指定分支。

撰寫 README、操作指令及實驗說明。

但下列行為必須另外取得我明確同意：

Force Push 或重寫 Git 歷史。

刪除重要程式、資料、checkpoint 或實驗結果。

修改 Repository 權限、公開狀態或管理設定。

啟動大量 GPU 訓練或長時間消耗運算資源。

提交 Kaggle。

上傳私人資料、模型權重或大型資料集到外部服務。

新增需要付費的服務或資源。

十五、最重要的總原則
GitHub 的工作是負責開發、驗證、保存和部署程式碼；Server 的工作是進行實際 GPU 訓練與推論；Kaggle 的工作是評估提交的模型預測。

三者的狀態不得混為一談。

每次接手任務時：

先讀現有程式 → 確認已完成進度 → 修改必要功能 → 執行測試 → Commit／Push → 核對 GitHub → 指導 Server 執行 → 驗證結果。

請優先追求正確、可重現、安全且能真正改善 Kaggle 成績的實驗，不要重複已做過的工作，也不要過度工程化。

所有說明使用繁體中文。

以上規則適用於後續所有 GitHub 相關工作。

另外，在 Claude 裡真正連接 GitHub 帳號，通常還需要你在 Claude 的整合或連接器設定中授權 GitHub，並選擇允許存取的 Repository；貼這份規則本身不會自動完成帳號授權。實際能否 Commit、Push，取決於該連接器的權限及 Claude 所處的執行環境。

我建議你把這份放在 Claude Project 的 Instructions（專案指示），而先前那份 V1～V17 實驗交接內容則放在 Project Knowledge（專案知識）。這樣可以把「固定操作規矩」和「目前專案進度」分開，後續比較好維護。