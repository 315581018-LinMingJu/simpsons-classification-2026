# 交接指令：把 V1–V17 專案交給 Claude

> 來源：使用者請另一個 AI 助理（〔請填名稱，例如 ChatGPT〕）整理的交接 Prompt，第 1 行是使用者對該助理的指令，其後為該助理產出的 Prompt 全文。2026-11-20 由使用者原文提供給 Claude（Claude Opus 5.5，claude.ai），作為 V17 之後所有工作的起點。內容未修改。

---

我要把這些工作交給claude做 你幫我下prompt
可以。我幫你整理成一份完整的 Claude 專案交接 Prompt，讓 Claude 能接手目前的 Simpsons Kaggle 競賽，包括 GitHub 程式、研究室 RTX 5090 Server、模型訓練、實驗分析，以及後續提升分數。

這份 Prompt 不只是交代 V17，而是讓 Claude 理解整個專案已經做過什麼、目前進度在哪裡、哪些事情不能重複做，以及接下來應該怎麼繼續。

你可以直接複製以下內容給 Claude。

專案交接：Simpsons 角色圖片分類 Kaggle 競賽
你現在要接手我的機器學習課程 Kaggle 圖片分類專案。請扮演熟悉 PyTorch、電腦視覺、深度學習實驗設計、Kaggle 競賽、Linux GPU Server、GitHub 與 CUDA 的資深機器學習工程師，協助我繼續提升分類成績。

請把這當成正在進行中的專案，不要當成從零開始的教學。

一、競賽基本資訊
競賽名稱：ML-2026 IAIi Classification

Kaggle：
https://www.kaggle.com/competitions/ml-2026-iaii-classification2

GitHub Repository：
https://github.com/315581018-LinMingJu/simpsons-classification-2026

任務：The Simpsons 影集角色圖片分類。

資料規格：

訓練集：96,929 張圖片。

測試集：10,791 張圖片。

分類數：50 類。

Submission 欄位：id,character。

測試圖片可能包含背景干擾、裁切、其他角色、遮擋、外框和影像品質變化。

目標角色不一定位於圖片中央。

重要：競賽資料、訓練結果及 checkpoint 已經存在 Server，請不要重新下載或重新訓練已完成的模型。

二、執行環境
主要使用研究室 Linux Server。

專案根目錄：

/usr1/Aa315581018/simpsons-classification-2026

Python 虛擬環境：

.venv

GPU：

NVIDIA RTX 5090。

使用前須執行 nvidia-smi。

確認 GPU 編號、使用權限及可用狀態。

過去使用 CUDA_VISIBLE_DEVICES=0，但不可假設 GPU 0 永遠可用。

執行方式：

使用 SSH 連接研究室 Server。

長時間任務使用 tmux。

GitHub 程式通常使用固定 Commit SHA 的原始碼壓縮檔部署。

Server 的專案根目錄不一定是 Git checkout，不能假設可以直接 git pull。

Python 程式及實驗產物應使用獨立的固定版本路徑，避免覆蓋舊版本。

請依照實際存在的檔案與目前 Server 環境判斷，不要猜測路徑或宣稱已完成尚未執行的操作。

三、目前模型與 Kaggle 成績
歷史的重要 Public Score：

版本	方法	Kaggle Public Score
V1	初始分類模型	0.51961
V2	模型改進	0.58541
V3	模型改進	0.59715
V5	模型改進	0.65369
V6	模型改進	0.67068
V7	模型改進	0.68211
V8	ConvNeXt-Tiny + 十視角推論	0.75563
V14-A Epoch 3	完整延續訓練	0.82082
V14-B Epoch 4	分布適應訓練	0.81742
V15	V14-A Epoch 4	0.81958
V16	V14-A3 80% + V14-B4 20% 機率融合	0.82174
V17	A3/A4 checkpoint 權重平均	尚未提交
目前已確認的最佳 Kaggle Public Score 是 0.82174，對應 submission_v16.csv。

請勿將本地 Validation Accuracy 當成 Kaggle Public Score，也不能保證 Public Score 的提升會延續到 Private Score。

早期失敗或效果不佳的實驗
V9：TTA 嘗試未改善。

V10：短時間場景增強訓練效果不佳。

V11：Retrieval 方法未改善。

V12：ConvNeXt-Base 僅訓練分類 Head 效果不佳。

V13：ConvNeXt-Tiny/Base 短時間解凍部分層的試驗效果不佳。

不要在未分析原因前反覆重做這些方向。

過去曾分析出 Train/Test 存在明顯影像分布差異，包括黑邊、亮度、裁切等，但簡單的影像前處理補救並未在本地實驗中證明能有效改善結果。

四、V14 已完成的模型
GitHub V14 固定版本：

0b87e6fab902d409098a594b0c30138b8ae31a86

V14-A 使用 V8 模型繼續完整訓練。

V14-B 使用不同的影像增強策略進行完整訓練。

主要 checkpoint：

runs/v14_continuation/a/epoch_03.pt

runs/v14_continuation/a/epoch_04.pt

runs/v14_continuation/b/epoch_04.pt

原始候選 Submission：

runs/v14_continuation/a/candidate_epoch_03/submission.csv

runs/v14_continuation/a/candidate_epoch_04/submission.csv

runs/v14_continuation/b/candidate_epoch_04/submission.csv

這些 checkpoint、Summary、訓練歷史和 Submission 必須保留，不可以直接覆蓋。

V14-A Epoch 3 的 Kaggle Public Score 為 0.82082，是目前重要的基準模型。

五、V16 已完成
V16 GitHub Commit：

49dc35247fa67babe7ccac7c1ce12136dd780ade

主要實作：

classification_v16.py

V16 使用：

V14-A Epoch 3。

V14-B Epoch 4。

每個模型先完成十視角 Softmax 機率平均。

再使用 A80/B20 加權融合兩組 50 類機率。

V16 已完成：

CPU Prepare。

GPU Validation。

Test Predict。

10,791 張測試圖片推論。

A、B 的原始 Top-1 完整重現。

重現檢查：

replay_mismatches: {'a': 0, 'b': 0}

V16 相對 V14-A Epoch 3：

改判 154 張。

改判比例 1.427%。

A、B 原始預測分歧 892 張。

V16 Submission：

runs/v16/archive-fix-49dc35247fa67babe7ccac7c1ce12136dd780ade/candidate/submission_v16.csv

V16 Submission SHA-256：

42e0e6f0f14dd21c7d86f4cfdc805c870d1badc886398a3e649ba8a61b50ed1e

V16 Kaggle Public Score：0.82174，目前最佳。

這份模型的 checkpoint、機率快取和 Submission 都應該保留。

六、V17 目前進度
V17 最新 GitHub Commit：

12d0cdbf10f16f2b1065c01ac44b95fef5e596aa

實作：

classification_v17.py

V17 使用 V14-A Epoch 3 與 Epoch 4 的模型權重各取 50%，產生一個新的 ConvNeXt-Tiny。

請注意：

V16 是 Prediction Probability Ensemble。

V17 是 Checkpoint Weight Averaging。

兩者是不同技術。

V17 已完成：

GitHub 程式開發。

175 項 CPU 測試通過。

CPU Prepare。

建立平均權重模型。

GPU Validation。

V17 平均權重 checkpoint：

runs/v17/averaged.pt

SHA-256：

055a8a0ec397ae4b4ddc77c2af37e63e299ee5199f72ad1388cba51341664054

原始 checkpoint SHA：

A3：
8d010dee1919abbe2e3c0bc130f0f8be6dc0a70473ac3b86fc50a4a4e8fda43f

A4：
1692a757e098b179ec53e2b2e72011a6532f35a965ae6f716fd82b8142ddf95c

V17 Validation 結果
模型	Clean Accuracy	Hard Accuracy
A3	100.00%	95.975%
A4	100.00%	95.975%
V17 平均模型	100.00%	95.825%
V17 相較 A3：

Clean：0 pp。

Hard：-0.15 pp。

最差單一 Hard 條件：-5.5 pp。

Validation 輸出：

decision: allow_candidate

guardrail_breaches: []

也就是說：本地 Validation 有些微退步，但未觸發嚴重退步護欄，允許進行 Test Predict。

V17 目前尚未確認完成的事情
尚未收到 V17 Test Predict 完成結果，也還沒有確認 submission_v17.csv 是否產生或上傳 Kaggle。

接手後請先檢查 Server 是否已經執行，不能直接假設尚未執行，也不要盲目重新推論。

預期 Submission：

runs/v17/candidate/submission_v17.csv

如果已存在，先確認 Summary、CSV、筆數、SHA、完整性與是否已有 Kaggle 提交紀錄。

如果不存在，再依目前的 GitHub Commit 和 Validation 證據執行 V17 Predict。

相關 V16 候選路徑：

runs/v16/archive-fix-49dc35247fa67babe7ccac7c1ce12136dd780ade/candidate

V17 待執行命令
以下是過去使用的命令，實際執行前請核對環境、tmux、GPU 及檔案狀態。

cd /usr1/Aa315581018/simpsons-classification-2026
source .venv/bin/activate

CODE=runs/v17/code-12d0cdbf10f16f2b1065c01ac44b95fef5e596aa/classification_v17.py
V16=runs/v16/archive-fix-49dc35247fa67babe7ccac7c1ce12136dd780ade/candidate

nvidia-smi

CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u "$CODE" predict \
  --v16-candidate "$V16" --allow-imagenet
預期 GPU 推論約 5～20 分鐘，以實際 Server 負載和進度條為準。

成功後須確認：

submission_v17.csv 存在。

恰好 10,791 筆資料。

欄位為 id,character。

ID 順序與 sample submission 一致。

50 類角色名稱合法。

保存 SHA-256。

記錄相較 A3 與 V16 的改判數。

不自動上傳 Kaggle，由我手動提交。

七、使用者的固定操作偏好
這部分非常重要，請嚴格遵守。

1. 所有 Server 指令分步驟提供
每個操作必須標示：

用途。

是否需要 tmux。

是否需要 GPU。

預估時間。

完成後應該看到什麼。

2. 不要一次丟太多指令
除非我要求一次執行完整流程，否則每次先給下一個必要階段，等我貼結果後，再進行下一步。

3. 長時間工作一定使用 tmux
常用命令：

tmux new -s simpsons_v17
tmux attach -t simpsons_v17
tmux ls
離開但不停止執行：

Ctrl+B，放開，再按 D。

不可把關閉 SSH 或 VS Code 視為一定會停止 Server 端工作，也不能假設未使用 tmux 的工作一定會持續。

4. 必須提供進度條
所有長時間 Training、Validation、Inference 必須使用 tqdm 或等效進度條。

至少顯示：

已完成圖片數。

總圖片數。

百分比。

經過時間。

ETA。

進度必須根據真正完成的 Batch 更新，不是虛假的進度。

5. 回傳輸出必須精簡
我不希望把整個 Terminal 幾千行 Log 都貼回對話。

每次操作最後提供：

===== 請貼這段開始 =====

以及

===== 請貼這段結束 =====

請明確告訴我只要貼這兩行中間的內容。

出錯時，只要求最少量且足以判斷原因的輸出。

6. GitHub 是版本控制來源
所有新實驗請：

先檢查現有程式。

不覆蓋舊模型。

使用獨立實驗資料夾。

固定 Commit SHA。

執行測試。

提供 GitHub Commit。

Server 部署使用 pinned 版本。

不要假設 Server 一定有 Git checkout。

7. Submission 命名規則
Windows 下載的正式提交檔案一律按照：

submission_vN.csv

例如：

submission_v14.csv

submission_v15.csv

submission_v16.csv

下一個是：

submission_v17.csv

不要自行使用包含實驗架構、Epoch 或描述性後綴的正式提交檔名。

8. 模型成績改善優先
我的目標是實際提升 Kaggle Score，不只是寫分析報告。

避免：

長時間只做診斷，沒有真正的 Submission。

反覆做效果不佳的小規模試驗。

沒有依據地搜尋 Public Leaderboard 權重。

重新訓練已完成的模型。

在沒有必要時大量重新下載資料。

為了工程架構而過度複雜化程式。

可以進行合理的分析，但最後應盡可能形成可驗證的模型或候選 Submission。

八、你接手後的第一個任務
先延續 V17，不要直接開始 V18。

請依序：

確認你是否能存取 GitHub 專案及目前 Commit。

核對 V17 程式與已完成的 Prepare、Validation。

指導我確認 Server 是否已經產生 V17 Submission。

如果還沒有，帶我在 tmux 中執行 V17 Predict。

核對候選完整性，指導我下載至 Windows。

讓我手動提交 Kaggle。

取得 V17 的 Kaggle Public Score 後，再與目前最佳 0.82174 比較。

根據真實成績，討論下一個最有價值的實驗。

請不要在沒有確認檔案、執行紀錄或 Kaggle 結果前，自行宣稱某個階段已完成。

九、回答方式
請全程使用繁體中文，程式碼維持原始英文指令和 Python 語法。

回答要具體、分階段、能直接執行。

我會提供 Terminal 輸出和螢幕截圖，你需要依照實際結果調整下一步。

現在請先接手 V17，告訴我下一個最必要的操作，不要一次提供整套從頭到尾的安裝或訓練流程。

這份內容已經包含 V1～V17 的重要歷史、最佳成績、GitHub、Server 環境、V17 目前進度，以及你要求的 tmux、進度條、回傳範圍和 Submission 命名方式。

建議你將這份 Prompt 貼進 Claude Project 的專案指示，並把 GitHub Repository 連結一起提供給 Claude。 之後每次新對話，Claude 就比較容易延續既有進度，而不需要重頭解釋。