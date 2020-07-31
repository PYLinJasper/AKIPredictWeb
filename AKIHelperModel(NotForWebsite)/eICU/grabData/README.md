# eICUGrabData
- 分別介紹XGBoost跟LSTM在eICU資料集如何取資料
  - JP_Delta_GrabData -> XGBoost的資料取法
  - JP_RNN_GrabData -> LSTM的資料取法
- 共通特性
  - 詳細特徵訊息參考[合理性判斷](https://hackmd.io/OJLvqC3NQAO7KF3A54BjiA?both)
  - 變數strsql_timeoffst決定往前推多久時間（分鐘）
      - e.g.往前推一天就設定為60*24 = 1440
  - 共通特徵篩選標準（RNN特徵篩選範圍為主）
  - 皆剔除已洗腎病人
  - 性別0為生理男性，1為生理女性
 
|Age|weight|Nettotal|Potassium|Sodium|Hct|BUN|pH|OI(Oxygon Index)|Temperature|DiastolicBP|SystolicBP|albumin|creatinine|glucose|
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
|<120 (years)|1 - 635 (kg)|< 3000 (ml)|< 10 (meq/l)|100 - 180 (meq/l)|10 - 70 (%)|1-150 (mg)|6.8 - 7.8|100 - 700|33 - 42 (celsius)|30 - 140 (mmHg)|60 - 200 (mmHg)|1 - 10 (g/dl)|0.2 - 10 (mg/dl)|10 - 600 (mg/dl)|
## JP_Delta_GrabData
- 取的欄位數為該欄位有值的欄位中位數
- 採用 MSI (MSI = HR / MAP)
- 提供三種eGFR之結果（MDRD_Simple , MDRD_Fine , CKI）
- 缺值以1000000表示
## JP_RNN_GrabData
- 採用 SI (SI = HR / BP)，但變數名稱仍然為MSI
- 未採用eGFR但可由JP_Delta_GrabData提供之算法獲得
- 缺值以-1表示
- 變數offset為取值時間區間（分鐘）
  - e.g. 四小時一個區間則offset = 60*4 = 240
- 特徵後綴數字為距離AKI多久前的資料（後綴數字從1開始）
  - e.g. 設offset為240（四個小時），pH_2 後綴為2則時間區間為發聲AKI前4 - 8個小時之資料
- 補值方式為前後相加除以2，若只有前面有值或後面有值直接補該值

e.g.1 前後有值（粗體為補值）
|pH_1|pH_2|pH_3|
|-|-|-|
|6.9|**7**|7.1|

e.g.1 前有值（粗體為補值）
|pH_1|pH_2|pH_3|
|-|-|-|
|6.9|**6.9**|**6.9**|

e.g.1 後有值（粗體為補值）
|pH_1|pH_2|pH_3|
|-|-|-|
|**7.1**|**7.1**|7.1|
