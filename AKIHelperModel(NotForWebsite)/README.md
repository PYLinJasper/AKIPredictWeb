# AKI Project
- 目標是為能達成隨時間變化進行預測的模型，所以最終採用Long Short Term Memory 這系列有時間（序）性概念的模型，而先使用XGBoost模型訓練則是為了使用其能力--篩選特徵重要性，進而選出對AKI判斷影響顯著的特徵進行下一步訓練，同時也期望能降低所需要的特徵數量，提供未來預測使用時的方便性。
## 資料前處理
- 在`code/MIMIC_Preprocessing.py`中的這份程式碼是用來處理MIMIC的資料集，包括從已建好的資料庫中抓不同feature的資料、處理缺失值、處理LSTM及Xgboost模型的input資料格式。
- 參數設定:
    1. 設定要連接的資料庫:必須設定這五項參數:host, database, user, password, port。
    2. 抓取資料的時間範圍:可更改Main function中的TimeOffset的值，24代表抓取24小時的資料。
    3. 處理缺失值的方式:可選擇**Copy prev or next** or **Mean** or **Zero**，分別代表補前後、補平均、不補值。
- Output Data:
    - MIMIC_LSTM_Input.csv
    - MIMIC_Xgboost_Input.csv

## Xgboost_model
- 在`code/Xgboost_model.ipynb`中的這份程式碼是用Xgboost模型來預測病患有無AKI，主要會有各項feature的重要性、預測結果，以及使用外部資料做validation的結果。
- feature重要性:
![](https://i.imgur.com/nRNgy8O.png)
- 預測結果(test_size設0.25):
    - acc:0.85
    - sensitivity:0.85
    - specificity:0.85
    - confusion matrix:
    
    ![](https://i.imgur.com/G1Ax1qa.png)
- external validation:
    - 使用eICU資料做validation
    - 結果:
        - acc:0.66
        - sensitivity:0.5
        - specificity:0.82
        - confusion matrix:
        
        ![](https://i.imgur.com/F6xmsQK.png)
    - 分析:
        1. sensitivity非常低，機率跟亂猜一樣。
        2. 兩個資料集的分布差異(舉幾個feature為例):
            - creatinine:![](https://i.imgur.com/5fY1uHD.png)
            - urine:![](https://i.imgur.com/NJ4B834.png)
            - BUN:![](https://i.imgur.com/xvp18Wa.png)
        3. 缺值的多寡可能也有影響。
        4. 因為醫院來源不同，其AKI之判斷方法、量測數值精度等皆會有差異。

## AKIHelper_LSTM_model
該檔案為訓練LSTM模型的程式碼以及分析模型成效，程式語言為Python，編譯環境為Google提供之Colab，使用GPU進行訓練。
- 資料前處理
    - 將資料每六筆(row)集合成一個list，丟入模型中訓練
    - Input Shape為 [(None, 6, 16)]
        - 表6個時段，16個特徵
- GRUModel: LSTM架構模型
- 分析方式
    - ROC Curve
    - loss隨epoch的分布圖
    - Threshold與Accuracy間的分布圖
- ACC計算方式
    - 將6個時段預測出來的機率平均，大於Threshold判為1，反之則0，再與正確答案做比對得到Accuracy
- 成效

| 預測方式 | Sensitivity | Specificity | ACC |
| -------- | -------- | -------- | --------- |
| 以6個時段平均計算|0.69|0.815|0.76|
| 以最後一個時段計算|0.754| 0.874|0.824|
| 拿最後一個時段機率|0.79|0.8|0.79|
| 單看斜率| 0.79|0.65|0.73|
| 斜率+6個時段任一機率>0.8設有AKI;<0.2設無AKI|0.84|0.61|0.74|
| 斜率+6個時段任一機率>0.85設有AKI;<0.15設無AKI|0.82|0.63|0.74|
| 斜率+6個時段任一機率>0.9設有AKI;<0.1設無AKI|0.81|0.64|0.74|

![](https://i.imgur.com/U4P0rUK.png)
