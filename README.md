# AKI Project
## 資料前處理
- 在`code/MIMIC_Preprocessing.py`中的這份程式碼是用來處理MIMIC的資料集，包括從已建好的資料庫中抓不同feature的資料、處理缺失值、處理LSTM及Xgboost模型的input資料格式
- 參數設定:
    1. 設定要連接的資料庫:必須設定這五項參數:host, database, user, password, port。
    2. 抓取資料的時間範圍:可更改Main function中的TimeOffset的值，24代表抓取24小時的資料。
    3. 處理缺失值的方式:可選擇**Copy prev or next** or **Mean** or **Zero**，分別代表補前後、補平均、不補值。
- Output Data:
    - MIMIC_LSTM_Input.csv
    - MIMIC_Xgboost_Input.csv

## Xgboost_model
- 在`code/Xgboost_model.ipynb`中的這份程式碼是用Xgboost模型來預測病患有無AKI，主要會有各項feature的重要性、預測正確率，以及使用外部資料做validation的結果。
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

## AKIHelper_LSTM_model
該檔案為訓練LSTM模型的程式碼以及分析模型成效，程式語言為Python，編譯環境為Colab
- GRUModel: LSTM架構模型
- 分析方式包含
    - ROC Curve
    - loss隨epoch的分布圖
    - Threshold與Accuracy間的分布圖
## 網頁
- 登入/註冊
![](https://i.imgur.com/hp7lOok.png)
- 首頁
![](https://i.imgur.com/As0wcUR.png)
- 簡介
![](https://i.imgur.com/iiBZlac.png)
- 使用資料庫簡介
![](https://i.imgur.com/MUL6WjQ.png)
- 模型簡介
![](https://i.imgur.com/OReu7nO.png)
- 輸入資料方式
![](https://i.imgur.com/fEPXNAO.png)
- 手動輸入
![](https://i.imgur.com/FS89Ev9.png)
- 檔案輸入
![](https://i.imgur.com/aLkvZUw.png)
- 預測結果
![](https://i.imgur.com/FTcXB9j.png)
