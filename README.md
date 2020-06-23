# AKI Project
## MIMIC_Preprocessing.py
- 在code資料夾中的這份程式碼是用來處理MIMIC的資料集，包括從已建好的資料庫中抓不同feature的資料、處理缺失值、處理LSTM及Xgboost模型的input資料格式
- 參數設定:
    1. 設定要連接的資料庫:必須設定這五項參數:host, database, user, password, port。
    2. 抓取資料的時間範圍:可更改Main function中的TimeOffset的值，24代表抓取24小時的資料。
    3. 處理缺失值的方式:可選擇**Copy prev or next** or **Mean** or **Zero**，分別代表補前後、補平均、不補值。
- Output Data:
    - MIMIC_LSTM_Input.csv
    - MIMIC_Xgboost_Input.csv
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
