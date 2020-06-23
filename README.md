# AKI Project
## MIMIC_Preprocessing.py
- 這份code是用來處理MIMIC的資料集，包括從已建好的資料庫中抓不同feature的資料、處理缺失值、處理LSTM及Xgboost模型的input資料的格式
- 參數設定:
    1. 設定要連接的資料庫:必須設定這五項參數:host, database, user, password, port。
    2. 抓取資料的時間範圍:可更改Main function中的TimeOffset的值，24代表抓取24小時的資料。
    3. 處理缺失值的方式:可選擇**Copy prev or next** or **Mean** or **Zero**，分別代表補前後、補平均、不補值
- Output:
    - MIMIC_LSTM_Input.csv
    - MIMIC_Xgboost_Input.csv
