# 自動人臉識別考勤系統

## 專案簡介

這是一個基於人工智慧的自動考勤系統，運用電腦視覺和深度學習技術，實現無接觸式自動點名。系統透過攝影機即時捕捉影像，自動識別人臉並記錄出勤情況，大幅提升考勤效率並減少人為干預。

## 核心功能

- 🎥 **即時人臉偵測**：使用OpenCV即時捕捉和偵測人臉
- 👤 **身份識別**：運用深度學習模型進行人臉比對和身份確認
- ⚡ **快速簽到**：自動記錄出勤時間，無需手動操作
- 📊 **數據管理**：完整的考勤記錄儲存和查詢功能
- 📱 **多平台支援**：支援各類攝影裝置，適用於不同場景

## 技術架構

### 前端技術

- HTML5/CSS3/JavaScript
- Bootstrap 響應式設計
- Vue.js/React 前端框架

### 後端技術

- Python 3.9+
- OpenCV 人臉偵測
- face_recognition 人臉識別
- Flask/Django Web框架
- SQLite/MySQL 資料庫

### 硬體需求

- 網路攝影機（建議1080p以上）
- CPU: Intel i5/AMD Ryzen 5 或更高
- RAM: 8GB 以上
- 儲存空間: 256GB SSD

## 專案部署與安裝

### 環境需求

- Python 3.9+
- pip 套件管理工具
- 網路攝影機（建議1080p以上）
- CPU: Intel i5/AMD Ryzen 5 或更高
- RAM: 8GB 以上
- 儲存空間: 256GB SSD

### 安裝步驟

1. 下載並安裝相依套件：
```bash
pip install -r requirements.txt
```

2. 下載面部特徵點模型：
```bash
curl -L "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" -o shape_predictor_68_face_landmarks.dat.bz2
bunzip2 shape_predictor_68_face_landmarks.dat.bz2
```

3. 建立必要資料夾：
```
mkdir known_faces
mkdir logs
```

## 使用說明

### 人員註冊

1. 將人員放入 #known_faces 資料夾
2. 照片命名格式：`姓名.jpg`
3. 每張照片須為清晰正面照

### 系統操作

1. 啟動系統：
```bash
python main.py
```

2. 系統功能：
* 即時顯示攝影機畫面
* 自動框選並識別人臉
* 顯示辨識結果與簽到狀態
* 按 'q' 鍵結束程式

### 簽到紀錄
* 儲存位置：`attendance.csv`
* 記錄格式：姓名、日期時間
* 可使用 Excel 開啟查看

## 常見問題

1. 攝影機無法開啟
* 檢查攝影機連接狀態
* 確認權限設定
* 重新啟動程式

2. 人臉辨識失敗
* 確保照明充足
* 調整攝影機角度
* 更新人員照片

3. 系統效能問題
* 降低影像解析度
* 關閉不必要程式
* 升級硬體配置