import os
import cv2
import dlib
import pickle
import numpy as np
from tqdm import tqdm
import face_recognition
from pathlib import Path

class FaceModelTrainer:
    def __init__(self):
        self.face_detector = dlib.get_frontal_face_detector()
        self.known_face_encodings = []
        self.known_face_names = []
        self.model_path = "face_model.pkl"
        
    def train_from_folder(self, folder_path="training_faces"):
        """從資料夾訓練模型"""
        print(f"開始從 {folder_path} 訓練模型...")
        
        # 確保資料夾存在
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"已建立 {folder_path} 資料夾")
            print("請在資料夾中放入照片後重新執行程式")
            return False
            
        # 取得所有圖片檔案
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(Path(folder_path).glob(ext))
            
        if not image_files:
            print("未找到任何圖片檔案！")
            return False
            
        print(f"找到 {len(image_files)} 張圖片")
        
        # 處理每張圖片
        for img_path in tqdm(image_files, desc="處理圖片"):
            try:
                # 讀取圖片
                image = cv2.imread(str(img_path))
                if image is None:
                    print(f"無法讀取圖片: {img_path}")
                    continue
                    
                # 轉換顏色空間
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # 偵測人臉
                faces = self.face_detector(rgb_image)
                
                if len(faces) == 0:
                    print(f"在 {img_path.name} 中未偵測到人臉")
                    continue
                    
                if len(faces) > 1:
                    print(f"警告: {img_path.name} 中偵測到多個人臉，使用第一個")
                
                # 提取人臉特徵
                face_encoding = face_recognition.face_encodings(rgb_image)[0]
                
                # 儲存特徵和對應的名稱（使用檔名作為身份標識）
                self.known_face_encodings.append(face_encoding)
                self.known_face_names.append(img_path.stem)  # 使用檔名（不含副檔名）
                
            except Exception as e:
                print(f"處理 {img_path.name} 時發生錯誤: {str(e)}")
                
        # 儲存模型
        if self.known_face_encodings:
            self.save_model()
            print(f"\n成功訓練 {len(self.known_face_encodings)} 個人臉模型")
            return True
        else:
            print("\n沒有成功提取任何人臉特徵")
            return False
    
    def save_model(self):
        """儲存訓練結果"""
        model_data = {
            'encodings': self.known_face_encodings,
            'names': self.known_face_names
        }
        
        with open(self.model_path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"\n模型已儲存至 {self.model_path}")
    
    def validate_image(self, image_path):
        """驗證單張圖片"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return False, "無法讀取圖片"
            
            if image.size == 0:
                return False, "圖片是空的"
                
            min_size = 64  # 最小可接受的圖片尺寸
            if image.shape[0] < min_size or image.shape[1] < min_size:
                return False, f"圖片太小 (最小 {min_size}x{min_size} 像素)"
                
            return True, "圖片正常"
        except Exception as e:
            return False, f"圖片驗證失敗: {str(e)}"

def main():
    trainer = FaceModelTrainer()
    
    print("=== 人臉識別模型訓練程式 ===")
    print("請將要訓練的人臉照片放在 'training_faces' 資料夾中")
    print("照片檔名格式：人名.jpg (例如：student001.jpg)")
    print("每個人只需要一張清晰的正面照\n")
    
    input("準備好後請按 Enter 開始訓練...")
    
    if trainer.train_from_folder():
        print("\n訓練完成！")
        print("模型已儲存為 'face_model.pkl'")
        print("您可以使用這個模型檔案來進行人臉識別")
    else:
        print("\n訓練失敗，請檢查錯誤訊息並重試")

if __name__ == "__main__":
    main()
