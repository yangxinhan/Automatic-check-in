import os
import cv2
import dlib
import pickle
import numpy as np
from tqdm import tqdm
import face_recognition
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class FaceModelTrainer:
    def __init__(self):
        self.face_detector = dlib.get_frontal_face_detector()
        self.known_face_encodings = []
        self.known_face_names = []
        self.model_path = "face_model.pkl"
        self.label_encoder = LabelEncoder()
        self.cnn_model = self.build_advanced_model()

    def build_advanced_model(self):
        """建立進階的深度學習模型"""
        model = models.Sequential([
            layers.Input(shape=(128,)),  # 人臉特徵向量的維度
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.BatchNormalization(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

    def augment_image(self, image):
        """使用 OpenCV 進行簡單的圖像增強"""
        augmented_images = []
        
        # 水平翻轉
        flipped = cv2.flip(image, 1)
        augmented_images.append(flipped)
        
        # 調整亮度
        bright = cv2.convertScaleAbs(image, alpha=1.2, beta=10)
        dark = cv2.convertScaleAbs(image, alpha=0.8, beta=-10)
        augmented_images.extend([bright, dark])
        
        return augmented_images

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
        
        # 收集和增強數據
        face_encodings = []
        labels = []
        
        for img_path in tqdm(image_files, desc="處理圖片"):
            try:
                image = cv2.imread(str(img_path))
                if image is None:
                    print(f"無法讀取圖片: {img_path}")
                    continue
                
                # 基本預處理
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                faces = self.face_detector(rgb_image)
                
                if len(faces) == 0:
                    print(f"在 {img_path.name} 中未偵測到人臉")
                    continue
                
                # 原始特徵
                face_encoding = face_recognition.face_encodings(rgb_image)[0]
                face_encodings.append(face_encoding)
                labels.append(img_path.stem)
                
                # 資料增強
                augmented_images = self.augment_image(rgb_image)
                for aug_image in augmented_images:
                    try:
                        aug_encoding = face_recognition.face_encodings(aug_image)[0]
                        face_encodings.append(aug_encoding)
                        labels.append(img_path.stem)
                    except:
                        continue
                
            except Exception as e:
                print(f"處理 {img_path.name} 時發生錯誤: {str(e)}")
        
        if not face_encodings:
            print("\n沒有成功提取任何人臉特徵")
            return False
            
        # 轉換數據格式
        X = np.array(face_encodings)
        encoded_labels = self.label_encoder.fit_transform(labels)
        y = np.array(encoded_labels)
        
        # 分割訓練和驗證數據
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, 
            test_size=0.2,
            random_state=42,
            stratify=y if len(np.unique(y)) > 1 else None
        )

        # 訓練模型
        print("\n開始訓練深度學習模型...")
        history = self.cnn_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=32,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.2,
                    patience=3
                )
            ],
            verbose=1
        )
        
        # 儲存模型和訓練結果
        self.save_model(face_encodings, labels, history)
        return True

    def save_model(self, face_encodings, labels, history):
        """儲存模型和訓練數據"""
        model_data = {
            'encodings': face_encodings,
            'names': labels,
            'training_history': history.history,
            'label_encoder': self.label_encoder
        }
        
        with open(self.model_path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"\n模型已儲存至 {self.model_path}")
        
        # 儲存深度學習模型
        self.cnn_model.save('face_recognition_model')

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