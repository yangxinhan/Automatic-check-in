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
from albumentations import (
    Compose, RandomBrightnessContrast, RandomRotate90,
    HorizontalFlip, VerticalFlip, ShiftScaleRotate, GaussNoise
)

class FaceModelTrainer:
    def __init__(self):
        self.face_detector = dlib.get_frontal_face_detector()
        self.known_face_encodings = []
        self.known_face_names = []
        self.model_path = "face_model.pkl"
        self.label_encoder = LabelEncoder()  # 添加這行初始化
        
        self.augmentation = Compose([
            RandomBrightnessContrast(p=0.5),
            HorizontalFlip(p=0.3),
            ShiftScaleRotate(p=0.3, shift_limit=0.1, scale_limit=0.1, rotate_limit=15),
            GaussNoise(p=0.2)
        ])
        
        # 建立進階的深度學習模型
        self.cnn_model = self.build_advanced_model()
        
    def build_advanced_model(self):
        """建立進階的深度學習模型"""
        # 修改模型架構以適應128維的人臉特徵向量
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
        """數據增強"""
        augmented = self.augmentation(image=image)
        return augmented['image']

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
        augmented_encodings = []
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
                
                # 資料增強 - 每張照片產生3個增強版本
                for _ in range(3):
                    augmented_image = self.augment_image(rgb_image)
                    try:
                        aug_encoding = face_recognition.face_encodings(augmented_image)[0]
                        augmented_encodings.append(aug_encoding)
                        labels.append(img_path.stem)
                    except:
                        continue
                
            except Exception as e:
                print(f"處理 {img_path.name} 時發生錯誤: {str(e)}")
        
        # 合併原始和增強的特徵
        all_encodings = face_encodings + augmented_encodings
        
        if not all_encodings:
            print("\n沒有成功提取任何人臉特徵")
            return False
            
        # 轉換數據格式
        X = np.array(all_encodings)
        encoded_labels = self.label_encoder.fit_transform(labels)
        y = np.array(encoded_labels)
        
        # 修改分割比例並確保每個類別都有足夠的樣本
        if len(np.unique(y)) > 5:
            test_size = 0.1  # 如果類別較多，使用較小的測試集
        else:
            test_size = 0.2

        try:
            # 分割訓練和驗證數據
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, 
                test_size=test_size,
                random_state=42,
                stratify=y if len(np.unique(y)) > 1 else None
            )
        except ValueError as e:
            print("警告：因為樣本數量限制，跳過資料分割，直接使用全部數據訓練")
            X_train, y_train = X, y
            X_val, y_val = X, y

        # 訓練模型
        print("\n開始訓練深度學習模型...")
        
        # 確保輸入數據的維度正確
        X_train = np.array(X_train).astype('float32')
        if X_val is not None:
            X_val = np.array(X_val).astype('float32')
        
        # 確保標籤是整數類型
        y_train = np.array(y_train).astype('int32')
        if y_val is not None:
            y_val = np.array(y_val).astype('int32')
        
        # 打印模型摘要
        print("\n模型結構：")
        self.cnn_model.summary()
        
        # 打印訓練數據形狀
        print(f"\n訓練數據形狀：")
        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")
        
        # 根據數據量調整批次大小
        batch_size = min(32, len(X_train))
        
        # 使用回調函數
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='loss' if X_val is None else 'val_loss',
                patience=5,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='loss' if X_val is None else 'val_loss',
                factor=0.2,
                patience=3
            )
        ]
        
        # 如果驗證集和訓練集相同，則不使用驗證資料
        validation_data = (X_val, y_val) if X_val is not None and not np.array_equal(X_train, X_val) else None
        
        history = self.cnn_model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=100,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # 儲存模型和訓練結果
        self.save_advanced_model(face_encodings, labels, history)
        
        return True

    def save_advanced_model(self, face_encodings, labels, history):
        """儲存增強的模型和訓練數據"""
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

opencv-python-headless