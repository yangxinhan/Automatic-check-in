import cv2
import face_recognition
import numpy as np
from datetime import datetime 
import csv
import os
import time
import dlib
import urllib.request
import bz2

class FaceRecognitionSystem:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.attendance_file = "attendance.csv"
        self.last_attendance = {}
        self.face_detector = dlib.get_frontal_face_detector()
        
        # 檢查並下載特徵點模型
        self.shape_predictor_path = "shape_predictor_68_face_landmarks.dat"
        if not os.path.exists(self.shape_predictor_path):
            self.download_shape_predictor()
        
        self.shape_predictor = dlib.shape_predictor(self.shape_predictor_path)
        
        if not os.path.exists(self.attendance_file):
            with open(self.attendance_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Name', 'Time'])
    
    def download_shape_predictor(self):
        """下載並解壓縮特徵點模型"""
        print("下載特徵點模型中...")
        url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
        compressed_path = self.shape_predictor_path + ".bz2"
        
        # 下載壓縮檔
        urllib.request.urlretrieve(url, compressed_path)
        
        # 解壓縮檔案
        print("解壓縮模型檔案...")
        with bz2.open(compressed_path) as fr, open(self.shape_predictor_path, 'wb') as fw:
            fw.write(fr.read())
        
        # 刪除壓縮檔
        os.remove(compressed_path)
        print("特徵點模型準備完成！")

    def load_known_faces(self, faces_dir="known_faces"):
        try:
            if not os.path.exists(faces_dir):
                os.makedirs(faces_dir)
                print(f"已建立 {faces_dir} 資料夾")
                return
            
            for filename in os.listdir(faces_dir):
                if filename.endswith((".jpg", ".png")):
                    image_path = os.path.join(faces_dir, filename)
                    try:
                        # 使用 dlib 直接處理影像
                        image = cv2.imread(image_path)
                        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        faces = self.face_detector(rgb_image)
                        
                        if len(faces) > 0:
                            shape = self.shape_predictor(rgb_image, faces[0])
                            face_descriptor = np.array(face_recognition.face_encodings(rgb_image)[0])
                            self.known_face_encodings.append(face_descriptor)
                            self.known_face_names.append(os.path.splitext(filename)[0])
                            print(f"成功載入 {filename}")
                    except Exception as e:
                        print(f"載入 {filename} 失敗: {str(e)}")
        except Exception as e:
            print(f"載入人臉資料時發生錯誤: {str(e)}")

    def mark_attendance(self, name):
        current_time = time.time()
        if name in self.last_attendance and current_time - self.last_attendance[name] < 60:
            return
        
        self.last_attendance[name] = current_time
        try:
            now = datetime.now()
            time_string = now.strftime('%Y-%m-%d %H:%M:%S')
            with open(self.attendance_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([name, time_string])
                print(f"{name} 簽到成功: {time_string}")
        except Exception as e:
            print(f"記錄簽到時發生錯誤: {str(e)}")
    
    def run(self):
        try:
            video_capture = cv2.VideoCapture(2)
            if not video_capture.isOpened():
                raise Exception("無法開啟攝影機")
            
            print("系統啟動中...")
            
            while True:
                ret, frame = video_capture.read()
                if not ret:
                    print("無法獲取影像")
                    time.sleep(1)
                    continue
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = self.face_detector(rgb_frame)
                
                for face in faces:
                    shape = self.shape_predictor(rgb_frame, face)
                    face_descriptor = np.array(face_recognition.face_encodings(rgb_frame)[0])
                    
                    matches = []
                    for known_encoding in self.known_face_encodings:
                        match = face_recognition.compare_faces([known_encoding], face_descriptor)[0]
                        matches.append(match)
                    
                    name = "未知"
                    if True in matches:
                        first_match_index = matches.index(True)
                        name = self.known_face_names[first_match_index]
                        self.mark_attendance(name)
                    
                    left = face.left()
                    top = face.top()
                    right = face.right()
                    bottom = face.bottom()
                    
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(frame, name, (left, bottom + 30), 
                              cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 1)
                
                cv2.imshow('人臉辨識簽到系統', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            video_capture.release()
            cv2.destroyAllWindows()
            
        except Exception as e:
            print(f"系統執行時發生錯誤: {str(e)}")
        finally:
            if 'video_capture' in locals():
                video_capture.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    system = FaceRecognitionSystem()
    system.load_known_faces()
    system.run()