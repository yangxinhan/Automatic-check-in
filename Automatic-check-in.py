import cv2
import face_recognition  # 添加這行
import numpy as np
from datetime import datetime 
import csv
import os
import time
import dlib
import pickle

class FaceRecognitionSystem:
    def __init__(self):
        self.attendance_file = "attendance.csv"
        self.model_file = "face_model.pkl"
        self.last_attendance = {}
        self.present_people = {}
        self.face_detector = dlib.get_frontal_face_detector()
        
        # 載入訓練好的模型
        if not os.path.exists(self.model_file):
            raise Exception("找不到模型檔案，請先執行 train_model.py 訓練模型")
            
        with open(self.model_file, 'rb') as f:
            model_data = pickle.load(f)
            self.known_face_encodings = model_data['encodings']
            self.known_face_names = model_data['names']
            print(f"已載入 {len(self.known_face_names)} 個人臉模型")
        
        # 建立或檢查CSV檔案
        if not os.path.exists(self.attendance_file):
            with open(self.attendance_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Name', 'Check_In_Time', 'Check_Out_Time', 'Status'])

    def mark_attendance(self, name):
        current_time = time.time()
        
        # 檢查是否在短時間內重複偵測
        if name in self.last_attendance:
            if current_time - self.last_attendance[name] < 30:  # 30秒內不重複記錄
                return
        
        now = datetime.now()
        time_string = now.strftime('%Y-%m-%d %H:%M:%S')
        
        try:
            # 檢查是否已在場內
            if name in self.present_people:
                # 執行簽退
                check_in_time = self.present_people[name]
                with open(self.attendance_file, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([name, check_in_time, time_string, "簽退"])
                del self.present_people[name]
                print(f"{name} 簽退成功: {time_string}")
            else:
                # 執行簽到
                self.present_people[name] = time_string
                with open(self.attendance_file, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([name, time_string, "", "簽到"])
                print(f"{name} 簽到成功: {time_string}")
            
            self.last_attendance[name] = current_time
            
        except Exception as e:
            print(f"記錄考勤時發生錯誤: {str(e)}")

    def recognize_face(self, face_encoding):
        """比對人臉特徵"""
        matches = []
        distances = []
        
        for known_encoding in self.known_face_encodings:
            # 計算特徵向量距離
            distance = np.linalg.norm(known_encoding - face_encoding)
            distances.append(distance)
            # 設定閾值，距離小於0.6視為相同人臉
            matches.append(distance < 0.6)
        
        if True in matches:
            # 找出最匹配的人臉
            best_match_idx = distances.index(min(distances))
            return self.known_face_names[best_match_idx]
        return None

    def run(self):
        try:
            video_capture = cv2.VideoCapture(2)
            if not video_capture.isOpened():
                raise Exception("無法開啟攝影機")
            
            # 設定攝影機參數
            video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            video_capture.set(cv2.CAP_PROP_FPS, 15)
            
            print("系統啟動中...")
            frame_count = 0
            
            while True:
                ret, frame = video_capture.read()
                if not ret:
                    print("無法獲取影像")
                    time.sleep(1)
                    continue
                
                # 縮小影像加快處理速度
                small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                
                # 每3幀處理一次
                if frame_count % 3 == 0:
                    # 偵測人臉
                    face_locations = self.face_detector(rgb_small_frame)
                    
                    for face in face_locations:
                        try:
                            # 提取人臉特徵
                            face_encoding = np.array(face_recognition.face_encodings(rgb_small_frame)[0])
                            
                            # 識別人臉
                            name = self.recognize_face(face_encoding)
                            if name:
                                self.mark_attendance(name)
                            else:
                                name = "未知"
                            
                            # 繪製框框和名字
                            left = face.left() * 2
                            top = face.top() * 2
                            right = face.right() * 2
                            bottom = face.bottom() * 2
                            
                            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                            cv2.putText(frame, name, (left, bottom + 30), 
                                      cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 1)
                        except Exception as e:
                            print(f"處理人臉時發生錯誤: {str(e)}")
                            continue
                
                frame_count += 1
                cv2.imshow('人臉辨識簽到系統', frame)
                
                if cv2.waitKey(30) & 0xFF == ord('q'):
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
    try:
        system = FaceRecognitionSystem()
        system.run()
    except Exception as e:
        print(f"程式執行失敗: {str(e)}")
        print("請確認是否已執行 train_model.py 訓練模型")