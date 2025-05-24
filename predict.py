# predict.py
# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import string
import joblib

class SignLanguageClassifier:
    def __init__(self, model_path, input_size=(64, 64), roi_size=400):
        self.model = joblib.load(model_path)
        self.input_size = input_size  # 图像大小（与训练时保持一致）
        self.roi_size = roi_size

        self.class_names = list(string.ascii_uppercase)  # 'A'~'Z'
        self.frame_w = self.frame_h = None
        self.x0 = self.x1 = self.y0 = self.y1 = None

    def setup_roi(self):
        cx, cy = self.frame_w // 2, self.frame_h // 2
        h = self.roi_size // 2
        self.x0, self.x1 = max(cx - h, 0), min(cx + h, self.frame_w)
        self.y0, self.y1 = max(cy - h, 0), min(cy + h, self.frame_h)

    def preprocess_image(self, frame):
        """提取图像 ROI 区域灰度像素并展平为一维特征"""
        crop = frame[self.y0:self.y1, self.x0:self.x1]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        resized = cv2.resize(blur, self.input_size)
        norm = resized.astype('float32') / 255.0
        return norm.flatten().reshape(1, -1)

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("无法打开摄像头")
            return

        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头帧")
            return

        self.frame_h, self.frame_w = frame.shape[:2]
        self.setup_roi()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            feat = self.preprocess_image(frame)
            pred_idx = self.model.predict(feat)[0]
            label = self.class_names[pred_idx]

            # 绘图
            cv2.rectangle(frame, (self.x0, self.y0), (self.x1, self.y1), (0, 255, 0), 2)
            cv2.putText(frame, label, (self.x0 + 10, self.y0 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

            cv2.imshow("Sign Language Recognition", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q') or k == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    model_path = r"E:\360MoveData\Users\HYT\Desktop\AI_Interactive_Practice\Project\rf.pkl"
    clf = SignLanguageClassifier(model_path)
    clf.run()
