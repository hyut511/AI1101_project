# predict_mediapipe.py
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import joblib
import mediapipe as mp
import string
import time
from sentence_generator import generate_sentence_from_letters

class MediaPipePredictor:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)
        self.class_names = [
            'A','B','C','D','E','F','G','H','I','K',
            'L','M','N','O','P','Q','R','S','T','U',
            'V','W','X','Y'
        ]
        mp_hands = mp.solutions.hands
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

        self.last_label = None
        self.last_time = 0
        self.sequence = []      # 累积的字母列表
        self.result_sentence = []  # 生成的句子

    def extract_features(self, frame, roi=None):
        if roi:
            x0, y0, x1, y1 = roi
            frame = frame[y0:y1, x0:x1]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)
        if not result.multi_hand_landmarks:
            return None
        landmarks = result.multi_hand_landmarks[0].landmark
        keypoints = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
        wrist = keypoints[0:3]
        middle_mcp = keypoints[9*3:9*3+3]
        direction = middle_mcp - wrist
        direction = direction / (np.linalg.norm(direction) + 1e-6)
        feature = np.concatenate([keypoints, direction])
        return feature.reshape(1, -1)

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("无法打开摄像头")
            return

        print("按 'c' 键将当前字母序列发送给Qwen，生成句子；按 'q' 或 ESC 退出。")

        # 初始化状态
        self.sequence = []
        self.result_sentence = []
        current_label = None
        label_start_time = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            feat = self.extract_features(frame)
            detected = "No Hand"

            if feat is not None:
                pred = self.model.predict(feat)[0]
                detected = self.class_names[pred]

            # 如果检测到字母与上次不同，重置计时
            if detected != current_label:
                current_label = detected
                label_start_time = time.time()
            else:
                # 相同字母持续超过2秒且不是 "No Hand"，才记录
                if current_label != "No Hand" and (time.time() - label_start_time) > 2:
                    self.sequence.append(current_label)
                    print(f"Recorded: {current_label}, sequence now: {''.join(self.sequence)}")
                    # 重置开始时间，避免重复记录
                    label_start_time = time.time()

            # 在画面上显示信息
            cv2.putText(frame, f"Current: {detected}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Sequence: {''.join(self.sequence)}", (30, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(frame, f"Result: {self.result_sentence}", (30, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

            cv2.imshow("MediaPipe Sign Language Prediction", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('c'):  # 触发调用 OpenAI
                if self.sequence:
                    print("Sending to OpenAI:", "".join(self.sequence))
                    self.result_sentence.append(generate_sentence_from_letters(self.sequence))
                    self.sequence = []

            if key in (ord('q'), 27):  # 'q' 或 ESC 退出
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    model_path = "./mediapipe_rf.pkl"
    predictor = MediaPipePredictor(model_path)
    predictor.run()
    print(predictor.result_sentence)