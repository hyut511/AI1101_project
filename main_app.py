# main_app.py

import sys
import cv2
import numpy as np
import joblib
import mediapipe as mp
import time

from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap

from Ui_myui import Ui_MainWindow
from speech_recognizer import SpeechRecognizer
from sentence_generator import generate_sentence_from_letters


class CameraThread(QThread):
    """
    该线程完成两件事：
    1. 从摄像头不断抓帧并通过 frame_signal 发给主线程显示在 QLabel（左上）；
    2. 对抓到的每一帧调用 MediaPipe + 随机森林预测当前手势字母，
       并通过 gesture_signal 发给主线程更新右上角 label_2。
    """
    frame_signal = pyqtSignal(QImage)
    gesture_signal = pyqtSignal(str, str, str)  # (current_label, sequence_str, result_sentence)

    def __init__(self, model_path, parent=None):
        super().__init__(parent)
        # 加载模型及 MediaPipe Hands
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

        self._running = False
        self.cap = None

        # 用于判断同一字母连续2秒
        self.current_label = "No Hand"
        self.label_start_time = 0

        # 拼接序列与结果句子
        self.sequence = []
        self.result_sentence = ""

    def extract_features(self, frame):
        """
        从一帧 BGR 图像中提取 MediaPipe 关键点 + 方向向量特征。
        如果没有检测到手，返回 None。
        """
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
        # 打开摄像头
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("无法打开摄像头")
            return

        self._running = True
        self.current_label = "No Hand"
        self.label_start_time = time.time()
        self.sequence = []
        self.result_sentence = ""

        cv2.namedWindow("MediaPipe Gestures (按 'c' 生成句子，按 'q'/ESC 退出)")

        while self._running:
            ret, frame = self.cap.read()
            if not ret:
                break

            # —— 手势识别 —— #
            feat = self.extract_features(frame)
            detected = "No Hand"
            if feat is not None:
                pred = self.model.predict(feat)[0]
                detected = self.class_names[pred]

            # 重置计时或累积
            if detected != self.current_label:
                self.current_label = detected
                self.label_start_time = time.time()
            else:
                if (self.current_label != "No Hand" and
                    (time.time() - self.label_start_time) > 2):
                    # 持续2秒才算检测到该字母
                    self.sequence.append(self.current_label)
                    # 重置计时
                    self.label_start_time = time.time()

            # 绘制文字到帧（可选，用于本地调试）
            cv2.putText(frame, f"Current: {detected}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Seq: {''.join(self.sequence)}", (30, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(frame, f"Result: {self.result_sentence}", (30, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

            # 转为 QImage 并发信号给主线程显示
            rgb_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_display.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_display.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.frame_signal.emit(qt_image)

            # 监听键盘：按 'c' 生成句子，按 'q'/ESC 退出
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                # 如果 sequence 非空，调用 OpenAI 接口生成句子
                if self.sequence:
                    # 生成句子（示例用本地函数，可替换为实际 API 调用）
                    self.result_sentence = generate_sentence_from_letters(self.sequence)
                    self.sequence = []
            elif key in (ord('q'), 27):
                self._running = False
                break

            # 发信号给主线程更新右上角 label_2
            self.gesture_signal.emit(
                detected,
                "".join(self.sequence),
                self.result_sentence
            )

        # 释放资源
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

    def stop(self):
        self._running = False
        self.wait()


class SpeechThread(QThread):
    """
    用于在后台循环执行语音识别，将结果通过信号传给主线程。
    """
    speech_signal = pyqtSignal(str)

    def __init__(self, recognizer, parent=None):
        super().__init__(parent)
        self.recognizer = recognizer
        self._running = False

    def run(self):
        self._running = True
        for text in self.recognizer.start_listening():
            if not self._running:
                break
            self.speech_signal.emit(text)

    def stop(self):
        self._running = False
        self.recognizer.stop_listening()
        self.wait()


class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # —— 语音识别初始化 —— #
        APP_ID = '118123310'
        API_KEY = 'v1OwFE3KnBbcN5Mi5YaG3VCI'
        SECRET_KEY = 'lkxmyECj2cDjixsurHpkvaP62RxtVxHB'
        self.speech_recognizer = SpeechRecognizer(APP_ID, API_KEY, SECRET_KEY)
        self.speech_thread = None

        # 绑定“开始/结束 语音识别”按钮
        self.ui.pushButton_3.clicked.connect(self.on_start_speech)
        self.ui.pushButton_4.clicked.connect(self.on_stop_speech)

        # —— 摄像头 + 手势识别线程 初始化 —— #
        model_path = "mediapipe_rf.pkl"  # 确保该文件在本目录下
        self.camera_thread = CameraThread(model_path)
        self.camera_thread.frame_signal.connect(self.update_camera_label)
        self.camera_thread.gesture_signal.connect(self.update_gesture_label)

        # 绑定“开启/关闭 摄像头”按钮
        self.ui.pushButton.clicked.connect(self.on_open_camera)
        self.ui.pushButton_2.clicked.connect(self.on_close_camera)

        # 初始化界面提示
        self.ui.label.setText("等待摄像头开启")    # 左上 摄像头预览
        self.ui.label_2.setText("手势识别文本")   # 右上 手势识别
        self.ui.label_3.setText("语音识别文本")   # 右下 语音识别

    # —— 摄像头相关槽函数 —— #
    def on_open_camera(self):
        if not self.camera_thread.isRunning():
            self.ui.label.setText("")
            self.camera_thread.start()

    def on_close_camera(self):
        if self.camera_thread.isRunning():
            self.camera_thread.stop()
            self.ui.label.clear()
            self.ui.label.setText("等待摄像头开启")
            self.ui.label_2.setText("手势识别文本")

    def update_camera_label(self, qt_image):
        pix = QPixmap.fromImage(qt_image).scaled(
            self.ui.label.width(), self.ui.label.height(), Qt.KeepAspectRatio
        )
        self.ui.label.setPixmap(pix)

    def update_gesture_label(self, current, sequence, result_sentence):
        """
        更新右上角 label_2 的文本：
        current: 当前识别字母
        sequence: 累积的字母序列
        result_sentence: 按 'c' 生成的句子
        """
        display_text = (
            f"Current: {current}\n"
            f"Sequence: {sequence}\n"
            f"Result: {result_sentence}"
        )
        self.ui.label_2.setText(display_text)

    # —— 语音识别相关槽函数 —— #
    def on_start_speech(self):
        if self.speech_thread is None or not self.speech_thread.isRunning():
            self.ui.label_3.setText("语音识别中...")
            self.speech_thread = SpeechThread(self.speech_recognizer)
            self.speech_thread.speech_signal.connect(self.update_speech_label)
            self.speech_thread.start()

    def on_stop_speech(self):
        if self.speech_thread and self.speech_thread.isRunning():
            self.speech_thread.stop()
            self.ui.label_3.setText("语音识别已停止")

    def update_speech_label(self, text):
        """
        更新右下角 label_3 显示最新的语音识别文本
        """
        self.ui.label_3.setText(f"识别结果: {text}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())
