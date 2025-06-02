# speech_synthesizer.py

from aip import AipSpeech
import os
import tempfile
from playsound import playsound  # 改用 playsound 播放音频

class SpeechSynthesizer:
    """
    百度语音合成封装：给定一段文字，调用 TTS 生成本地临时 MP3 文件，并用 playsound 播放。
    """

    def __init__(self, app_id, api_key, secret_key):
        self.client = AipSpeech(app_id, api_key, secret_key)

    def text_to_speech(self, text, dst_path=None):
        """
        将 text 转为音频，默认保存到临时文件夹，返回音频文件完整路径。
        """
        if not text:
            return None

        result = self.client.synthesis(
            text,
            'zh',  # 中文；如果要读英文则改 'en'
            1,     # 发音人，0 为女声，1 为男声，3 为小童，4 为度逍遥
            {
                'vol': 5,  # 音量 0-15
                'spd': 5,  # 语速 0-9
                'pit': 5,  # 语调 0-9
                'per': 0   # 度小美
            }
        )
        if not isinstance(result, dict):
            if dst_path is None:
                fd, dst_path = tempfile.mkstemp(suffix=".mp3")
                os.close(fd)
            with open(dst_path, 'wb') as f:
                f.write(result)
            return dst_path
        else:
            print("TTS 合成失败：", result)
            return None

    def play(self, audio_path):
        """
        用 playsound 播放音频文件（MP3）。
        """
        if not audio_path or not os.path.exists(audio_path):
            return
        try:
            playsound(audio_path)
        except Exception as e:
            print("播放失败：", e)
