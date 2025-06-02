from aip import AipSpeech
import speech_recognition as sr


class SpeechRecognizer:
    def __init__(self, app_id, api_key, secret_key):
        self.APP_ID = app_id
        self.API_KEY = api_key
        self.SECRET_KEY = secret_key
        self.client = AipSpeech(self.APP_ID, self.API_KEY, self.SECRET_KEY)
        self.recognizer = sr.Recognizer()
        self.mic = sr.Microphone()
        self.listening = False  # 控制监听状态

    def get_text(self, wav_bytes):
        result = self.client.asr(wav_bytes, 'wav', 16000, {'dev_pid': 1537})
        try:
            text = result['result'][0]
        except Exception as e:
            print(f"Recognition error: {e}")
            text = ''
        return text

    def start_listening(self):
        self.listening = True
        with self.mic as source:
            self.recognizer.adjust_for_ambient_noise(source)
            while self.listening:
                try:
                    audio = self.recognizer.listen(source, timeout=5)
                    wav_data = audio.get_wav_data(convert_rate=16000)
                    text = self.get_text(wav_data)
                    yield text
                except sr.WaitTimeoutError:
                    continue
                except Exception as e:
                    print(f"Error during listening: {e}")
                    break

    def stop_listening(self):
        self.listening = False