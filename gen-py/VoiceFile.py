from vosk import Model, KaldiRecognizer, SetLogLevel
import wave
import json

SetLogLevel(-1)

# 加载模型
model_path = "VoiceModelSmall"  # 替换为你的模型路径
model = Model(model_path)

results = []
with wave.open('我好冷.wav', 'rb') as wf:
    # 验证音频格式
    assert wf.getnchannels() == 1, "必须单声道音频"
    assert wf.getsampwidth() == 2, "必须16-bit PCM"

    rec = KaldiRecognizer(model, wf.getframerate())

    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            res = json.loads(rec.Result())
            results.append(res.get('text', ''))

    final = json.loads(rec.FinalResult())
    full_text = ' '.join(results + [final.get('text', '')])
    print("完整识别结果:", full_text)