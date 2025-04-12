import re
from vosk import Model, KaldiRecognizer, SetLogLevel
import wave
import json
import glob
import os

SetLogLevel(-1)


def find_audio(text, folder_path):
    """支持查找.wav/.mp3等格式，并返回规范化路径"""
    for ext in ['.wav', '.mp3', '.ogg']:
        match = glob.glob(os.path.join(folder_path, f"{text}{ext}"))
        if match:
            return os.path.normpath(match[0])  # 规范化路径（自动去除./）
    return None


def smart_split(cmd):
    """
    综合方案：优先用人工规则，失败后尝试正则
    """
    # 人工规则词表
    actions = ["识别", "播放", "打开", "关闭"]
    targets = ["语音文件", "视频", "空调", "灯光"]

    # 尝试精确匹配
    for action in actions:
        if cmd.startswith(action):
            remaining = cmd[len(action):]
            for target in targets:
                if remaining.startswith(target):
                    feeling = remaining[len(target):].strip()
                    return action, target, feeling or None

    # 失败后使用正则兜底
    match = re.match(r"^(\S{2})(\S{4})(.*)$", cmd)
    if match:
        return match.groups()

    return (None, None, None)


def VoiceFileRecognition(wav_file):
    # 加载模型
    model_path = "VoiceModelSmall"  # 替换为你的模型路径
    model = Model(model_path)

    results = []
    with wave.open(wav_file, 'rb') as wf:
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
        return full_text


if __name__ == '__main__':
    content = smart_split("识别语音文件我好冷")
    voicefile = find_audio(content[2], "./")
    result = VoiceFileRecognition(voicefile)
    print(result)
