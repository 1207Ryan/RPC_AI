from vosk import Model, KaldiRecognizer, SetLogLevel
import pyaudio
import json

SetLogLevel(-1)


def VoiceRecognition():
    # 加载模型
    model = Model('VoiceModelSmall')
    # 麦克风
    microphone = pyaudio.PyAudio()

    stream = microphone.open(
        format=pyaudio.paInt16,  # 16位深度音频设置
        channels=1,  # 声道,单声道
        rate=16000,  # 采样率
        input=True,  # 从麦克风获取数据
        frames_per_buffer=4000  # 每次读取数据块大小
    )
    # 语音识别器
    recognizer = KaldiRecognizer(model, 16000)  # 模型，采样率

    print("开始语音识别，请说话...")
    silence_count = 0
    max_silence = 3  # 最大静音次数阈值

    while True:
        # 从麦克风读取数据
        data = stream.read(4000)

        if recognizer.AcceptWaveform(data):
            # 获取识别结果
            result = json.loads(recognizer.Result())
            text = result.get('text', '').replace(' ', '')

            if text:  # 如果有识别结果
                print(f"识别到: {text}")
                # 关闭麦克风和pyaudio实例
                stream.stop_stream()
                stream.close()
                microphone.terminate()
                return text
            else:
                silence_count += 1
                if silence_count >= max_silence:
                    print("长时间未检测到语音，退出识别")
                    break
        else:
            # 获取部分结果
            partial_result = json.loads(recognizer.PartialResult())
            partial_text = partial_result.get('partial', '').replace(' ', '')
            if partial_text:
                print(f"正在识别: {partial_text}", end='\r')

    # 如果没有识别到任何内容
    stream.stop_stream()
    stream.close()
    microphone.terminate()
    return None
