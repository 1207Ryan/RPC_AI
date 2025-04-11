import wave

with wave.open('今天又是元气满满的一天，要打起精神来哟.wav', 'rb') as wf:
    print(f"声道数: {wf.getnchannels()}")
    print(f"采样宽度: {wf.getsampwidth()}")  # 应该为2(16-bit)
    print(f"采样率: {wf.getframerate()}")  # 应该与模型匹配(通常16000)