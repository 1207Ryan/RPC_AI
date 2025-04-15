import json


def simplify_spacing_rules(data, default_values=None):
    """
    简化 JSON，每种参数保留一个合理的示例值

    :param data: 输入的JSON数据（Python字典）
    :param default_values: 自定义的默认值映射，例如 {"p-": "4", "px-": "2"}
    :return: 简化后的间距配置
    """
    spacing = data["spacing"]
    value_placeholders = data["value_placeholders"]

    # 默认的推荐值（可覆盖）
    default_values = default_values or {
        "p-": "4",  # 全局padding用4
        "px-": "2",  # 水平padding用2
        "py-": "3",  # 垂直padding用3
        "m-": "4",  # 全局margin用4
        "mx-": "auto",  # 水平margin优先用auto
        "space-x-": "4",
        "-m-": "2"  # 负margin用2
    }

    simplified_spacing = {}

    for category, patterns in spacing.items():
        simplified_spacing[category] = []
        for pattern in patterns:
            if "{number}" in pattern:
                # 查找最匹配的默认值
                value = "0"  # 保底值
                for prefix, default_val in default_values.items():
                    if pattern.startswith(prefix):
                        value = default_val
                        break
                simplified_spacing[category].append(pattern.replace("{number}", value))
            else:
                simplified_spacing[category].append(pattern)

    return {"spacing": simplified_spacing}


# 从文件读取配置
with open('config.json', 'r', encoding='utf-8') as f:
    input_data = json.load(f)

# 生成简化版（使用推荐默认值）
simplified_data = simplify_spacing_rules(input_data)

print("简化后的配置：")
print(json.dumps(simplified_data, indent=2, ensure_ascii=False))

# 也可以自定义默认值
custom_defaults = {"p-": "8", "px-": "4"}  # 更大的默认间距
custom_simplified = simplify_spacing_rules(input_data, custom_defaults)

# import time
# from datetime import datetime
# from typing import Dict, List, Optional, Union, Any
# import numpy as np
# import requests
# from sparkai.core.messages import ChatMessage
# from sparkai.llm.llm import ChatSparkLLM, ChunkPrintHandler
# from volcenginesdkarkruntime import Ark
# from vosk import Model, KaldiRecognizer, SetLogLevel
# import pyaudio
# import json
# import re
# import wave
# import json
# import glob
# import os
# import sys
# from thrift.protocol import TBinaryProtocol
# from thrift.transport import TSocket
# from thrift.transport import TTransport
# from thrift.server import TServer
# import logging