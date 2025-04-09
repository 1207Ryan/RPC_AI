import os
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Union
from dataclasses import field
import requests
from prompts import *
from sparkai.llm.llm import ChatSparkLLM, ChunkPrintHandler
from sparkai.core.messages import ChatMessage
from volcenginesdkarkruntime import Ark
from voice import VoiceRecognition


class DialogHistory:
    def __init__(self, max_length: int = 3):
        self.history: List[Dict] = []
        self.max_length = max_length
        self.current_scene: Optional[str] = None  # 明确标注可为None
        self.scene_persist_counter = 0  # 场景持续计数器

    def add(self, user_input: str, response: list) -> None:
        self.history.append({
            "user": user_input,
            "system": response,
            "timestamp": time.time()
        })
        if len(self.history) > self.max_length:
            self.history.pop(0)

    def get_context(self) -> str:
        return "\n======\n".join(
            f"用户:{item['user']}\n系统:{item['system']}"
            for item in self.history
        )

    def detect_scene(self, user_input: str) -> Optional[str]:
        """实时场景检测（单条触发机制）"""
        if self.current_scene is not None:
            self.scene_persist_counter -= 1  # 每轮无匹配递减
            return self.current_scene

        # 清空场景的条件（计数器归零或没有设备匹配）
        if self.scene_persist_counter <= 0:
            self.current_scene = None

        # 实时关键词检测
        detected_scene = None
        for scene, keywords in SCENE_KEYWORDS.items():
            if any(keyword in user_input for keyword in keywords):
                detected_scene = scene
                break

        # 更新场景状态
        if detected_scene:
            self.current_scene = detected_scene
            self.scene_persist_counter = self.max_length  # 设置几轮对话的持续期

        return self.current_scene

    def force_exit_scene(self):
        self.current_scene = None


history = DialogHistory()


class UserProfile:
    # 基础信息
    age: Optional[int]
    gender: Optional[str]
    region: Optional[str]
    # 家庭信息
    family_members: int
    has_children: bool
    has_elderly: bool
    has_pet: bool
    #生活习惯
    work_schedule: str  # "regular", "night_shift", "flexible"
    cooking_habits: str  # "rare", "medium", "frequent"
    device_usage: Dict[str, int] = field(default_factory=dict)  # 设备使用次数统计

    def __init__(self, age: Optional[int] = 20, gender: Optional[str] = "male",
                 region: Optional[str] = "south", family_members: Optional[int] = 1,
                 has_children: Optional[bool] = False, has_elderly: Optional[bool] = False,
                 has_pet: Optional[bool] = False, work_schedule: Optional[str] = "regular",
                 cooking_habits: Optional[str] = "medium", device_usage: Dict[str, int] = None):
        self.age = age
        self.gender = gender
        self.region = region
        self.family_members = family_members
        self.has_children = has_children
        self.has_elderly = has_elderly
        self.has_pet = has_pet
        self.work_schedule = work_schedule
        self.cooking_habits = cooking_habits
        self.device_usage = device_usage

    def record_device_usage(self, device_name: str):
        """记录设备使用次数"""
        self.device_usage[device_name] = self.device_usage.get(device_name, 0) + 1

    def save_to_file(self, filepath: str):
        """保存完整用户数据"""
        data = {
            "basic_info": {
                "age": self.age,
                "gender": self.gender,
                "region": self.region,
            },
            "family_info": {
                "family_members": self.family_members,
                "has_children": self.has_children,
                "has_elderly": self.has_elderly,
                "has_pet": self.has_pet,
            },
            "device_data": {
                "work_schedule": self.work_schedule,
                "cooking_habits": self.cooking_habits,
                "usage": self.device_usage,
            }
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load_from_file(cls, filepath: str):
        """从文件加载完整数据"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return cls(
                    age=data["basic_info"].get("age"),
                    gender=data["basic_info"].get("gender"),
                    region=data["basic_info"].get("region"),
                    family_members=data["family_info"].get("family_members", 1),
                    has_children=data["family_info"].get("has_children", False),
                    has_elderly=data["family_info"].get("has_elderly", False),
                    has_pet=data["family_info"].get("has_pet", False),
                    work_schedule=data["family_info"].get("work_schedule", None),
                    cooking_habits=data["family_info"].get("cooking_habits", None),
                    device_usage=data["device_data"].get("usage", {})
                )
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            return cls()  # 返回默认配置


def get_seasonal_context() -> Dict[str, str]:
    """Get current season and time of day"""
    now = datetime.now()
    month = now.month
    hour = now.hour
    weekday_num = now.weekday()  # 返回0-6的数字，0代表周一，6代表周日
    # 转换为中文星期
    weekdays = "weekday" if 0 <= weekday_num <= 4 else "weekend"

    season = (
        "spring" if 3 <= month <= 5 else
        "summer" if 6 <= month <= 8 else
        "autumn" if 9 <= month <= 11 else
        "winter"
    )

    time_of_day = (
        "morning" if 5 <= hour < 11 else
        "daytime" if 11 <= hour < 17 else
        "evening" if 17 <= hour < 23 else
        "night"
    )

    return {"season": f"{season}",
            "weekday": f"{weekdays}",
            "time": f"{time_of_day}"}


def recommend_devices(user: UserProfile, match_devices: List[Union[str, List[str]]]) -> List[str]:
    """使用层次分析法(AHP)综合推荐设备，同组设备只选评分最高的"""
    # 1. 获取当前上下文
    time_dict = get_seasonal_context()
    season = time_dict["season"]
    weekday = time_dict["weekday"]
    time = time_dict["time"]

    # 2. 定义AHP层次结构和权重
    """
    层次结构：
    - 目标层：选择最佳设备
    - 准则层：
        1. 地域特征（权重：0.3）
        2. 家庭特征（权重：0.25）
        3. 生活习惯（权重：0.2）
        4. 时间相关（权重：0.15）
        5. 使用频率（权重：0.1）
    - 方案层：各候选设备
    """
    CRITERIA_WEIGHTS = {
        'region': 0.3,
        'family': 0.25,
        'lifestyle': 0.2,
        'time_related': 0.15,
        'usage': 0.1
    }

    # 3. 计算每个准则下的设备得分（归一化到0-1）
    def calculate_ahp_score(device: str) -> float:
        # 各准则下的原始得分
        criteria_scores = {
            'region': 0.0,
            'family': 0.0,
            'lifestyle': 0.0,
            'time_related': 0.0,
            'usage': 0.0
        }

        # 地域特征得分
        if device in REGION_DEVICE_MAP[user.region].get(season, []):
            criteria_scores['region'] = 1.0

        # 家庭特征得分
        family_features = 0
        matched_features = 0
        for feature, devices in FAMILY_FEATURE_MAP.items():
            family_features += 1
            if getattr(user, feature) and device in devices:
                matched_features += 1
        if family_features > 0:
            criteria_scores['family'] = matched_features / family_features

        # 生活习惯得分
        lifestyle_score = 0.0
        # 烹饪习惯
        if device in LIFESTYLE_FEATURES_MAP["cooking"][user.cooking_habits]:
            lifestyle_score += 0.6  # 烹饪习惯权重60%
        # 工作时间
        if user.work_schedule != "regular" and device in LIFESTYLE_FEATURES_MAP["work_schedule"][user.work_schedule]:
            lifestyle_score += 0.4  # 工作时间权重40%
        criteria_scores['lifestyle'] = lifestyle_score

        # 时间相关得分
        time_score = 0.0
        if device in SEASON_DEVICE_MAP[season]:
            time_score += 0.5  # 季节权重50%
        if device in TIME_DEVICE_MAP[weekday][time]:
            time_score += 0.5  # 时间权重50%
        criteria_scores['time_related'] = time_score

        # 使用频率得分（归一化到0-1）
        max_usage = max(user.device_usage.values()) if user.device_usage else 1
        criteria_scores['usage'] = user.device_usage.get(device, 0) / max_usage if max_usage > 0 else 0

        # 计算加权总分
        total_score = sum(criteria_scores[criteria] * CRITERIA_WEIGHTS[criteria]
                          for criteria in CRITERIA_WEIGHTS)
        return total_score

    # 4. 处理设备组
    result = []
    processed_groups = set()

    for group in match_devices:
        if isinstance(group, str):
            # 单个设备直接评分
            score = calculate_ahp_score(group)
            result.append((group, score))
        else:
            # 设备组转换为元组作为key
            group_key = tuple(sorted(group))
            if group_key not in processed_groups:
                # 找出组内最高分设备
                best_device = max(group, key=lambda d: calculate_ahp_score(d))
                score = calculate_ahp_score(best_device)
                result.append((best_device, score))
                processed_groups.add(group_key)

    # 5. 按评分排序并返回设备列表
    result.sort(key=lambda x: -x[1])
    return [device for device, score in result] if result else None


def match_keyword(text: str) -> Optional[list[str]]:
    """返回匹配到的设备列表，未匹配返回None"""
    text = text.lower()
    result = list()
    for keyword, devices in KEYWORD_MAP.items():
        if keyword in text:
            if len(devices) == 1:
                result.append(devices[0])
            else:
                result.append(devices)
    return result


def get_access_token() -> str:
    """获取百度千帆API的访问令牌"""
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {
        "client_id": os.environ.get("QIANFAN_API_KEY"),
        "client_secret": os.environ.get("QIANFAN_SECRET_KEY"),
        "grant_type": "client_credentials"
    }
    response = requests.post(url, params=params)
    return response.json()['access_token']


def chat_qianfan(content: str) -> str:
    """与百度千帆AI聊天并获取响应"""
    payload = json.dumps({
        "messages": [
            {
                "role": "user",
                "content": content  # 用户输入的内容
            }
        ],
        "temperature": 0.5  # 可选参数，控制生成结果的随机性
    })

    url = f"https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions_pro?access_token={get_access_token()}"
    response = requests.post(url, headers={'Content-Type': "application/json"}, data=payload)
    return response.json()['result']


def chat_spark(content: str) -> str:
    """与讯飞星火AI聊天并获取响应"""
    spark = ChatSparkLLM(
        spark_api_url='wss://spark-api.xf-yun.com/v1.1/chat',
        spark_app_id='54f0b31e',
        spark_api_key=os.environ.get("SPARK_API_KEY"),
        spark_api_secret=os.environ.get("SPARK_SECRET_KEY"),
        spark_llm_domain='lite',
        streaming=False,
    )
    messages = [ChatMessage(
        role="user",
        content=content
    )]
    handler = ChunkPrintHandler()
    a = spark.generate([messages], callbacks=[handler])
    return a.generations[0][0].message.content


def chat_doubao(content: str) -> str:
    """与豆包AI聊天并获取响应"""
    client = Ark(
        # 此为默认路径，您可根据业务所在地域进行配置
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        # 从环境变量中获取您的 API Key。此为默认方式，您可根据需要进行修改
        api_key=os.environ.get("ARK_API_KEY"),
    )

    completion = client.chat.completions.create(
        # 指定您创建的方舟推理接入点 ID，此处已帮您修改为您的推理接入点 ID
        model="ep-20250329165324-l8vxt",
        messages=[
            {"role": "user", "content": f"{content}"},
        ],
        extra_headers={'x-is-encrypted': 'true'},
    )
    return completion.choices[0].message.content


def chat_deepseek(content: str) -> str:
    """与Deepseek AI聊天并获取响应"""
    client = Ark(
        # 此为默认路径，您可根据业务所在地域进行配置
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        # 从环境变量中获取您的 API Key。此为默认方式，您可根据需要进行修改
        api_key=os.environ.get("DS_API_KEY"),
    )

    completion = client.chat.completions.create(
        # 指定您创建的方舟推理接入点 ID，此处已帮您修改为您的推理接入点 ID
        model="ep-20250329165324-l8vxt",
        messages=[
            {"role": "user", "content": f"{content}"},
        ],
        extra_headers={'x-is-encrypted': 'true'},
    )
    return completion.choices[0].message.content


def check_device(matched_devices: list) -> bool:
    if matched_devices is None:
        return False
    if all(isinstance(item, str) for item in matched_devices) is True:
        return True
    else:
        return False


def get_device(user_input: str, user_profile: UserProfile) -> list[str] | str:
    # 实时场景检测
    current_scene = history.detect_scene(user_input)
    # print(current_scene)
    if current_scene:
        print("处于" + current_scene + "（输入结束场景来停止）")
    # 先尝试关键词匹配
    matched_devices = match_keyword(user_input)
    # print(matched_devices)
    # 场景敏感的设备过滤
    if matched_devices and current_scene:
        current_scene_devices = SCENE_DEVICE_MAP.get(current_scene, [])
        filtered_devices = []
        for device in matched_devices:
            if isinstance(device, str):
                if device in current_scene_devices:
                    filtered_devices.append(device)
            elif isinstance(device, list):
                for dev in device:
                    if dev in current_scene_devices:
                        filtered_devices.append(dev)
        matched_devices = filtered_devices
        if matched_devices:
            return matched_devices
        if not matched_devices:  # 关键修改：无匹配立即退出场景
            history.force_exit_scene()

    # print(matched_devices)
    if not current_scene:
        matched_devices = match_keyword(user_input)
        # print(matched_devices)
        # 根据时间和用户画像选择最匹配的电器
        matched_devices = recommend_devices(user_profile, matched_devices)
        # print(matched_devices)
        if check_device(matched_devices):
            return matched_devices

        # 如果经过所有匹配流程还是没有设备
    # if not matched_devices:
    #     return ["未知设备"]

    # 无匹配则走AI流程
    prompt = PROMPT.format(
        user_input=user_input,
    )

    response = chat_qianfan(prompt)
    return response


def process_input(user_input: str, user_profile: UserProfile):
    """处理用户输入的核心逻辑"""
    if not user_input:
        return "未知设备"

    if user_input.lower() in ("退出", "exit"):
        return "已退出"  # 终止信号
    elif user_input.lower() == "结束场景":
        history.force_exit_scene()
        print("场景已结束")
        return "场景已结束"

    device = get_device(user_input, user_profile)
    if "未知设备" in device:
        return device
    history.add(user_input, device)

    for dev in device:
        user_profile.record_device_usage(dev)

    user_profile.save_to_file("user_profile.json")
    return device


def initialize_user_profile() -> UserProfile:
    """Initialize a new user profile by collecting information from user input"""
    print("\n===== 新用户配置向导 =====")
    print("请回答以下问题来初始化您的个人资料 (直接回车可跳过问题)\n")

    # Basic Information
    age = input("1. 您的年龄: ")
    print("\n2. 您的性别 (男/女): ")
    print("  1) 男")
    print("  2) 女")
    sex = input("请选择(1-2): ")
    gender = ["male", "female"][int(sex) - 1] if sex in "12" else "male"

    print("\n3. 您所在的地区 (north/south): ")
    print("  1) 北方")
    print("  2) 南方")
    location = input("请选择(1-2): ")
    region = ["north", "south"][int(location) - 1] if location in "12" else "south"

    # Family Information
    family_members = input("4. 家庭成员数量: ")
    has_children = input("5. 家中有小孩吗? (y/n): ").lower() == 'y'
    has_elderly = input("6. 家中有老人吗? (y/n): ").lower() == 'y'
    has_pet = input("7. 家中有宠物吗? (y/n): ").lower() == 'y'

    # Lifestyle
    print("\n8. 您的工作时间:")
    print("  1) 朝九晚五 (regular)")
    print("  2) 夜班 (night_shift)")
    print("  3) 灵活工作时间 (flexible)")
    work_choice = input("请选择(1-3): ")
    work_schedule = ["regular", "night_shift", "flexible"][int(work_choice) - 1] if work_choice in "123" else "regular"

    print("\n9. 您的烹饪频率:")
    print("  1) 很少做饭 (rare)")
    print("  2) 偶尔做饭 (medium)")
    print("  3) 经常做饭 (frequent)")
    cook_choice = input("请选择(1-3): ")
    cooking_habits = ["rare", "medium", "frequent"][int(cook_choice) - 1] if cook_choice in "123" else "medium"

    # Initialize with collected data (convert empty strings to None)
    user_profile = UserProfile(
        age=int(age) if age else None,
        gender=gender if gender else None,
        region=region if region else None,
        family_members=int(family_members) if family_members else 1,
        has_children=has_children,
        has_elderly=has_elderly,
        has_pet=has_pet,
        work_schedule=work_schedule,
        cooking_habits=cooking_habits,
        device_usage={}  # Start with empty device usage
    )

    # Save the profile
    user_profile.save_to_file("user_profile.json")
    print("\n用户配置已完成并保存!")
    return user_profile


def main():
    # 读取用户信息
    try:
        with open('user_profile.json', 'r', encoding='utf-8') as user_file:
            user_profile = UserProfile.load_from_file('user_profile.json')  # 自动处理嵌套结构
            print(user_profile)
    except FileNotFoundError:
        user_profile = initialize_user_profile()
    while True:
        print("\n请选择输入方式:")
        print("1. 文本输入")
        print("2. 语音输入")
        print("3. 初始化用户信息")
        print("4. 退出")
        choice = input("请输入选项(1-4): ").strip()
        # print("当前用户画像:", ContextService.get_user_context(user_profile))
        # print(user_profile.get_sorted_devices())
        if choice == "1":
            user_input = input("用户输入: ").strip()
            result = process_input(user_input, user_profile)
            if result:
                print(f"需要操作的设备：{result}")
        elif choice == "2":
            user_input = VoiceRecognition()
            result = process_input(user_input, user_profile)
            if result:
                print(f"需要操作的设备：{result}")
        elif choice == "3":
            user_file = initialize_user_profile()
        elif choice == "4":
            break


if __name__ == "__main__":
    main()
