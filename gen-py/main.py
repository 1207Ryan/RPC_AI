import os
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
from dataclasses import field
import requests
from prompts import *
from sparkai.llm.llm import ChatSparkLLM, ChunkPrintHandler
from sparkai.core.messages import ChatMessage
from volcenginesdkarkruntime import Ark
from voice import VoiceRecognition
from sklearn.preprocessing import MinMaxScaler
import numpy as np


class DialogHistory:
    def __init__(self, max_length: int = 3):
        self.history: List[Dict] = []
        self.max_length = max_length
        self.current_scene: Optional[str] = "默认场景"  # 明确标注可为None
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
        if self.current_scene != "默认场景":
            self.scene_persist_counter -= 1  # 每轮无匹配递减
            return self.current_scene

        # 清空场景的条件（计数器归零或没有设备匹配）
        if self.scene_persist_counter <= 0:
            self.current_scene = "默认场景"

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
        self.current_scene = "默认场景"


history = DialogHistory()


class UserProfile:
    def __init__(self, user_id: int, age: Optional[int] = 20, gender: Optional[str] = "male",
                 region: Optional[str] = "south", family_members: Optional[int] = 1,
                 has_children: Optional[bool] = False, has_elderly: Optional[bool] = False,
                 has_pet: Optional[bool] = False, work_schedule: Optional[str] = "regular",
                 cooking_habits: Optional[str] = "medium", device_usage: Dict[str, int] = None):
        # 基础信息
        self.user_id = user_id
        self.age = age
        self.gender = gender
        self.region = region
        # 家庭信息
        self.family_members = family_members
        self.has_children = has_children
        self.has_elderly = has_elderly
        self.has_pet = has_pet
        # 生活习惯
        self.work_schedule = work_schedule if work_schedule is not None else "regular"  # "regular", "night_shift", "flexible"
        self.cooking_habits = cooking_habits if cooking_habits is not None else "medium"  # "rare", "medium", "frequent"
        self.device_usage = device_usage if device_usage is not None else {}

    def record_device_usage(self, device_name: str):
        """记录设备使用次数"""
        self.device_usage[device_name] = self.device_usage.get(device_name, 0) + 1

    def record_device(self, device: str | list[str]):
        for dev in device:
            if isinstance(dev, str):
                self.record_device_usage(dev)
            elif isinstance(dev, list):
                for d in dev:
                    self.record_device_usage(d)

        self.save_to_file(f"user_profiles/user_{self.user_id}.json")  # 修改为按用户ID保存

    def save_to_file(self, filepath: str):
        """保存完整用户数据"""
        # 确保用户目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        data = {
            "user_id": self.user_id,  # 保存用户ID
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
                    user_id=data.get("user_id", "default"),  # 读取用户ID
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
            return cls(user_id=0)  # 返回默认配置


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


def recommend_devices(user: UserProfile, match_devices: List[Union[str, List[str]]]) -> list[Any] | None:
    """使用TOPSIS综合推荐设备，同组设备只选评分最高的"""
    # 1. 获取当前上下文
    time_dict = get_seasonal_context()
    season = time_dict["season"]
    weekday = time_dict["weekday"]
    time = time_dict["time"]

    # 2. 定义评价指标和权重（与AHP相同，但TOPSIS权重需归一化）
    CRITERIA_WEIGHTS = np.array([
        0.3,  # region
        0.25,  # family
        0.2,  # lifestyle
        0.15,  # time_related
        0.1  # usage
    ])
    CRITERIA_WEIGHTS = CRITERIA_WEIGHTS / CRITERIA_WEIGHTS.sum()  # 归一化权重

    # 3. 构建决策矩阵（每行是一个设备，每列是一个准则的得分）
    def get_device_scores(device: str) -> list:
        """计算设备在各准则下的得分（与AHP逻辑一致）"""
        # (1) 地域特征：设备是否适应当前地区和季节？
        scores = [1.0 if device in REGION_DEVICE_MAP[user.region].get(season, []) else 0.0]
        # (2) 家庭特征：设备是否符合家庭情况（小孩/老人/宠物）？
        matched_features = sum(
            1 for feature, devices in FAMILY_FEATURE_MAP.items()
            if getattr(user, feature) and device in devices
        )
        family_features = len(FAMILY_FEATURE_MAP)
        scores.append(matched_features / family_features if family_features > 0 else 0.0)
        # (3) 生活习惯：设备是否符合用户的工作和烹饪习惯？
        lifestyle_score = 0.0
        if device in LIFESTYLE_FEATURES_MAP["cooking"][user.cooking_habits]:
            lifestyle_score += 0.6
        if user.work_schedule != "regular" and device in LIFESTYLE_FEATURES_MAP["work_schedule"][user.work_schedule]:
            lifestyle_score += 0.4
        scores.append(lifestyle_score)
        # (4) 时间相关：设备是否适合当前季节和时间段？
        time_score = 0.0
        if device in SEASON_DEVICE_MAP[season]:
            time_score += 0.5
        if device in TIME_DEVICE_MAP[weekday][time]:
            time_score += 0.5
        scores.append(time_score)
        # (5) 使用频率：设备是否经常被使用？
        max_usage = max(user.device_usage.values()) if user.device_usage else 1
        usage_score = user.device_usage.get(device, 0) / max_usage if max_usage > 0 else 0
        scores.append(usage_score)
        return scores

    # 4. 处理设备组（每组只保留最高分设备）
    candidate_devices = []
    processed_groups = set()
    for group in match_devices:
        if isinstance(group, str):
            candidate_devices.append(group)
        else:
            group_key = tuple(sorted(group))
            if group_key not in processed_groups:
                candidate_devices.append(max(group, key=lambda d: sum(get_device_scores(d))))
                processed_groups.add(group_key)

    if not candidate_devices:
        return None

    # 5. 构建决策矩阵并归一化（向量归一化，TOPSIS标准方法）
    decision_matrix = np.array([get_device_scores(device) for device in candidate_devices])

    # 处理可能全为零的列
    column_norms = np.linalg.norm(decision_matrix, axis=0, keepdims=True)
    # 将零范数列替换为1，避免除以零
    column_norms[column_norms == 0] = 1
    norm_matrix = decision_matrix / column_norms

    # 6. 加权归一化矩阵
    weighted_matrix = norm_matrix * CRITERIA_WEIGHTS

    # 7. 确定理想解和负理想解（所有指标均为正向指标）
    ideal_best = weighted_matrix.max(axis=0)
    ideal_worst = weighted_matrix.min(axis=0)

    # 8. 计算距离和相对接近度
    dist_best = np.linalg.norm(weighted_matrix - ideal_best, axis=1)
    dist_worst = np.linalg.norm(weighted_matrix - ideal_worst, axis=1)

    # 处理可能的零距离情况
    with np.errstate(divide='ignore', invalid='ignore'):
        closeness = np.where((dist_best + dist_worst) != 0,
                             dist_worst / (dist_best + dist_worst),
                             0)  # 如果分母为零，则设为0

    # 9. 按接近度排序设备
    ranked_devices = [
        device for _, device in sorted(zip(closeness, candidate_devices), reverse=True)
    ]
    return ranked_devices


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
    if "未知设备" in response:
        return ["未知设备"]

    if isinstance(response, str):
        response = json.loads(response)
    filtered_devices = []
    for device in response:
        if isinstance(device, str):
            filtered_devices.append(device)
        elif isinstance(device, list):
            for dev in device:
                filtered_devices.append(dev)
    return filtered_devices


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

    user_profile.record_device(device)

    return device


def get_user_profile(user_id: int) -> UserProfile:
    """根据用户ID获取用户配置"""
    # 验证user_id是否为有效整数
    try:
        user_id = int(user_id)
    except (ValueError, TypeError):
        user_id = 0  # 无效ID使用默认值

    profile_path = f"user_profiles/user_{user_id}.json"
    if os.path.exists(profile_path):
        return UserProfile.load_from_file(profile_path)
    else:
        return initialize_user_profile(user_id)


def initialize_user_profile(user_id: int) -> UserProfile:
    """Initialize a new user profile by collecting information from user input"""
    print(f"\n===== 用户 {user_id} 配置向导 =====")
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
        user_id=user_id,
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
    user_profile.save_to_file(f"user_profiles/user_{user_id}.json")
    print("\n用户配置已完成并保存!")
    return user_profile


def main():
    # 获取用户ID
    global user_profile
    user_id = input("请输入您的用户ID: ")
    if not user_id:
        user_id = 0

    # 获取用户配置
    user_profile = get_user_profile(user_id)

    while True:
        print("\n请选择输入方式:")
        print("1. 文本输入")
        print("2. 语音输入")
        print("3. 初始化用户信息")
        print("4. 切换用户")
        print("5. 退出")
        choice = input("请输入选项(1-5): ").strip()
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
            user_profile = initialize_user_profile(user_id)
        elif choice == "4":
            new_user_input = input("请输入新的64位整数用户ID: ").strip()
            new_user_id = int(new_user_input)
            if new_user_id != user_id:
                user_id = new_user_id
                user_profile = get_user_profile(user_id)
        elif choice == "5":
            break


if __name__ == "__main__":
    main()
