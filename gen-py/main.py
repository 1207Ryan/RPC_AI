import time
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
import requests
from voice import VoiceRecognition
from voicefile import *
from device import *
import os
import json
from collections import defaultdict


class DialogHistory:
    def __init__(self, max_length: int = 3):
        self.history: List[Dict] = []
        self.max_length = max_length
        self.current_scene: Optional[str] = "默认场景"  # 明确标注可为None
        self.scene_persist_counter = 0  # 场景持续计数器
        # 初始化时构建所有设备名称列表
        self.all_device_names = [appliance.name for appliance in appliances]
        # 预构建设备-场景映射
        self.device_scene_map = {
            appliance.name: appliance.scene for appliance in appliances
        }

        # 预构建场景-设备映射
        self.scene_device_map = {}
        for appliance in appliances:
            # 如果场景不在映射中，创建空列表
            if appliance.scene not in self.scene_device_map:
                self.scene_device_map[appliance.scene] = []
            # 将设备添加到对应场景的列表中
            self.scene_device_map[appliance.scene].append(appliance.name)

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

    def set_scene(self, device: str | list[str]):
        self.current_scene = self.device_scene_map[device]
        self.scene_persist_counter = self.max_length

    def detect_scene(self, user_input: str) -> Optional[str]:
        """实时场景检测（基于家电场景）"""
        if self.current_scene != "默认场景":
            self.scene_persist_counter -= 1
            if self.scene_persist_counter <= 0:
                self.current_scene = "默认场景"
            return self.current_scene
        return "默认场景"

    def force_exit_scene(self):
        self.current_scene = "默认场景"


history = DialogHistory()


class UserProfile:
    def __init__(self, user_id: int, age: Optional[int] = 20, gender: Optional[str] = "male",
                 region: Optional[str] = "south", family_members: Optional[int] = 1,
                 has_children: Optional[bool] = False, has_elderly: Optional[bool] = False,
                 has_pet: Optional[bool] = False, work_schedule: Optional[str] = "regular",
                 cooking_habits: Optional[str] = "medium", device_usage: Dict[str, int] = None,
                 device_sequences: Dict[str, List[str]] = None):
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
        # 设备使用序列（用于预测）
        self.device_sequences = device_sequences if device_sequences is not None else defaultdict(list)
        self.markov_model = defaultdict(lambda: defaultdict(int))  # 马尔可夫转移矩阵
        self._build_markov_model()  # 初始化时构建模型
        if "recent" not in self.device_sequences:
            self.device_sequences["recent"] = []

    def _build_markov_model(self):
        """基于历史序列构建马尔可夫转移矩阵"""
        for seq in self.device_sequences.values():
            for i in range(len(seq) - 1):
                current = seq[i]
                next_dev = seq[i + 1]
                self.markov_model[current][next_dev] += 1

    def _get_markov_prediction(self, current_device: str, top_n: int = 3) -> List[str]:
        """
        结合熵权法的马尔可夫链预测逻辑
        :param current_device: 当前设备名称
        :param top_n: 返回前N个预测结果
        :return: 预测的设备列表（可能为空）
        """
        if not current_device or current_device not in self.markov_model:
            return []

        # 获取所有可能的下一设备及其转移次数
        next_devices = self.markov_model[current_device]

        # 计算每个设备的使用频率熵权
        device_names = list(next_devices.keys())
        usage_counts = np.array([self.device_usage.get(dev, 1) for dev in device_names])

        # 计算熵权
        def entropy_weight(usage):
            """计算单个设备的使用频率熵权"""
            if len(usage) == 0:
                return 0

            # 归一化使用频率
            p = usage / usage.sum()

            # 避免对数计算中的零值
            p = np.maximum(p, 1e-10)

            # 计算熵值
            entropy = -np.sum(p * np.log(p))

            # 计算差异系数（信息效用值）
            d = 1 - entropy

            # 计算权重（归一化）
            return d

        # 为每个设备计算熵权
        weights = np.array([entropy_weight(np.array([count])) for count in usage_counts])

        # 结合转移次数和熵权计算综合得分
        scores = {dev: next_devices[dev] * weights[i] for i, dev in enumerate(device_names)}

        # 按得分排序
        sorted_devices = sorted(
            scores.items(),
            key=lambda x: -x[1]  # 按综合得分降序
        )

        # 返回top_n个设备（排除当前设备）
        return [dev for dev, _ in sorted_devices if dev != current_device][:top_n]

    def predict_next_devices(self, current_device: str = None) -> List[str]:
        """改进后的预测方法（优先级：场景 > 马尔可夫 > 近期模式 > 通用推荐）"""
        predictions = []

        # 1. 场景优先预测
        if history.current_scene != "默认场景":
            scene_key = f"scene_{history.current_scene}"
            scene_sequence = self.device_sequences.get(scene_key, [])
            if scene_sequence:
                scene_devices = [d for d in scene_sequence if d != current_device]
                predictions.extend(list(set(scene_devices))[:3])

        # 2. 马尔可夫链预测（当有当前设备时）
        if not predictions and current_device:
            # 调用结合熵权法的马尔可夫预测
            markov_pred = self._get_markov_prediction(current_device, top_n=2)
            predictions.extend(markov_pred)

        # 3. 近期使用模式（原逻辑）
        if not predictions:
            recent_sequence = self.device_sequences.get("recent", [])
            if len(recent_sequence) >= 2:
                last_two = tuple(recent_sequence[-2:])
                possible_next = []
                for seq in self.device_sequences.values():
                    for i in range(len(seq) - 2):
                        if tuple(seq[i:i + 2]) == last_two and seq[i + 2] != current_device:
                            possible_next.append(seq[i + 2])
                if possible_next:
                    from collections import Counter
                    predictions.extend([item[0] for item in Counter(possible_next).most_common(2)])

        # 4. 通用推荐（原TOPSIS逻辑）
        if not predictions:
            all_devices = list(self.device_usage.keys())
            if all_devices:
                recommended = recommend_devices(self, [[d] for d in all_devices])
                predictions.extend([d for d in recommended if d != current_device][:2])

        return list(set(predictions))[:3]  # 去重并限制数量

    def record_device_sequence(self, device_name: str):
        """增强的记录方法（自动更新马尔可夫模型）"""
        if "recent" not in self.device_sequences:
            self.device_sequences["recent"] = []
        # 原有记录逻辑
        if len(self.device_sequences["recent"]) >= 5:
            self.device_sequences["recent"].pop(0)
        self.device_sequences["recent"].append(device_name)

        if history.current_scene != "默认场景":
            scene_key = f"scene_{history.current_scene}"
            # 确保场景键存在于字典中
            if scene_key not in self.device_sequences:
                self.device_sequences[scene_key] = []
            if len(self.device_sequences[scene_key]) >= 3:
                self.device_sequences[scene_key].pop(0)
            self.device_sequences[scene_key].append(device_name)

        # 实时更新马尔可夫模型（仅更新最近两个状态）
        if len(self.device_sequences["recent"]) >= 2:
            prev_device = self.device_sequences["recent"][-2]
            self.markov_model[prev_device][device_name] += 1

    def record_device_usage(self, device_name: str):
        """记录设备使用次数"""
        self.device_usage[device_name] = self.device_usage.get(device_name, 0) + 1

    def record_device(self, device: str | list[str]):
        for dev in device:
            if isinstance(dev, str):
                self.record_device_usage(dev)
                self.record_device_sequence(dev)
            elif isinstance(dev, list):
                for d in dev:
                    self.record_device_usage(d)
                    self.record_device_sequence(d)

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
                "sequences": self.device_sequences
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
                    device_usage=data["device_data"].get("usage", {}),
                    device_sequences=data["device_data"].get("sequences", defaultdict(list))
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
    """使用TOPSIS综合推荐设备，同组设备只选评分最高的，权重由固定值确定"""
    # 1. 获取当前上下文
    time_dict = get_seasonal_context()
    season = time_dict["season"]
    weekday = time_dict["weekday"]
    time = time_dict["time"]

    # 2. 构建决策矩阵（每行是一个设备，每列是一个准则的得分）
    def get_device_scores(device: str) -> list:
        """计算设备在各准则下的得分"""
        # Find the appliance
        appliance = next((a for a in appliances if a.name == device), None)
        if not appliance:
            return [0.0] * 5  # Default scores if appliance not found

        # 安全获取 lifestyle_features，默认为空字典
        lifestyle_features = appliance.lifestyle_features or {}

        # 生活习惯评分
        cooking_match = 0.6 if user.cooking_habits in lifestyle_features.get("cooking", []) else 0.0
        work_schedule_match = 0.4 if user.work_schedule in lifestyle_features.get("work_schedule", []) else 0.0
        lifestyle_score = cooking_match + work_schedule_match

        # 时间相关评分
        season_match = 0.5 if season in appliance.season_suitability else 0.0
        time_match = 0.5 if time in appliance.time_suitability.get(weekday, []) else 0.0
        time_score = season_match + time_match

        # 使用频率评分，避免除零错误
        usage_scores = list(user.device_usage.values())
        max_usage = max(usage_scores) if usage_scores else 1.0  # 避免除零

        scores = [
            # (1) 地域特征
            1.0 if user.region in appliance.region_suitability else 0.0,

            # (2) 家庭特征
            sum(1 for feature in appliance.family_features if getattr(user, feature, False)) / len(
                appliance.family_features) if appliance.family_features else 0.0,

            # (3) 生活习惯
            lifestyle_score,

            # (4) 时间相关
            time_score,

            # (5) 使用频率
            user.device_usage.get(device, 0) / max_usage
        ]

        return scores

    # 3. 处理设备组（每组只保留最高分设备）
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

    # 4. 构建决策矩阵
    decision_matrix = np.array([get_device_scores(device) for device in candidate_devices])

    # 5. 使用固定权重
    CRITERIA_WEIGHTS = np.array([0.25, 0.20, 0.15, 0.15, 0.25])  # 固定权重分配

    # 6. 归一化决策矩阵（向量归一化）
    column_norms = np.linalg.norm(decision_matrix, axis=0, keepdims=True)
    column_norms[column_norms == 0] = 1  # 避免除以零
    norm_matrix = decision_matrix / column_norms

    # 7. 加权归一化矩阵
    weighted_matrix = norm_matrix * CRITERIA_WEIGHTS

    # 8. 确定理想解和负理想解
    ideal_best = weighted_matrix.max(axis=0)
    ideal_worst = weighted_matrix.min(axis=0)

    # 9. 计算距离和相对接近度
    dist_best = np.linalg.norm(weighted_matrix - ideal_best, axis=1)
    dist_worst = np.linalg.norm(weighted_matrix - ideal_worst, axis=1)

    # 处理可能的零距离情况
    with np.errstate(divide='ignore', invalid='ignore'):
        closeness = np.where((dist_best + dist_worst) != 0,
                             dist_worst / (dist_best + dist_worst),
                             0)  # 如果分母为零，则设为0

    # 10. 按接近度排序设备
    ranked_devices = [
        device for _, device in sorted(zip(closeness, candidate_devices), reverse=True)
    ]
    return ranked_devices


def match_keyword(text: str) -> Optional[list[str]]:
    """返回匹配到的设备列表，未匹配返回None"""
    text = text.lower()
    result = []
    for appliance in appliances:
        if any(keyword in text for keyword in appliance.keywords):
            result.append(appliance.name)
    return result if result else None


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


def check_device(matched_devices: list) -> bool:
    if matched_devices is None:
        return False
    if all(isinstance(item, str) for item in matched_devices) is True:
        return True
    else:
        return False


def get_device(user_input: str, user_profile: UserProfile) -> tuple[list[str | Any], str | None] | list[Any] | None | \
                                                              list[str] | list[str | Any]:
    # 实时场景检测
    current_scene = history.detect_scene(user_input)
    # 先尝试关键词匹配
    matched_devices = match_keyword(user_input)
    # print(matched_devices)
    # 场景敏感的设备过滤
    if matched_devices and current_scene != "默认场景":
        current_scene_devices = history.scene_device_map[current_scene]
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
            return matched_devices, current_scene
        if not matched_devices:  # 关键修改：无匹配立即退出场景
            current_scene = "默认场景"
            history.force_exit_scene()

    # print(matched_devices)
    if not current_scene or current_scene == "默认场景":
        matched_devices = match_keyword(user_input)
        if matched_devices:
            # print(matched_devices)
            # 根据时间和用户画像选择最匹配的电器
            matched_devices = recommend_devices(user_profile, matched_devices)
            # print(matched_devices)
            if check_device(matched_devices):
                history.set_scene(matched_devices[0])
                return matched_devices, history.current_scene

    # 如果经过所有匹配流程还是没有设备
    # if not matched_devices:
    #     return ["未知设备"]

    # 无匹配则走AI流程
    prompt = PROMPT.format(
        user_input=user_input,
        device_names=", ".join([device for device in history.all_device_names]),
    )

    response = chat_qianfan(prompt)
    if "未知设备" in response:
        return ["未知设备"], "默认场景"

    if isinstance(response, str):
        response = json.loads(response)
    filtered_devices = []
    for device in response:
        if isinstance(device, str):
            filtered_devices.append(device)
        elif isinstance(device, list):
            for dev in device:
                filtered_devices.append(dev)
    return filtered_devices, current_scene


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

    device, scene = get_device(user_input, user_profile)
    if scene:
        print("处于" + scene + "（输入结束场景来停止）")

    if "未知设备" in device:
        return device

    history.add(user_input, device)
    user_profile.record_device(device)
    return device


def predict_next_devices(device, user_profile: UserProfile) -> list[str]:
    # 预测下一步可能需要的设备
    current_device = device[0] if isinstance(device, list) else device
    predicted_devices = user_profile.predict_next_devices(current_device)
    return predicted_devices if predicted_devices else ["暂无推荐设备"]


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
        device_usage={},  # Start with empty device usage
        device_sequences=defaultdict(list)  # Start with empty sequences
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
                print("接下来您可能需要：" + predict_next_devices(result, user_profile)[0])
        elif choice == "2":
            user_input = VoiceRecognition()
            result = process_input(user_input, user_profile)
            if result:
                print(f"需要操作的设备：{result}")
                print("接下来您可能需要：" + predict_next_devices(result, user_profile)[0])
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
