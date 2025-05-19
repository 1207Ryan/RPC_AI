from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class HomeAppliance:
    name: str
    scene: str
    keywords: List[str]
    season_suitability: List[str] = None
    time_suitability: Dict[str, List[str]] = None
    region_suitability: List[str] = None
    family_features: List[str] = None
    lifestyle_features: Dict[str, List[str]] = None


# 全量电器列表初始化（调整 lifestyle_features）
appliances = [
    # --------------------- 温度调节类 ---------------------
    HomeAppliance(
        name="空调",
        scene="温度调节",
        keywords=["热", "冷", "闷", "凉"],
        season_suitability=["summer", "winter", "all_season"],
        time_suitability={
            "weekday": ["night"],
            "weekend": ["night"]
        },
        region_suitability=["north", "south"],
        family_features=["has_children", "has_elderly"],
        lifestyle_features={}
    ),
    HomeAppliance(
        name="电风扇",
        scene="温度调节",
        keywords=["热", "凉"],
        season_suitability=["summer", "all_season"],
        time_suitability={
            "weekday": ["daytime", "night"],
            "weekend": ["daytime", "night"]
        },
        region_suitability=["north", "south"],
        family_features=["has_children"],
        lifestyle_features={}
    ),
    HomeAppliance(
        name="暖气",
        scene="温度调节",
        keywords=["冷"],
        season_suitability=["winter"],
        time_suitability={
            "weekday": ["night"],
            "weekend": ["night"]
        },
        region_suitability=["north"],
        family_features=["has_elderly"],
        lifestyle_features={}
    ),
    HomeAppliance(
        name="电热毯",
        scene="睡眠场景",
        keywords=["冷"],
        season_suitability=["winter"],
        time_suitability={
            "weekday": ["night"],
            "weekend": ["night"]
        },
        region_suitability=["north", "south"],
        family_features=["has_elderly"],
        lifestyle_features={}
    ),
    HomeAppliance(
        name="浴霸",
        scene="洗浴场景",
        keywords=["冷", "澡"],
        season_suitability=["winter"],
        time_suitability={
            "weekday": ["morning", "night"],
            "weekend": ["morning", "night"]
        },
        region_suitability=["north", "south"],
        family_features=["has_elderly"],
        lifestyle_features={}
    ),

    # --------------------- 用水相关类 ---------------------
    HomeAppliance(
        name="热水器",
        scene="洗浴场景",
        keywords=["烧水", "澡", "洗漱"],
        season_suitability=["all_season"],
        time_suitability={
            "weekday": ["morning", "evening"],
            "weekend": ["morning", "evening"]
        },
        region_suitability=["north", "south"],
        family_features=["has_children", "has_elderly"],
        lifestyle_features={}
    ),
    HomeAppliance(
        name="净水器",
        scene="用水相关",
        keywords=[],  # 通过场景映射触发
        season_suitability=["all_season"],
        time_suitability={
            "weekday": ["morning"],
            "weekend": ["morning"]
        },
        region_suitability=["north", "south"],
        family_features=["has_children"],
        lifestyle_features={}
    ),
    HomeAppliance(
        name="智能马桶",
        scene="默认场景",
        keywords=["厕所", "洗漱", "小便", "大便"],
        season_suitability=["all_season"],
        time_suitability={
            "weekday": ["morning", "night"],
            "weekend": ["morning", "night"]
        },
        region_suitability=["north", "south"],
        family_features=["has_elderly"],
        lifestyle_features={}
    ),

    # --------------------- 清洁需求类 ---------------------
    HomeAppliance(
        name="扫地机器人",
        scene="清洁场景",
        keywords=["地上", "地板", "扫地", "大扫除"],
        season_suitability=["spring", "summer", "autumn", "all_season"],
        time_suitability={
            "weekday": ["daytime"],
            "weekend": ["daytime"]
        },
        region_suitability=["north", "south"],
        family_features=["has_pet"],
        lifestyle_features={}
    ),
    HomeAppliance(
        name="洗衣机",
        scene="清洁场景",
        keywords=["脏", "洗衣"],
        season_suitability=["spring", "summer", "autumn", "all_season"],
        time_suitability={
            "weekday": ["evening"],
            "weekend": ["daytime"]
        },
        region_suitability=["north", "south"],
        family_features=["has_children"],
        lifestyle_features={}
    ),
    HomeAppliance(
        name="烘干机",
        scene="清洁场景",
        keywords=["烘干", "脏", "洗衣"],
        season_suitability=["autumn", "winter"],
        time_suitability={
            "weekday": ["evening"],
            "weekend": ["daytime"]
        },
        region_suitability=["north", "south"],
        family_features=["has_children"],
        lifestyle_features={}
    ),

    # --------------------- 空气管理类 ---------------------
    HomeAppliance(
        name="加湿器",
        scene="空气管理",
        keywords=["加湿"],
        season_suitability=["autumn", "winter"],
        time_suitability={
            "weekday": ["night"],
            "weekend": ["night"]
        },
        region_suitability=["north"],
        family_features=["has_elderly"],
        lifestyle_features={}
    ),
    HomeAppliance(
        name="除湿机",
        scene="空气管理",
        keywords=["除湿"],
        season_suitability=["spring", "summer"],
        time_suitability={
            "weekday": ["daytime"],
            "weekend": ["daytime"]
        },
        region_suitability=["south"],
        family_features=["has_children"],
        lifestyle_features={}
    ),

    # --------------------- 安防相关类 ---------------------
    HomeAppliance(
        name="智能门锁",
        scene="安防场景",
        keywords=["锁门"],
        season_suitability=["all_season"],
        time_suitability={
            "weekday": ["morning", "night"],
            "weekend": ["morning", "night"]
        },
        region_suitability=["north", "south"],
        family_features=["has_children", "has_elderly"],
        lifestyle_features={
            "work_schedule": ["night_shift", "flexible"]
        }
    ),
    HomeAppliance(
        name="摄像头",
        scene="安防场景",
        keywords=["监控", "摄像头"],
        season_suitability=["all_season"],
        time_suitability={
            "weekday": ["all_time"],
            "weekend": ["all_time"]
        },
        region_suitability=["north", "south"],
        family_features=["has_children", "has_pet"],
        lifestyle_features={
            "work_schedule": ["night_shift", "flexible"]
        }
    ),

    # --------------------- 厨房电器类 ---------------------
    HomeAppliance(
        name="冰箱",
        scene="烹饪场景",
        keywords=["冷藏", "渴", "喝"],
        season_suitability=["all_season"],
        time_suitability={
            "weekday": ["morning", "evening"],
            "weekend": ["morning", "evening"]
        },
        region_suitability=["north", "south"],
        family_features=["has_children"],
        lifestyle_features={}
    ),
    HomeAppliance(
        name="烤箱",
        scene="烹饪场景",
        keywords=["做饭", "吃"],
        season_suitability=["autumn", "winter"],
        time_suitability={
            "weekday": ["evening"],
            "weekend": ["morning", "evening"]
        },
        region_suitability=["north", "south"],
        family_features=["has_children"],
        lifestyle_features={
            "cooking": ["frequent"],  # 匹配偶尔/中等烹饪频率
        }
    ),
    HomeAppliance(
        name="微波炉",
        scene="烹饪场景",
        keywords=["饿", "吃", "做饭"],
        season_suitability=["all_season"],
        time_suitability={
            "weekday": ["morning", "evening"],
            "weekend": ["morning", "evening"]
        },
        region_suitability=["north", "south"],
        family_features=["has_children", "has_elderly"],
        lifestyle_features={
            "cooking": ["rare", "medium"],  # 匹配偶尔/中等烹饪频率
            "work_schedule": ["night_shift"]  # 适配夜班
        }
    ),
    HomeAppliance(
        name="电饭煲",
        scene="烹饪场景",
        keywords=["饿", "吃", "做饭"],
        season_suitability=["all_season"],
        time_suitability={
            "weekday": ["morning", "evening"],
            "weekend": ["morning", "evening"]
        },
        region_suitability=["north", "south"],
        family_features=["has_children", "has_elderly"],
        lifestyle_features={
            "cooking": ["frequent", "medium"],  # 匹配偶尔/中等烹饪频率
            "work_schedule": ["day_shift", "night_shift"]  # 适配夜班
        }
    ),
    HomeAppliance(
        name="烧水壶",
        scene="烹饪场景",
        keywords=["烧水", "渴", "喝"],
        season_suitability=["winter", "all_season"],
        time_suitability={
            "weekday": ["morning"],
            "weekend": ["morning"]
        },
        region_suitability=["north", "south"],
        family_features=["has_elderly"],
        lifestyle_features={}
    ),

    # --------------------- 影音娱乐类 ---------------------
    HomeAppliance(
        name="电视",
        scene="娱乐场景",
        keywords=["看剧", "歌", "观影"],
        season_suitability=["all_season"],
        time_suitability={
            "weekday": ["evening"],
            "weekend": ["daytime", "evening"]
        },
        region_suitability=["north", "south"],
        family_features=["has_children", "has_elderly"],
        lifestyle_features={}
    ),
    HomeAppliance(
        name="投影仪",
        scene="娱乐场景",
        keywords=["看剧", "观影"],
        season_suitability=["all_season"],
        time_suitability={
            "weekday": ["evening"],
            "weekend": ["daytime", "evening"]
        },
        region_suitability=["north", "south"],
        family_features=["has_children"],
        lifestyle_features={}
    ),

    # --------------------- 睡眠场景类 ---------------------
    HomeAppliance(
        name="灯光",
        scene="默认场景",
        keywords=["灯", "困", "睡", "床", "暗", "亮"],
        season_suitability=["all_season"],
        time_suitability={
            "weekday": ["morning", "night"],
            "weekend": ["morning", "night"]
        },
        region_suitability=["north", "south"],
        family_features=["has_children", "has_elderly"],
        lifestyle_features={}
    ),
    HomeAppliance(
        name="窗帘",
        scene="默认场景",
        keywords=["起床", "暗", "亮"],
        season_suitability=["all_season"],
        time_suitability={
            "weekday": ["morning"],
            "weekend": ["morning"]
        },
        region_suitability=["north", "south"],
        family_features=["has_children"],
        lifestyle_features={}
    ),

    # --------------------- 其他设备 ---------------------
    HomeAppliance(
        name="插座",
        scene="默认场景",
        keywords=[],  # 通过场景映射触发
        season_suitability=["all_season"],
        time_suitability={
            "weekday": ["all_time"],
            "weekend": ["all_time"]
        },
        region_suitability=["north", "south"],
        family_features=["has_children"],
        lifestyle_features={}
    ),
    HomeAppliance(
        name="自动喂食器",
        scene="宠物喂食",
        keywords=["喂"],
        season_suitability=["all_season"],
        time_suitability={
            "weekday": ["morning", "night", "daytime"],
            "weekend": ["morning", "night", "daytime"]
        },
        region_suitability=["north", "south"],
        family_features=["has_pet"],
        lifestyle_features={
            "work_schedule": ["night_shift", "flexible"]
        }
    ),
    HomeAppliance(
        name="恒温器",
        scene="温度调节",
        keywords=["热", "冷", "凉"],
        season_suitability=["all_season"],
        time_suitability={
            "weekday": ["all_time"],
            "weekend": ["all_time"]
        },
        region_suitability=["north", "south"],
        family_features=["has_children", "has_elderly"],
        lifestyle_features={}
    )
]

PROMPT = """
你是一个专业家居设备识别系统，请严格从以下设备中选择最匹配的一项或多项：
[{device_names}]

用户指令：「{user_input}」

精准匹配规则：
1. 温度调节 → 空调/电风扇/暖气/电热毯/浴霸
2. 用水相关 → 热水器/净水器/智能马桶
3. 清洁需求 → 扫地机器人/洗衣机/烘干机/热水器
4. 空气管理 → 空气净化器/加湿器/除湿机/新风系统
5. 安防相关 → 智能门锁/摄像头
6. 厨房电器 → 冰箱/烤箱/微波炉/电饭煲
7. 影音娱乐 → 电视/投影仪
8. 睡眠场景 → 灯光/空调/电热毯

注意：
1.仅返回设备名称，按python列表格式返回，不要解释
2.不确定时返回"未知设备"
3.结合当前季节和时间信息进行推荐
4.结合用户画像进行推荐
5.示例正确格式：["灯光", "空调", "电视"]
"""