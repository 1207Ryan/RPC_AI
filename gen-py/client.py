import sys
import json
import time
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from api_ai import ChatService
from api_ai.ttypes import AIChatRequest, FirstAIChatRequest, ChatUserProfile
from voice import VoiceRecognition


def connect_to_server(host='localhost', port=9090):
    """建立与Thrift服务器的连接"""
    try:
        # 创建传输层
        transport = TSocket.TSocket(host, port)
        transport = TTransport.TBufferedTransport(transport)

        # 创建协议层
        protocol = TBinaryProtocol.TBinaryProtocol(transport)

        # 创建客户端
        client = ChatService.Client(protocol)

        # 打开连接
        transport.open()
        print(f"成功连接到服务器 {host}:{port}")
        return client, transport
    except Exception as e:
        print(f"连接服务器失败: {e}")
        sys.exit(1)


def test_first_chat(client, uid: int):
    """测试第一次聊天请求"""
    print("\n=== 测试第一次请求 ===")
    request = FirstAIChatRequest(
        input_text="你好",
        language="zh-CN",
        timestamp=int(time.time()),
        uid=uid,
    )
    try:
        response = client.FirstAIChat(request)
        print(f"响应: {response.scene}")
        print(response.needed_init)
        if response.needed_init is True:
            print(client.InitUserProfile(initialize_user_profile(uid)))

        return True
    except Exception as e:
        print(f"请求失败: {e}")
        return False


def test_ai_chat(client, text, uid: int):
    """测试常规AI聊天请求"""
    print(f"\n=== 测试请求: {text} ===")
    if "语音识别" in text:
        text = VoiceRecognition()
    # elif "语音文件" in text:
    #     text = VoiceFileRecognition()
    request = AIChatRequest(
        input_text=text,
        language="zh-CN",
        timestamp=int(time.time()),
        uid=uid,
    )
    try:
        response = client.AIChat(request)
        print(f"基础回复: {response.reply_text}")
        print("匹配场景:")
        for scene in response.scenes:
            print(f" - 场景: {scene.scene_name}, 组件: {scene.matched_component}, 布局: {scene.layout_fragment}")
        print(f"完整布局: {response.assemble_layout}")
        return True
    except Exception as e:
        print(f"请求失败: {e}")
        return False


def initialize_user_profile(uid: int):
    """初始化用户画像"""

    User_Profile = ChatUserProfile(uid=uid)
    print(f"\n===== 用户 {uid} 配置向导 =====")
    print("请回答以下问题来初始化您的个人资料 (直接回车可跳过问题)\n")

    # Basic Information
    User_Profile.age = int(input("1. 您的年龄: "))
    print("\n2. 您的性别 (男/女): ")
    print("  1) 男")
    print("  2) 女")
    sex = input("请选择(1-2): ")
    User_Profile.gender = ["male", "female"][int(sex) - 1] if sex in "12" else "male"

    print("\n3. 您所在的地区 (north/south): ")
    print("  1) 北方")
    print("  2) 南方")
    location = input("请选择(1-2): ")
    User_Profile.region = ["north", "south"][int(location) - 1] if location in "12" else "south"

    # Family Information
    User_Profile.family_members = int(input("4. 家庭成员数量: "))
    User_Profile.has_children = input("5. 家中有小孩吗? (y/n): ").lower() == 'y'
    User_Profile.has_elderly = input("6. 家中有老人吗? (y/n): ").lower() == 'y'
    User_Profile.has_pet = input("7. 家中有宠物吗? (y/n): ").lower() == 'y'

    # Lifestyle
    print("\n8. 您的工作时间:")
    print("  1) 朝九晚五 (regular)")
    print("  2) 夜班 (night_shift)")
    print("  3) 灵活工作时间 (flexible)")
    work_choice = input("请选择(1-3): ")
    User_Profile.work_schedule = ["regular", "night_shift", "flexible"][
        int(work_choice) - 1] if work_choice in "123" else "regular"

    print("\n9. 您的烹饪频率:")
    print("  1) 很少做饭 (rare)")
    print("  2) 偶尔做饭 (medium)")
    print("  3) 经常做饭 (frequent)")
    cook_choice = input("请选择(1-3): ")
    User_Profile.cooking_habits = ["rare", "medium", "frequent"][
        int(cook_choice) - 1] if cook_choice in "123" else "medium"
    return User_Profile


def interactive_test(client, uid: int):
    """交互式测试"""
    print("\n=== 交互模式 ===")
    print("输入 'exit' 退出")
    while True:
        text = input("\n请输入测试内容: ").strip()
        if text.lower() in ('exit', 'quit'):
            break
        test_ai_chat(client, text, uid)


def main():
    # 连接服务器
    client, transport = connect_to_server()

    uid = int(input("请输入您的用户ID: "))

    try:
        # 测试第一次请求
        if not test_first_chat(client, uid):
            return

        # 测试预设用例
        test_cases = [
            "打开客厅的灯",
            "把空调调到26度",
            "浴室太冷了",
            "我不在家的时候请打开监控"
        ]

        for case in test_cases:
            if not test_ai_chat(client, case, uid):
                break

        # 交互式测试
        interactive_test(client, uid)

    finally:
        # 关闭连接
        transport.close()
        print("\n连接已关闭")


if __name__ == "__main__":
    main()
