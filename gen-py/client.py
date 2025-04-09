import sys
import json
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from api_ai import ChatService
from api_ai.ttypes import AIChatRequest, FirstAIChatRequest


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


def test_first_chat(client):
    """测试第一次聊天请求"""
    print("\n=== 测试第一次请求 ===")
    request = FirstAIChatRequest(
        input_text="你好",
        language="zh-CN",
        timestamp=1234567890
    )
    try:
        response = client.FirstAIChat(request)
        print(f"响应: {response.scene}")
        return True
    except Exception as e:
        print(f"请求失败: {e}")
        return False


def test_ai_chat(client, text):
    """测试常规AI聊天请求"""
    print(f"\n=== 测试请求: {text} ===")
    request = AIChatRequest(
        input_text=text,
        language="zh-CN",
        timestamp=1234567890
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


def interactive_test(client):
    """交互式测试"""
    print("\n=== 交互模式 ===")
    print("输入 'exit' 退出")
    while True:
        text = input("\n请输入测试内容: ").strip()
        if text.lower() in ('exit', 'quit'):
            break
        test_ai_chat(client, text)


def main():
    # 连接服务器
    client, transport = connect_to_server()

    try:
        # 测试第一次请求
        if not test_first_chat(client):
            return

        # 测试预设用例
        test_cases = [
            "打开客厅的灯",
            "把空调调到26度",
            "浴室太冷了",
            "我不在家的时候请打开监控"
        ]

        for case in test_cases:
            if not test_ai_chat(client, case):
                break

        # 交互式测试
        interactive_test(client)

    finally:
        # 关闭连接
        transport.close()
        print("\n连接已关闭")


if __name__ == "__main__":
    main()