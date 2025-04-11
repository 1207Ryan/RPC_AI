from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer
from api_ai import ChatService
from api_ai.ttypes import AIChatResponse, SceneInfo, FirstAIChatResponse, ChatUserProfile
import json
import logging
from main import *  # Import your existing AI processing functions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChatServiceHandler:
    def InitUserProfile(self, up):
        user_profile = UserProfile(
            user_id=up.uid,
            age=int(up.age) if up.age else None,
            gender=up.gender if up.gender else None,
            region=up.region if up.region else None,
            family_members=int(up.family_members) if up.family_members else 1,
            has_children=up.has_children,
            has_elderly=up.has_elderly,
            has_pet=up.has_pet,
            work_schedule=up.work_schedule,
            cooking_habits=up.cooking_habits,
            device_usage={}  # Start with empty device usage
        )
        user_profile.save_to_file(f"user_profiles/user_{up.uid}.json")
        logger.info("用户配置已完成并保存!")
        return "用户配置已完成并保存!"

    def AIChat(self, req):
        """Handle regular AI chat requests"""
        logger.info(f"Received AIChat request: {req.input_text}")

        try:
            self.user_profile = UserProfile.load_from_file(f"user_profiles/user_{req.uid}.json")
            logger.info("用户文件获取成功")
        except:
            self.user_profile = UserProfile(req.uid)  # Default profile
            logger.info("用户文件获取失败，使用默认文件")

        try:
            result = process_input(req.input_text, self.user_profile)
        except Exception as e:
            logger.error(f"Error processing input: {e}")
            result = ["系统处理出错"]

        response = AIChatResponse()
        response.reply_text = "好的，请稍等"

        # 确保结果转为列表
        result_list = [result] if not isinstance(result, list) else result

        scenes = []
        for device in result_list:
            scene = SceneInfo()
            # 确保设置所有required字段
            scene.scene_name = history.current_scene
            scene.matched_component = str(device) if device else "无匹配设备"
            scene.layout_fragment = "default_layout_fragment"
            scenes.append(scene)

        response.scenes = scenes
        response.assemble_layout = "default_assemble_layout"

        return response

    def FirstAIChat(self, req):
        """Handle first AI chat request (special case)"""
        logger.info(f"Received FirstAIChat request: {req.input_text}")

        response = FirstAIChatResponse()
        response.scene = "智能家居"  # Always return "智能家居" for first request

        file_path = f"user_profiles/user_{req.uid}.json"
        if os.path.exists(file_path):
            response.needed_init = False
        else:
            response.needed_init = True

        return response


def main():
    # Create service handler
    handler = ChatServiceHandler()

    # Create processor
    processor = ChatService.Processor(handler)

    # Set up server
    transport = TSocket.TServerSocket(host="127.0.0.1", port=9090)
    tfactory = TTransport.TBufferedTransportFactory()
    pfactory = TBinaryProtocol.TBinaryProtocolFactory()

    # Create server
    server = TServer.TSimpleServer(processor, transport, tfactory, pfactory)

    logger.info("Starting the server...")
    server.serve()
    logger.info("Server stopped.")


if __name__ == "__main__":
    main()
