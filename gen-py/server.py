from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer
from api_ai import ChatService
from api_ai.ttypes import AIChatResponse, SceneInfo, FirstAIChatResponse
import json
import logging
from main import process_input, UserProfile  # Import your existing AI processing functions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChatServiceHandler:
    def __init__(self):
        # Initialize with default user profile or load from file
        try:
            self.user_profile = UserProfile.load_from_file("user_profile.json")
        except:
            self.user_profile = UserProfile()  # Default profile
            logger.warning("Using default user profile")

    def AIChat(self, req):
        """Handle regular AI chat requests"""
        logger.info(f"Received AIChat request: {req.input_text}")

        # Process the input using your existing logic
        result = process_input(req.input_text, self.user_profile)
        logger.info(f"AIChat response: {result}")
        # Prepare the response
        response = AIChatResponse()
        response.reply_text = "好的，请稍等"  # Default response text

        # Convert the result to SceneInfo objects
        scenes = []
        if isinstance(result, list):
            for device in result:
                scene = SceneInfo()
                scene.scene_name = "default"  # Default scene if not specified
                scene.matched_component = device
                scene.layout_fragment = "default_layout_fragment"
                scenes.append(scene)
        elif isinstance(result, str):
            scene = SceneInfo()
            scene.scene_name = "default"
            scene.matched_component = result
            scene.layout_fragment = "default_layout_fragment"
            scenes.append(scene)

        response.scenes = scenes
        response.assemble_layout = "default_assemble_layout"

        return response

    def FirstAIChat(self, req):
        """Handle first AI chat request (special case)"""
        logger.info(f"Received FirstAIChat request: {req.input_text}")
        result = process_input(req.input_text, self.user_profile)

        response = FirstAIChatResponse()
        response.scene = "智能家居"  # Always return "智能家居" for first request

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
