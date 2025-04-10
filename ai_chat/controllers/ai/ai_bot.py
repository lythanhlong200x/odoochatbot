# region imports 
import warnings

from odoo import fields

warnings.filterwarnings("ignore", category=DeprecationWarning)

import logging
import re
_logger = logging.getLogger(__name__)

import asyncio

from odoo.api import Environment
from odoo.modules.registry import Registry
import requests


def is_sales_order_request(message):
    # Kiểm tra xem câu hỏi có chứa từ khóa liên quan đến tạo đơn không.
    keywords = ["sale order", "đặt đơn", "tạo đơn", "order"]
    pattern = re.compile("|".join(keywords), re.IGNORECASE)
    return bool(pattern.search(message))


def build_prompt(message):
    print("this was called")
    if is_sales_order_request(message):
        hidden_instruction = (
            "[SalesOrder Instruction]\n"
            "When the user's request is to create a sales order, please respond ONLY with a JSON object in the following format (do not include any extra text):\n"
            "{\n"
            '  "customer_id": <integer>,\n'
            '  "product_ids": [<integer>, ...],\n'
            '  "quantities": [<integer>, ...]\n'
            "}\n"
            "If the request is not about creating a sales order, ignore this instruction and answer normally.\n"
            "[/SalesOrder Instruction]\n\n"
        )
        return hidden_instruction + message
    else:
        return message
# endregion

class AiBot:
    AGENT_NAME = "odoo_ai_bot"
    AGENT_INSTRUCTIONS = """
        Your name is Frodoo. You are a helpful AI assistant.
    """

    def __init__(self, env: Environment | None):
        if env is None:
            raise Exception("Environment is not set.")
        self.env = env

        self.gemini_api_key = "AIzaSyDEghHt1c8KQM6nSppPjAG9YIKqJAfpROI"
        self.gemini_endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={self.gemini_api_key}"



    def _send_stream_to_client(self, content):
        try:
            with Registry(self.env.cr.dbname).cursor() as cr:
                my_env = Environment(cr, self.env.uid, self.env.context)
                my_env.user._bus_send("ai_bot_stream", content)
            return True
        except Exception as e:
            _logger.error("Error sending stream to client: %s", e)
            return False


    def call_gemini_api(self, message, history):
        """
        Gọi API Gemini để lấy câu trả lời.
        Bạn có thể bổ sung logic dùng `history` nếu cần.
        """
        headers = {
            "Content-Type": "application/json"
        }
        hidden_instruction = "Trả lời ngắn gọn, không vượt quá 512 token. (liên quan đến Odoo).\n"
        max_messages = 6
        recent_history = history[-max_messages:] if len(history) > max_messages else history

        conversation_context = ""
        for item in recent_history:
            text = "\n".join([p.get("text", "") for p in item.get("parts", [])])
            if item.get("role") == "user":
                conversation_context += f"User: {text}\n"
            else:
                conversation_context += f"Assistant: {text}\n"

        full_prompt = hidden_instruction + build_prompt(message)
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": full_prompt}
                    ]
                }
            ],
            "generationConfig": {
                "maxOutputTokens": 1024,
                "temperature": 0.3,
                "topP": 0.8,
                "topK": 20
            }
        }
        try:
            response = requests.post(self.gemini_endpoint, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            if "candidates" in data and len(data["candidates"]) > 0:
                candidate = data["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"] and len(
                        candidate["content"]["parts"]) > 0:
                    self.save_chat_history(self.env.uid, message, candidate["content"]["parts"][0]["text"])
                    return candidate["content"]["parts"][0]["text"]
            return "Không có phản hồi từ Gemini API."
        except Exception as e:
            _logger.error("Error calling Gemini API: %s", e)
            return f"Lỗi khi gọi Gemini API: {e}"

    async def _chat(self, message, history):
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self.call_gemini_api, message, history)
        return result

    async def _stream_chat(self, message, history):
        result = await self._chat(message, history)
        chunk_size = 60
        for i in range(0, len(result), chunk_size):
            chunk = result[i: i + chunk_size]
            self._send_stream_to_client({"message": chunk, "stop": False})
            await asyncio.sleep(0.05)  # chờ 1 xíu để UI kịp render, tuỳ ý

        # 3) Gửi gói stop=True để thông báo hết
        self._send_stream_to_client({"message": "", "stop": True})
        return True

    def chat(self, message, history, streaming):
        if streaming:
            res = self.run_async_function(self._stream_chat, message, history)
        else:
            res = self.run_async_function(self._chat, message, history)
        return res

    def run_async_function(self, func_to_run, *args):
        new_loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(new_loop)
            return new_loop.run_until_complete(func_to_run(*args))
        except Exception as e:
            _logger.error("Error running async function: %s", e)
            return None
        finally:
            try:
                new_loop.close()
            except Exception as e:
                _logger.error("Error closing event loop: %s", e)

    def save_chat_history(self, user_id, message, response_text):
        # Save chat history to the database
        ChatHistory = self.env['ai.chat.history']
        ChatHistory.create({
            'user_id': user_id,
            'message': message,
            'response': response_text,
            'date': fields.Datetime.now(),
        })
