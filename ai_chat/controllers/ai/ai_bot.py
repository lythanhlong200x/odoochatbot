# region imports 
import warnings
import json
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
    keywords = ["sale order", "đặt đơn bán", "tạo đơn bán"," order bán"]
    pattern = re.compile("|".join(keywords), re.IGNORECASE)
    return bool(pattern.search(message))


def is_purchase_order_request(message):
    """
    Kiểm tra xem câu hỏi có liên quan đến việc tạo Purchase Order không.
    """
    # Các từ khóa liên quan đến Purchase Order
    keywords = ["purchase order", "đặt đơn mua", "tạo đơn mua", "order mua"]
    pattern = re.compile("|".join(keywords), re.IGNORECASE)
    return bool(pattern.search(message))

def build_prompt(message):
    print("this was called")
    if is_sales_order_request(message):
        hidden_instruction = (
            "[SalesOrder Instruction]\n"
            "When the user's request is to create a sales order, please respond ONLY with a JSON object in the following format (do not include any extra text):\n"
            "{\n"
            '  "customer_id": <customer_name>,\n'
            '  "product_ids":["<product_name1>", "<product_name2>", ...],\n'
            '  "quantities": [<quantity1>, <quantity2>, ...]\n'
            "}\n"
            "If the request is not about creating a sales order, ignore this instruction and answer normally.\n"
            "[/SalesOrder Instruction]\n\n"
        )
        return hidden_instruction + message
    else:
        return message
# endregion
def process_order_response(response_text):
    print(f"Response text: {response_text}")  # In ra toàn bộ response
    try:
        # Trích xuất phần 'text' trong response (cleaning)
        if isinstance(response_text, dict):
            response_text = response_text.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get(
                'text', '')

        print(f"Cleaned response text: {response_text}")  # In ra text đã được làm sạch

        # Loại bỏ markdown ` ```json` và ` ``` `
        cleaned_text = response_text.replace('```json\n', '').replace('```', '')

        # Parse chuỗi JSON đã làm sạch
        data = json.loads(cleaned_text)
        print(f"Parsed data: {data}")  # In ra dữ liệu JSON đã parse

        # Kiểm tra nếu tất cả các key cần thiết có mặt
        if all(k in data for k in ("customer_id", "product_ids", "quantities")):
            # Kiểm tra nếu customer_id và product_ids hợp lệ
            if data["customer_id"] is None or not data["product_ids"]:
                print("Invalid customer_id or product_ids")
                return None  # Trả về None nếu dữ liệu không hợp lệ
            return data
    except Exception as e:
        print(f"Error processing sales order response: {e}")
        pass
    return None


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
        hidden_instruction = "Short message, not over 512 tokens (refer to Odoo).\n"
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
            if is_sales_order_request(message):
                sales_order_params = process_order_response(data)
                if sales_order_params:
                    order_result = self.create_sales_order(
                        sales_order_params["customer_id"],
                        sales_order_params["product_ids"],
                        sales_order_params["quantities"],
                        order_type="sale"
                    )
            elif is_purchase_order_request(message):
                # Xử lý tạo Purchase Order
                purchase_order_params = process_order_response(data)
                if purchase_order_params:
                    order_result = self.create_purchase_order(
                        purchase_order_params["supplier_name"],
                        purchase_order_params["product_names"],
                        purchase_order_params["quantities"]
                    )

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

    def get_record_by_name(self, model, field_name, search_value):
        """
        Tìm kiếm bất kỳ bản ghi nào trong Odoo theo tên gần đúng.
        """
        record = self.env[model].sudo().search([(field_name, 'ilike', search_value)], limit=1)
        if record:
            print(record)
            return record
        else:
            print(f"{model} '{search_value}' not found.")
            return None

    # def get_customer_by_name(self, customer_name):
    #     """
    #     Tìm khách hàng theo tên trong Odoo.
    #     """
    #     customer = self.env['res.partner'].sudo().search([('name', 'ilike', customer_name)], limit=1)
    #     print(customer)
    #     if customer:
    #         return customer
    #     else:
    #         return None
    #
    # def get_product_by_name(self, product_name):
    #     """
    #     Tìm sản phẩm theo tên trong Odoo.
    #     """
    #     product = self.env['product.product'].sudo().search([('name', 'ilike', product_name)], limit=1)
    #     print(product)
    #     if product:
    #         return product
    #     else:
    #         return None

    def create_sales_order(self, customer_name, product_names, quantities, order_type):
        """
        Tạo Sales Order với dữ liệu khách hàng (theo tên) và sản phẩm (theo tên).
        """
        try:
            # Tìm khách hàng theo tên
            customer = self.get_record_by_name("res.partner", "name", customer_name)
            if not customer:
                print("Customer not found")
                return "Customer not found"
            order_model = "sale.order" if order_type == "sale" else "purchase.order"
            order = self.env[order_model].sudo().create({
                'partner_id': customer.id,
            })
            # Tìm và thêm sản phẩm vào order
            order_lines = []
            for prod_name, qty in zip(product_names, quantities):
                product = self.get_record_by_name("product.product", "name", prod_name)
                if not product:
                    print(f"Product '{prod_name}' not found")
                    return f"Product '{prod_name}' not found"
                order_lines.append((0, 0, {
                    'product_id': product.id,
                    'product_uom_qty': qty,
                }))

            # Ghi lại các dòng đơn hàng vào đơn hàng
            order.write({'order_line': order_lines})
            return f"Sales Order created with ID: {order.id}"
        except Exception as e:
            print(f"Error creating sales order: {e}")
            return f"Error: {e}"

    def create_purchase_order(self, supplier_name, product_names, quantities):
        return self.create_sales_order(supplier_name, product_names, quantities, order_type="purchase")

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
