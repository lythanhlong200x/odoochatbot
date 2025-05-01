# region imports 
import warnings
import json
from odoo import fields
from odoo import SUPERUSER_ID
warnings.filterwarnings("ignore", category=DeprecationWarning)

import logging
import re
_logger = logging.getLogger(__name__)

import asyncio

from odoo.api import Environment
from odoo.modules.registry import Registry
import requests
import os
import joblib
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PKL_PATH = os.path.join(BASE_DIR, "static", "data", "intent_classifier_advanced.pkl")

print("Đang load file tại:", PKL_PATH)
# Tải mô hình đã huấn luyện
model = joblib.load(PKL_PATH)

def predict_order_type(message):
    """
    Dự đoán loại đơn hàng từ câu hỏi của người dùng.
    """
    prediction = model.predict([message])
    return prediction[0]

def build_prompt(message):
    print("build_prompt was called")

    order_type = predict_order_type(message)

    if order_type == "create_sales_order":
        hidden_instruction = (
            "You are an Odoo assistant.\n"
            "If the user wants to create a Sales Order, respond ONLY with a JSON in this format:\n"
            "{\n"
            '  "customer_id": "Customer name",\n'
            '  "product_ids": ["Product A", "Product B"],\n'
            '  "quantities": [2, 3]\n'
            "}\n"
            "Do not explain, do not include extra text or markdown.\n"
            "Only return a valid clean JSON object.\n"
            "If the user's request is not about sales order, answer normally.\n\n"
        )
        return hidden_instruction + message

    elif order_type == "create_purchase_order":
        hidden_instruction = (
            "You are an Odoo assistant.\n"
            "If the user wants to create a Purchase Order, respond ONLY with a JSON in this format:\n"
            "{\n"
            '  "supplier_id": "Supplier name",\n'
            '  "product_ids": ["Product A", "Product B"],\n'
            '  "quantities": [5, 10]\n'
            "}\n"
            "Do not explain, do not include extra text or markdown.\n"
            "Only return a valid clean JSON object.\n"
            "If the user's request is not about purchase order, answer normally.\n\n"
        )
        return hidden_instruction + message
    elif order_type == "check_stock":
        hidden_instruction = (
            "You are an Odoo assistant.\n"
            "If the user is asking to check product stock or inventory, respond ONLY with the product name they want to check.\n"
            "For example:\n"
            "\"FRISOLAC 1 400G\"\n"
            "Return only the clean product name. No explanation, no markdown, no JSON.\n"
            "If multiple products are mentioned, just return the main one.\n"
        )
        return hidden_instruction + message
    elif order_type == "find_order":
        hidden_instruction = (
            "You are an Odoo assistant.\n"
            "If the user wants to find or review an order, return ONLY the order number or customer name used to search.\n"
            "Examples:\n"
            "\"S00025\"\n"
            "or\n"
            "\"PLUS ONE CONFINEMENT CENTRE\"\n"
            "No other formatting, no markdown, just return the search keyword.\n"
            "Do not explain anything.\n"
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

def process_purchase_response(response_text):
    # In ra debug tương tự
    text = response_text if isinstance(response_text, str) else \
        response_text.get('candidates',[{}])[0]\
                     .get('content',{})\
                     .get('parts',[{}])[0]\
                     .get('text','')
    cleaned = text.replace('```json\n','').replace('```','')
    data = json.loads(cleaned)
    print(f"Parsed purchase data: {data}")
    if all(k in data for k in ("supplier_id","product_ids","quantities")):
        return {
            "supplier_id": data["supplier_id"],
            "product_ids":  data["product_ids"],
            "quantities":   data["quantities"],
        }
    return None


def clean_response_text(response_text):
    """
    Làm sạch response từ Gemini (dạng text thường, không phải JSON).
    Trích phần cần dùng để tra cứu (tên sản phẩm hoặc mã đơn).
    """
    print(f"Raw Gemini response: {response_text}")

    try:
        # Nếu là dict (kiểu response chuẩn từ Gemini)
        if isinstance(response_text, dict):
            response_text = response_text.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '')

        # Xử lý markdown và ký tự thừa
        cleaned = response_text.strip()
        cleaned = cleaned.replace('```', '')
        cleaned = cleaned.replace('"', '')
        cleaned = cleaned.replace('\n', ' ')
        cleaned = cleaned.strip()

        print(f"Cleaned keyword: {cleaned}")
        return cleaned

    except Exception as e:
        print(f"Error in clean_response_text: {e}")
        return ""


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
            intent = predict_order_type(message)
            order_result = None
            keyword = clean_response_text(data)
            if intent == "create_sales_order":
                print("create_sales_order was called")
                sales_order_params = process_order_response(data)
                if sales_order_params:
                    order_result = self.create_sales_order(
                        sales_order_params["customer_id"],
                        sales_order_params["product_ids"],
                        sales_order_params["quantities"],
                        order_type="sale"
                    )
            elif intent == "create_purchase_order":
                print("create_purchase_order was called")
                # Xử lý tạo Purchase Order
                purchase_order_params = process_purchase_response(data)
                if purchase_order_params:
                    order_result = self.create_purchase_order(
                        purchase_order_params["supplier_id"],
                        purchase_order_params["product_ids"],
                        purchase_order_params["quantities"],
                    )
            elif intent == "check_stock":
                return self.check_product_stock(keyword)
            if intent == "find_by_number":
                return (
                        "[FindByNumber]\n"
                        "Return ONLY JSON: {\"model\": \"sale.order\" or \"purchase.order\", \"order_number\": \"S00032\"}\n"
                        "[/FindByNumber]\n" + message
                )
            elif intent == "find_by_partner":
                return (
                        "[FindByPartner]\n"
                        "Return ONLY JSON: {\"model\": \"sale.order\", \"partner\": \"Customer Name\"}\n"
                        "[/FindByPartner]\n" + message
                )
            elif intent == "list_recent":
                return (
                        "[ListRecent]\n"
                        "Return ONLY JSON: {\"model\": \"sale.order\", \"limit\": 5}\n"
                        "[/ListRecent]\n" + message
                )
            if "candidates" in data and len(data["candidates"]) > 0:
                candidate = data["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"] and len(
                        candidate["content"]["parts"]) > 0:
                    base_text = candidate["content"]["parts"][0]["text"]
                    if order_result:
                        base_text += f"\n\n✅ {order_result}"

                    self.save_chat_history(self.env.uid, message, base_text)
                    return base_text
            return "No response from Gemini API."
        except Exception as e:
            _logger.error("Error calling Gemini API: %s", e)
            return f"Error calling Gemini API: {e}"

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
            company_id = customer.company_id.id if customer.company_id else self.env.company.id
            order_model = "sale.order" if order_type == "sale" else "purchase.order"

            # Dùng SUPERUSER_ID để vượt quyền, rồi thêm context đúng công ty vào model, không phải env
            order_model_obj = self.env[order_model].sudo().with_user(SUPERUSER_ID).with_context(force_company=company_id)
            _logger.info("Creating %s for partner %s in company %s", order_model, customer.id, company_id)
            # Tạo đơn hàng
            order = order_model_obj.create({
                'partner_id': customer.id,
                'company_id': company_id,
                'user_id': self.env.uid,
            })
            _logger.info("Order created with ID %s", order.id)
            # Tìm và thêm sản phẩm vào order
            order_lines = []
            for prod_name, qty in zip(product_names, quantities):
                product = self.get_record_by_name("product.product", "name", prod_name)
                if not product:
                    print(f"Product '{prod_name}' not found")
                    return f"Product '{prod_name}' not found"
                field = 'product_uom_qty' if order_type == "sale" else 'product_qty'
                order_lines.append((0, 0, {'product_id': product.id, field: qty}))

            # Ghi lại các dòng đơn hàng vào đơn hàng
            order.write({'order_line': order_lines})
            model_label = "Sales Order" if order_type == "sale" else "Purchase Order"
            return f"✅ {model_label} {order.name} created successfully."
        except Exception as e:
            print(f"Error creating sales order: {e}")
            return f"Error: {e}"

    def create_purchase_order(self, supplier_name, product_names, quantities):
        print("caleddddddddddddddddddddddddddddddddddddddd")
        return self.create_sales_order(supplier_name, product_names, quantities, order_type="purchase")

    def check_product_stock(self, product_name):
        """
        Check the current on-hand and forecasted stock for a given product name.
        Returns a formatted string with both quantities.
        """
        # Lấy recordset product với quyền sudo
        Product = self.env['product.product'].sudo()

        # Tìm sản phẩm theo tên (ilike)
        product = Product.search([('name', 'ilike', product_name)], limit=1)
        if not product:
            return f"❌ Product '{product_name}' not found."

        # Đọc hai field qty_available và virtual_available
        on_hand = product.qty_available
        forecasted = product.virtual_available

        return (
            f"📦 Product: {product.name}\n"
            f"  • On Hand: {on_hand:.2f}\n"
            f"  • Forecasted: {forecasted:.2f}\n"
        )

    def find_by_number(self, model, number):
        Order = self.env[model].sudo()
        rec = Order.search([("name", "=", number)], limit=1)
        if not rec:
            return f"No order {number} found."
        return f"{rec.name}: Customer={rec.partner_id.name}, Total={rec.amount_total:.2f}, Status={rec.state}"

    def find_by_partner(self, model, partner_name):
        Order = self.env[model].sudo()
        recs = Order.search([("partner_id.name", "ilike", partner_name)], limit=10)
        if not recs:
            return f"No orders for {partner_name}."
        return "\n".join(f"- {r.name} ({r.state}): {r.amount_total:.2f}" for r in recs)

    def list_recent(self, model, limit):
        Order = self.env[model].sudo()
        recs = Order.search([], order="date_order desc", limit=limit)
        if not recs:
            return "No recent orders."
        return "\n".join(f"- {r.name} ({r.state}): {r.amount_total:.2f}" for r in recs)
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
