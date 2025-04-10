from odoo import models, fields, api

class AiChatHistory(models.Model):
    _name = "ai.chat.history"
    _description = "Chat History for AI Chatbot"  # Lịch sử chat cho chatbot AI

    user_id = fields.Many2one("res.users", string="User", required=True)
    message = fields.Text(string="User Message", required=True)
    response = fields.Text(string="Bot Response", required=True)
    date = fields.Datetime(string="Date", default=fields.Datetime.now, required=True)
