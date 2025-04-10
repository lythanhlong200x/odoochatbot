# -*- coding: utf-8 -*-

from odoo import http
from odoo.http import request

from .ai.ai_bot import AiBot

class AiChatController(http.Controller):
    @http.route('/ai_chat/chat', type='json', auth='user', website=True)
    def get_quickboard_item(self, message, history, streaming):
        env = request.env
        bot = AiBot(env)
        res = bot.chat(message, history, streaming)
        return res

class AiChatHistoryController(http.Controller):
    @http.route('/ai_chat/history', type='json', auth='user')
    def get_history(self):
        records = request.env['ai.chat.history'].sudo().search([('user_id', '=', request.env.uid)], order='date asc')
        result = []
        for rec in records:
            result.append({
                "role": "user",
                "parts": [{"text": rec.message}]
            })
            # assistant response
            result.append({
                "role": "assistant",
                "parts": [{"text": rec.response}]
            })
        return result
