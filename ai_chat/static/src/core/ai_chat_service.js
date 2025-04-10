/** @odoo-module */

import { registry } from "@web/core/registry";
import { rpc } from "@web/core/network/rpc";

const aiChatService = {

    start() {
        const aiChat = {}
        async function chat(message, history, streaming) {
            const res = await rpc("/ai_chat/chat", {
                "message": message,
                "history": history,
                "streaming": streaming,
            });
            return res;
        };
        async function getHistory() {
            const history = await rpc("/ai_chat/history", {});
            return history;
        };
        async function createSalesOrder(customerId, productIds, quantities) {
            return await rpc("/ai_chat/create_sales_order", {
                customer_id: customerId,
                product_ids: productIds,
                quantities: quantities,
            });
        }
        aiChat.chat = chat;
        aiChat.getHistory = getHistory;
        aiChat.createSalesOrder = createSalesOrder;
        return aiChat;
    }
};

registry.category("services").add("ai_chat", aiChatService);
