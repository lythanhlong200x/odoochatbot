from odoo import http
from odoo.http import request

class SalesOrderController(http.Controller):
    @http.route('/ai_chat/create_sales_order', type='json', auth='user', website=True)
    def create_sales_order(self, customer_id, product_ids, quantities):
        """
        Tạo một sales order cho khách hàng với danh sách sản phẩm và số lượng tương ứng.
        """
        try:
            # Lấy environment
            env = request.env
            # Tạo đơn hàng
            order = env['sale.order'].sudo().create({
                'partner_id': customer_id,
            })
            # Tạo các dòng đơn hàng
            order_lines = []
            for prod_id, qty in zip(product_ids, quantities):
                order_lines.append((0, 0, {
                    'product_id': prod_id,
                    'product_uom_qty': qty,
                }))
            order.write({'order_line': order_lines})
            return {"status": "success", "order_id": order.id}
        except Exception as e:
            return {"status": "error", "message": str(e)}
