from odoo import api, SUPERUSER_ID

def post_init_hook(env):
    # Reset expiration date ngay lập tức
    env['expiration.reset.cron'].cron_force_reset_expiration()
    model = env['ir.model'].search([('model', '=', 'expiration.reset.cron')], limit=1)
    if model:
        if not env['ir.cron'].sudo().search([('name', '=', 'Force Reset Expiration Date')]):
            env['ir.cron'].sudo().create({
                'name': 'Force Reset Expiration Date',
                'model_id': model.id,
                'state': 'code',
                'code': "model.cron_force_reset_expiration()",
                'interval_number': 1,
                'interval_type': 'days',
                'active': True,
            })
