# -*- coding: utf-8 -*-
from odoo import models, fields, api, SUPERUSER_ID
import logging

_logger = logging.getLogger(__name__)


class PublisherWarrantyContract(models.AbstractModel):
    _inherit = "publisher_warranty.contract"

    def _get_sys_logs(self):
        _logger.info("Bypassed _get_sys_logs – no request sent to Odoo server.")
        return {"messages": [], "enterprise_info": {}}

    def update_notification(self, cron_mode=True):
        _logger.info("Bypassed update_notification – expiration date not updated.")
        return True


class IrConfigParameterExpirationOverride(models.Model):
    _inherit = "ir.config_parameter"

    @api.model
    def set_expiration_date_2099(self):
        _logger.info("Force-set expiration date to 2099-12-31")
        self.sudo().set_param("database.expiration_date", "2099-12-31 23:59:59")


class ExpirationResetCron(models.Model):
    _name = "expiration.reset.cron"
    _description = "Force reset expiration date periodically"

    @api.model
    def cron_force_reset_expiration(self):
        _logger.info("[CRON] Resetting database.expiration_date to 2099-12-31")
        self.env["ir.config_parameter"].set_expiration_date_2099()

