# -*- coding: utf-8 -*-
{
    'name': 'Disable Expiration Check',
    'version': '1.0',
    'summary': 'Bypass expiration date update from Odoo license server',
    'category': 'Tools',
    'author': 'Suirad',
    'license': 'LGPL-3',
    'depends': ['base','mail'],
    'data': [
        # 'data/expiration_cron.xml',
        'views/test_button.xml',
    ],
    'installable': True,
    'application': False,
    'auto_install': False,
    'post_init_hook': 'post_init_hook',
}
