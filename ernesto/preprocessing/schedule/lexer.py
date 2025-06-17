import logging
import ply.lex as lex


class ScheduleLexer:
    """

    """

    def __init__(self):
        self._lexer = lex.lex(module=self)

    tokens = (
        'DISCHARGE', 'CHARGE', 'REST', 'HOLD', 'AT',
        'NUMBER', 'UNIT', 'MINUTES', 'HOURS', 'RATIO',
        'SECONDS', 'UNTIL', 'FOR', 'OR', 'CRATE',
    )

    # Ignored characters
    t_ignore = ' \t'

    # Token definitions
    t_DISCHARGE = r'Discharge'
    t_CHARGE = r'Charge'
    t_REST = r'Rest'
    t_HOLD = r'Hold'
    t_AT = r'at'
    t_NUMBER = r'\d+(\.\d+)?'
    t_RATIO = r'/'
    t_CRATE = r'C'
    t_UNIT = r'(mV|V|A|mA|W|mW)'
    t_HOURS = r'hours?'
    t_MINUTES = r'minutes?'
    t_SECONDS = r'seconds?'
    t_UNTIL = r'until'
    t_FOR = r'for'
    t_OR = r'or'

    def t_error(self, t):
        logging.error(f"Illegal character '{t.value[0]}' at index {t.lexpos}")
        t.lexer.skip(1)

