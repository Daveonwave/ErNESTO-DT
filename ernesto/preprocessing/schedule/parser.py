import logging
import ply.yacc as yacc
from src.preprocessing.schedule.lexer import ScheduleLexer


class ScheduleParser:
    """

    """
    def __init__(self, lexer):
        self._parser = yacc.yacc(module=self)
        self._lexer = lexer
        self._schedule_dict = {}

    tokens = ScheduleLexer.tokens

    def parse(self, inputs, debug=0):
        """
        Parse the input string, debug is used to check the parser states and reductions.
        """
        self._schedule_dict = {}
        p = self._parser.parse(inputs, debug=debug)
        return self._schedule_dict, p

    # Parsing rules
    def p_command_discharge(self, p):
        """
        command : DISCHARGE what
        """
        p[0] = f"Discharge {p[2]}"
        self._schedule_dict['action'] = 'discharge'

    def p_command_charge(self, p):
        """
        command : CHARGE what
        """
        p[0] = f"Charge {p[2]}"
        self._schedule_dict['action'] = 'charge'

    def p_command_rest(self, p):
        """
        command : REST time
        """
        p[0] = f"Rest {p[2]}"
        self._schedule_dict['action'] = 'rest'

    def p_command_hold(self, p):
        """
        command : HOLD what
        """
        p[0] = f"Hold {p[2]}"
        self._schedule_dict['action'] = 'hold'

    def p_command_what(self, p):
        """
        what : AT rate time
        """
        p[0] = f"at {p[2]} {p[3]}"

    def p_rate(self, p):
        """
        rate : CRATE RATIO NUMBER
             | NUMBER CRATE
             | NUMBER UNIT
        """
        if len(p) == 4:
            p[0] = f"{p[1]} {p[2]} {p[3]}"
            self._schedule_dict['rate'] = {'value': float(p[3]), 'unit': p[1]}

        else:
            p[0] = f"{p[1]} {p[2]}"
            self._schedule_dict['rate'] = {'value': float(p[1]), 'unit': p[2]}

    def p_until_rate(self, p):
        """
        until_rate : UNTIL CRATE RATIO NUMBER
                   | UNTIL NUMBER CRATE
                   | UNTIL NUMBER UNIT
        """
        if len(p) == 5:
            p[0] = f"until {p[2]} {p[3]} {p[4]}"
            self._schedule_dict['until_rate'] = {'value': float(p[4]), 'unit': p[2]}

        else:
            p[0] = f"until {p[2]} {p[3]}"
            self._schedule_dict['until_rate'] = {'value': float(p[2]), 'unit': p[3]}

    def p_time(self, p):
        """
        time : duration
             | until_rate
             | duration OR until_rate
             | until_rate OR duration
        """
        if len(p) == 4:
            p[0] = f"{p[1]} or {p[3]}"
        else:
            p[0] = f"{p[1]}"

    def p_for_duration(self, p):
        """
        duration : FOR NUMBER HOURS
                 | FOR NUMBER MINUTES
                 | FOR NUMBER SECONDS
        """
        p[0] = f"for {p[2]} {p[3]}"

        if p[3].startswith("hour"):
            value = float(p[2]) * 3600
        elif p[3].startswith("minute"):
            value = float(p[2]) * 60
        else:
            value = float(p[2])

        self._schedule_dict['duration'] = value

    def p_error(self, p):
        # get formatted representation of stack
        stack_state_str = ' '.join([symbol.type for symbol in self._parser.symstack][1:])
        logging.error('Syntax error in input! Parser State:{} {} . {}'.format(self._parser.state,
                                                                              stack_state_str, p))

