from src.preprocessing.schedule.lexer import ScheduleLexer
from src.preprocessing.schedule.parser import ScheduleParser
from src.preprocessing.data_preparation import validate_parameters_unit


class Schedule:
    """

    """
    def __init__(self, instructions: [str], c_value: float):
        """

        Args:
            instructions ():
        """
        self._instructions = instructions
        self._lexer = ScheduleLexer()
        self._parser = ScheduleParser(self._lexer)

        self._C_value = c_value
        self._command_buffer = []

        self._traslate_cmds()

    def _traslate_cmds(self):
        """
        Returns:
        """
        for sentence in self._instructions:
            info, sentence = self._parser.parse(sentence)
            cmd = {'sentence': sentence}

            if 'rate' in info:
                if info['rate']['unit'].endswith('V'):
                    info['rate']['var'] = 'voltage'
                    cmd['load'] = validate_parameters_unit({'voltage': info['rate']})

                elif info['rate']['unit'].endswith('A'):
                    info['rate']['var'] = 'current'
                    cmd['load'] = validate_parameters_unit({'current': info['rate']})

                elif info['rate']['unit'].endswith('W'):
                    info['rate']['var'] = 'power'
                    cmd['load'] = validate_parameters_unit({'power': info['rate']})

                else:
                    cmd['load'] = {'current': round(self._C_value / info['rate']['value'], 4)}

            if 'action' in info:
                if info['action'] == 'discharge':
                    if 'current' in cmd['load']:
                        cmd['load']['current'] *= -1
                    else:
                        cmd['load']['power'] *= -1

                if info['action'] == 'rest':
                    cmd['load'] = {'current': 0}

                cmd['action'] = info['action']

            if 'duration' in info:
                cmd['duration'] = info['duration']

            if 'until_rate' in info:
                if info['until_rate']['unit'].endswith('V'):
                    info['until_rate']['var'] = 'voltage'
                    cmd['until_cond'] = validate_parameters_unit({'voltage': info['until_rate']})

                elif info['until_rate']['unit'].endswith('A'):
                    info['until_rate']['var'] = 'current'
                    cmd['until_cond'] = validate_parameters_unit({'current': info['until_rate']})

                elif info['until_rate']['unit'].endswith('W'):
                    info['until_rate']['var'] = 'power'
                    cmd['until_cond'] = validate_parameters_unit({'power': info['until_rate']})

                else:
                    cmd['until_cond'] = {'current': round(self._C_value / info['until_rate']['value'], 4)}

            self._command_buffer.append(cmd)

    def get_cmd(self):
        """

        Returns:

        """
        return self._command_buffer[0]

    def is_empty(self):
        """
        Returns True if the command buffer is empty, False otherwise
        """
        return len(self._command_buffer) == 0

    def reset_cmd_buffer(self):
        self._command_buffer = []

    def next_cmd(self):
        """

        Returns:

        """
        self._command_buffer.pop(0)



if __name__ == '__main__':
    schedule = Schedule()
    exit()

    # Examples
    examples = [
        "Hold at 10 mV for 20 seconds",
        "Discharge at 1 C for 0.5 hour",
        "Discharge at C/20 for 0.5 hours",
        "Charge at 0.5 C for 45 minutes",
        "Discharge at 1 A for 0.5 hours",
        "Charge at 200 mA for 45 minutes",
        "Discharge at 1 W for 0.5 hours",
        "Charge at 200 mW for 45 minutes",
        "Rest for 10 minutes",
        "Charge at 1 C until 4.1V",
        "Hold at 4.1 V until 50mA",
        "Hold at 3 V until C/50",
        "Discharge at C/3 for 2 hours or until 2.5 V"
    ]

    # Test the parser
    for example in examples:
        info_data, result = schedule._parser.parse(example, debug=0)
        print(f"Input: {example}\nParsed Command: {result}\n")
        print(info_data)

