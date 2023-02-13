from pint import UnitRegistry
ureg = UnitRegistry()


class Unit:
    # Resistor
    OHM = ureg.ohm
    # Current
    AMPERE = ureg.ampere
    # Voltage
    VOLT = ureg.volt
    # Capacitor
    FARADAY = ureg.faraday

    # Hour
    HOUR = ureg.hour
    # Minute
    MINUTE = ureg.minute
    # Second
    SECOND = ureg.second


    # TODO: posso fare una classe che legge da yaml per decidere quali unita di misura utilizzare.
    #       In quel caso portei fare dei dizionari come attributi della classe

    def __init__(self):
        self.resistor = {
            'resistance': self.OHM,
            'current': self.AMPERE,
            'potential': self.VOLT
        }