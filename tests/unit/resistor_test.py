import unittest
from pint import UnitRegistry
ureg = UnitRegistry()
from ernesto.digital_twin.battery_models.electrical.ecm_components.resistor import Resistor


class ResistorTest(unittest.TestCase):
    def test_param_as_int(self):
        param = 5
        resistor = Resistor('name', param)
        self.assertEqual(str(resistor.resistance.units), str(ureg.ohm))
        self.assertEqual(resistor.resistance.magnitude, 5.)

    def test_param_as_float(self):
        param = 5.1
        resistor = Resistor('name', param)
        self.assertEqual(str(resistor.resistance.units), str(ureg.ohm))
        self.assertEqual(resistor.resistance.magnitude, 5.1)

    def test_param_as_quantity_int(self):
        param = 5 * ureg.ohm
        resistor = Resistor('name', param)
        self.assertEqual(str(resistor.resistance.units), str(ureg.ohm))
        self.assertEqual(resistor.resistance.magnitude, 5.)

    def test_param_as_quantity_float(self):
        param = 5.1 * ureg.ohm
        resistor = Resistor('name', param)
        self.assertEqual(str(resistor.resistance.units), str(ureg.ohm))
        self.assertEqual(resistor.resistance.magnitude, 5.1)

    def test_param_as_quantity_with_wrong_unit_and_correct_dimensionality(self):
        param = 5.1 * ureg.milliohm
        resistor = Resistor('name', param)
        self.assertEqual(str(resistor.resistance.units), str(ureg.ohm))
        self.assertEqual(round(resistor.resistance.magnitude, 4), 0.0051)

    def test_param_as_quantity_with_wrong_unit_and_dimensionality(self):
        param = 5.1 * ureg.meter
        self.assertRaises(Exception, Resistor, 'name', param)

    def test_param_of_wrong_type(self):
        param = 'wrong_type'
        self.assertRaises(Exception, Resistor, 'name', param)

    def test_compute_potential(self):
        param = 5 * ureg.ohm
        resistor = Resistor('name', param)
        curr = 10 * ureg.ampere
        self.assertEqual(resistor.compute_v(curr), 50.0 * ureg.volt)

    def test_compute_current(self):
        param = 5 * ureg.ohm
        resistor = Resistor('name', param)
        voltage = 10 * ureg.volt
        self.assertEqual(resistor.compute_i(voltage), 2.0 * ureg.ampere)

if __name__ == '__main__':
    unittest.main()