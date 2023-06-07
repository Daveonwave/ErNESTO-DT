from typing import Union
from scipy.interpolate import interp1d, interp2d


class GenericVariable:
    """

    """

    def __init__(self, name: str):
        self._name = name

    @property
    def name(self):
        return self._name

    def get_value(self, input_vars: dict):
        raise NotImplementedError

    def set_value(self, new_value):
        raise NotImplementedError


class Scalar(GenericVariable):
    """

    """

    def __init__(self, name: str, value: Union[int, float]):
        super().__init__(name)
        self._value = value

    def get_value(self, input_vars: dict = None):
        return self._value


class Function:
    """

    """
    pass


# TODO: understand how to handle Parametric Functions
class FunctionTerm:
    """

    """

    def __init__(self, variable, coefficient, operation, degree):
        self._variable = variable
        self._coefficient = coefficient
        self._operation = operation
        self._degree = degree


class ParametricFunction(GenericVariable):
    """

    """

    def __init__(self, name: str, function_terms: dict):
        super().__init__(name)
        self.function_terms = function_terms
        self._check_function_terms_format()

        self.input_vars = function_terms.keys()

    def _check_function_terms_format(self):
        pass

    def get_value(self, **params):
        result = 0
        for j, var in enumerate(self.input_vars):
            degrees = [deg for deg in range(len(self.coefficients[j]))]


class LookupTableFunction(GenericVariable):
    """

    """

    def __init__(self, name: str, y_values: list, x_names: list, x_values: list):
        super().__init__(name)

        self.y_values = y_values
        self.x_names = x_names
        self.x_values = x_values

        self.function = None

        if len(x_names) == 1:
            self.function = interp1d(x_values[0], y_values, fill_value='extrapolate')

        elif len(x_names) == 2:
            self.function = interp2d(x_values[0], x_values[1], y_values)

        else:
            raise Exception("Too many variables to interpolate, not implemented yet!")

    def get_value(self, input_vars: dict):
        """

        """
        input_values = []

        for expected_input, given_input in zip(self.x_names, input_vars.keys()):

            if expected_input != given_input:
                raise Exception("Given inputs aren't correct for the computation of {}! Required inputs are {}.".format(
                    self.name, self.x_names))

            input_values.append(input_vars[given_input])

        if isinstance(self.function, interp1d):
            return float(self.function(*[input_val for input_val in input_values]))

        elif isinstance(self.function, interp2d):
            return float(self.function(*[input_val for input_val in input_values]))

        else:
            raise Exception("Given inputs list has a wrong dimension for the computation of {}".format(self.name))


def instantiate_variables(var_dict: dict):
    """
    # TODO: cambiare configurazione dati in ingresso (esempio: LookupTable passata con un csv)
    """
    instantiated_vars = []

    for var in var_dict.keys():

        if var_dict[var]['type'] == "scalar":
            instantiated_vars.append(Scalar(name=var, value=var_dict[var]['scalar']))

        elif var_dict[var]['type'] == "function":
            instantiated_vars.append(Function())  # TODO: implement

        elif var_dict[var]['type'] == "lookup":
            instantiated_vars.append(
                LookupTableFunction(
                    name=var,
                    y_values=var_dict[var]['lookup']['output'],
                    x_names=var_dict[var]['lookup']['inputs'].keys(),
                    x_values=[var_dict[var]['lookup']['inputs'][key] for key in
                              var_dict[var]['lookup']['inputs'].keys()]
                ))

        else:
            raise Exception("The chosen 'type' for the variable '{}' is wrong or nonexistent! Try to select another"
                            " option among this list: ['scalar', 'function', 'lookup'].".format(var))

    return instantiated_vars
