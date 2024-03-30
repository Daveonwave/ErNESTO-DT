import pybamm

class Battery:
    def __init__(self):
        pass



class ThermalSubmodel(pybamm.BaseSubModel):
    def __init__(self, param, domain, options=None):

        super().__init__(param, domain, options=options)

    def get_fundamental_variables(self):
        T_cell = pybamm.Variable("Cell temperature [degC]", domain="positive")
        # heat = pybamm.Variable("Heat", domain="positive")


        return {
            "Cell temperature [degC]": T_cell
        }

    def get_coupled_variables(self, variables):
        pass

    def set_rhs(self, variables):
        T_cell = variables["Cell temperature [degC]"]



        res = ...

        self.rhs = {T_cell:""}

    def set_boundary_conditions(self, variables):
        pass

    def set_initial_conditions(self, variables):
        pass