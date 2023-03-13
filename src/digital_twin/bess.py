from src.digital_twin.battery_models.electrical_model import TheveninModel


class BatteryEnergyStorageSystem:
    """
    Class representing the battery abstraction.
    Here we select all the electrical, thermal and mathematical models to simulate the BESS behaviour.
    #TODO: can be done with multi-threading (one for each submodel)?
    """
    def __init__(self):

        self.electrical_model = None
        self.thermal_model = None
        self.degradation_model = None

        self.sampling_time = 5
        self.duration = 3600

        self.v_max = 10
        self.v_min = 0
        self.initial_soc = 0

        self.electrical_model = TheveninModel()

    def run(self):
        """

        """
        k = 0
        v_computed = []

        for time in range(0, self.duration, self.sampling_time):
            voltage = self.electrical_model.solve_components_cc_mode(dt=self.sampling_time, i_load=0.2, k=k)
            k =+ 1
            v_computed.append(voltage)

        return v_computed



if __name__ == '__main__':
    battery = BatteryEnergyStorageSystem()
    v = battery.run()
    print(v)