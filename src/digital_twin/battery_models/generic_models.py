from abc import ABC, abstractmethod


class AbstractEquivalentCircuitModel(ABC):

    @abstractmethod
    def get_soc(self):
        pass


class AbstractThermalModel(ABC):

    @abstractmethod
    def get_temp(self):
        pass


class AbstractDegradationModel(ABC):

    @abstractmethod
    def get_soh(self):
        pass
