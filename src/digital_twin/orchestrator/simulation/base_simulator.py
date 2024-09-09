class BaseSimulator:
    """
    
    """
    MODES = ['driven', 'scheduled', 'adaptive']
    
    @classmethod
    def get_instance(cls, mode: str):
        """
        Get the instance of the subclass for the current experiment mode, checking if the mode name is
        contained inside the subclass name.
        NOTE: this works because of the __init__.py, otherwise the method __subclasses__() cannot find
              subclasses in other not yet loaded modules.
        """
        assert mode in cls.MODES, "The simulation has been stopped because the experiment mode does not exist!"
        return next(c for c in cls.__subclasses__() if mode in c.__name__.lower())
    
    def __init__(self, **kwargs) -> None:
        pass
    
    def step(self, **kwargs):
        raise NotImplementedError
    
    def stop(self):
        raise NotImplementedError
    
    def run(self):
        raise NotImplementedError
    
    def solve(self):
        raise NotImplementedError
    
    def store_sample(self):
        raise NotImplementedError
        
    def close(self):
        raise NotImplementedError
    