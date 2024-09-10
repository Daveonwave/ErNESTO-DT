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
    
    def init(self):
        self._init()
        
    def _init(self):
        raise NotImplementedError
    
    def step(self, **kwargs):
        self._step(kwargs)
    
    def _step(self, **kwargs):
        raise NotImplementedError
    
    def stop(self):
        self._stop()
    
    def _stop(self):
        raise NotImplementedError
    
    def run(self):
        self._run()
        
    def _run(self):
        raise NotImplementedError
    
    def solve(self):
        self._solve()
    
    def _solve(self):
        raise NotImplementedError
    
    def store_sample(self):
        self._store_sample()
    
    def _store_sample(self):
        raise NotImplementedError
        
    def close(self):
        self._close()
    
    def _close(self):
        raise NotImplementedError
    