class BaseAdapter:
    """
    Base class for all adaptive routines.
    For example, there could be online learning routine based on active learning,
    or regime shift detection routine based on clustering and Mahalanobis distance.
    """
    def __init__(self, **kwargs):
        pass
    
    def reset(self, **kwargs):
        """
        Reset the adapter to its initial state.
        """
        pass
    
    def get_estimated_params(self):
        """
        Load the estimated parameters from the file.
        """
        return None
    
    def keep_monitoring(self):
        """
        Return True if the adapter is keeping monitoring the parameters.
        """
        return None
    
    def step(self):
        """
        Perform a step of the adaptation.
        """
        pass