class BaseModel:
    """
    Interface for model used in API
    """
    def __init__(self, logger, **kwargs):
        self.logger = logger

    def predict(self):
        raise NotImplementedError
