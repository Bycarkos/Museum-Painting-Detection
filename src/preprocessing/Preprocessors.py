import abc
from abc import  ABC

class Preprocessors(ABC):

    @abc.abstractmethod
    def __init__(self):
        raise NotImplementedError

