import numpy as np

class Spectrum(object):
    """ spectrum class
        save specturm as a np array as [[peak,intentisy],[peak,intentisy],[peak,intentisy],[peak,intentisy]]
    """
    def __init__(self) -> None:
        self._spectrum = np.array()
        self.__annotations = {}