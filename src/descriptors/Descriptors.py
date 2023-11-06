from core.CoreImage import CoreImage, Paint

from abc import ABC, abstractmethod
from typing import *

import numpy as np

class FeatureExtractors(ABC):

    @abstractmethod
    def extract(self, images: List[CoreImage], **kwargs) -> Dict[str, np.ndarray]:
        pass