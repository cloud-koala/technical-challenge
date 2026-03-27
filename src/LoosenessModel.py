from pydantic import BaseModel, Field
from typing import List


class Wave(BaseModel):

    time: List[float] = Field(..., description="Time points of the wave")
    signal: List[float] = Field(..., description="Signal values")


class LoosenessModel:

    def __init__(self, **params):
        # Store hyperparameters if needed
        self.params = params
        
    def predict(self, wave_hor: Wave, wave_ver: Wave, wave_axi: Wave) -> bool:
        """
        Predicts the presence of structural looseness based on horizontal,
        vertical, and axial wave data.

        Args:
            wave_hor (Wave): Horizontal wave data
            wave_ver (Wave): Vertical wave data
            wave_axi (Wave): Axial wave data

        Returns:
            bool: True if looseness is detected, False otherwise
        """
        raise NotImplementedError("The 'predict' method must be implemented.")

    def score(self, wave_hor: Wave, wave_ver: Wave, wave_axi: Wave) -> float:
        """
        Computes a confidence score (between 0 and 1) representing the
        likelihood of structural looseness. This is optional.
        
        Args:
            wave_hor (Wave): Horizontal wave data
            wave_ver (Wave): Vertical wave data
            wave_axi (Wave): Axial wave data
        
        Returns:
            float: score (0 = no looseness, 1 = high confidence of looseness)
        """
        raise NotImplementedError("The 'score' method must be implemented.")    

