import numpy as np
from typing import List
import logging
from dataclasses import dataclass
from Features import Feature_Info

logger = logging.getLogger(__name__)


class OptimizationLowDifferenceError(Exception):
    '''
    class to return low difference error.
    
    Parameters
    ----------
    percent_different: float, % change from the original value
    min_changing_rate: float, % minimum change required by IES
    tag: corresponding tag name
    
    '''
    def __init__(self, percent_difference, min_changing_rate, feature_name) -> None:
        self.percent_difference = percent_difference
        self.min_changing_rate = min_changing_rate
        self.feature_name = feature_name

    def __str__(self):
        return f"Optimization values for {self.feature_name} within {self.min_changing_rate:.2f}% of original values. Optimization {self.percent_difference:.2f}% different."


@dataclass(frozen=True)
class Postprocess:
    optimal_controls_vals: np.ndarray
    original_controls_vals: np.ndarray
    features: Feature_Info
    

    def calculate_percentage_changes(self, ) -> np.ndarray:
        """
        method to calculate percentage of change between original vals and optimized vals

        Returns
        -------
        percent_different : 
            Percent change between optimal and original
        """
        percent_difference = np.abs((self.optimal_controls_vals-self.original_controls_vals)/self.original_controls_vals+0.001)

        return percent_difference
    
    def track_different_controls(self, ) -> List[str]:
        """
        method to return differing controllable tags, difference being defined by threshold

        Record the small difference features into the logging

        Returns
        -------
        different_tags : 
            Different tags, similar being defined by the threshold 
        """
        different_tags = []
        min_rate_change_threshold = np.array([item['min_rate'] for item in self.features.controllable.values()])
        percentage_differences = self.calculate_percentage_changes()
        
        for controllable_index in range(percentage_differences.shape[0]):
            try:
                if percentage_differences[controllable_index] > min_rate_change_threshold[controllable_index]:
                    different_tags.append(list(self.features.controllable.keys())[controllable_index])
                else:
                    raise OptimizationLowDifferenceError(min_rate_change_threshold[controllable_index] * 100, list(self.features.controllable.keys())[controllable_index])
            except OptimizationLowDifferenceError as e:
                logger.warning(str(e))

        return different_tags
