import numpy as np
import pandas as pd
import logging

from functools import partial

# TODO: customize error messages!
class OperationalKPIError(Exception):
    def __init__(self, msg:str, value:float, floor:float, cap:float) -> None:
        self.msg = msg
        self.value = value
        self.floor = floor
        self.cap = cap

    def __str__(self):
        if self.value < self.cap:
            return f"Low KPI Error ({self.msg}) floor: {self.floor}, Value: {self.value:.2f}"
        else: 
            return f"High KPI Error ({self.msg}) Cap: {self.cap}, Value: {self.value:.2f}"


# FIXME: Regenerated HF to LAB Mass Ratio
def check_operational_kpi(operation_kpi_val: float, message: str, floor: float = 0.4, cap: float = 0.6) -> bool:
    """
    Function to check whether the operation_kpi2 input is above our threshold
    
    Parameters
    ----------
    operation_kpi_val : float
        operation KPI value for new data
    cutoff : float, default = 2000
        operation KPI floor threshold
    cap : float, default = 7500
        operation KPI default cap
        
    Returns
    -------
    _ : bool
        wether the new data is valid (True) or the threhsold is exceeded (False)
    """
    if operation_kpi_val < floor or cap < operation_kpi_val:
        raise OperationalKPIError(message, operation_kpi_val, floor, cap)

    return True

# NOTE: should we define the bounds here?
# define the operational kpis that will be used
# NOTE: swithcing floor from .4 to .25 (11:15am CT 07/29/2022)
define_reactor_hydrogen_diolefin_check = partial(check_operational_kpi, message="DeFine Reactor Hydrogen to Diolefin Molar Ratio", floor=.25, cap=1)
# ICC H2/Aromatic KPI
# TODO: fix bounds
icc_htwo_aromatic_check = partial(check_operational_kpi, message="ICC H2/Aromatic", floor=.25, cap=1)
# Reactor Hydrogen/HC KPI
# TODO: fix bounds
hydrogen_hc_check = partial(check_operational_kpi, message="Hydrogen HC KPI", floor=.25, cap=1)

class OptimizationLowDifferenceError(Exception):
    def __init__(self, percent_different, tag) -> None:
        self.percent_different = percent_different
        self.tag = tag

    def __str__(self):
        return f"Optimization values for {self.tag} within 3% of original values. Optimization {self.percent_different:.2f}% different."

def calculate_percent_change(optimized_vals, original_vals):
    """
    Function to calculate percentage of change between original vals and optimized vals
    Parameters
    ----------
    optimized_vals : List
        List of optimized values
    original_vals : List
        List of original values
    Returns
    -------
    percent_different : 
        Percent change between optimal and original
    """
    percent_different = abs(optimized_vals - original_vals)/original_vals

    return percent_different
    
def check_optimization_difference(percent_different, tag_labels, threshold = 0.0000001) -> bool:
    """
    Function to check how different optimization is than original values
    
    Parameters
    ----------
    percent_different : List
        List of differences between original and optimized values
    tag_labels : List
        labels of cotnrollable vars
    threshold : float
        Percentage cutoff for difference between original and optimal values
    """
    logger = logging.getLogger(__name__)

    for tag in range(percent_different.size):
        try:
            if percent_different[tag] < threshold:
                raise OptimizationLowDifferenceError(percent_different[tag] * 100, tag_labels[tag])
        except OptimizationLowDifferenceError as e:
            logger.warning(str(e))

    return True


def track_different_tags(percent_different, tag_list, threshold_list):
    """
    Function to return differing tags, difference being defined by threshold
    Parameters
    ----------
    percent_different : List
        List of differences between original and optimized values
    threshold : List
        List of percentage cutoff for difference between original and optimal values  Given by IES
    Returns
    -------
    different_tags : 
        Different tags, similar being defined by the threshold 
    """
    different_tags = []
    for tag in range(percent_different.size):
        if percent_different[tag] > threshold_list[tag]:
            different_tags.append(tag_list[tag])

    return different_tags