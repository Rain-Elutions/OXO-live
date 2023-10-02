import numpy as np
import pandas as pd
# Define custom errors
class LowBenzene(Exception):
    def __init__(self, benzene_val:float, benzene_cutoff:float) -> None:
        self.benzene_val = benzene_val
        self.benzene_cutoff = benzene_cutoff

    def __str__(self):
        return f"Low Benzene Warning: Threshold: {self.benzene_cutoff:.2f}, Value: {self.benzene_val:.2f}"

class HighBenzene(Exception):
    def __init__(self, benzene_val:float, cap: float) -> None:
        self.benzene_val = benzene_val
        self.benzene_cap = cap

    def __str__(self) -> str:
        return f"High Benzene Warning: Threshold: {self.benzene_cap}, Value: {self.benzene_val:.2f}"

class DecreasingProduct(Exception):
    def __init__(self, original_product:float, optimized_product:float) -> None:
        self.original_product = original_product
        self.optimized_product = optimized_product

    def __str__(self) -> str:
        return f"Decreasing Product Warning: Original ({self.original_product:.2f}) -> Optimized ({self.optimized_product:.2f})"

class IncreasingByproduct(Exception):
    def __init__(self, original_byproduct:float, optimized_byproduct:float) -> None:
        self.original_byproduct = original_byproduct
        self.optimized_byproduct = optimized_byproduct

    def __str__(self) -> str:
        return f"Increasing Byproduct Warning: Original ({self.original_byproduct:.2f}) -> Optimized ({self.optimized_byproduct:.2f})"

class NOConnectionError(Exception):
    """No Connection Error"""
    pass

# prerun checks
def benezene_input_check(benzene_val: float, floor: float = 2000, cap: float = 7500) -> bool:
    """
    Function to check whether the benzene input is above our threshold
    
    Parameters
    ----------
    benzene_val : float
        benzene value for new data
    cutoff : float, default = 2000
        benzene floor threshold
    cap : float, default = 7500
        benzene default cap
        
    Returns
    -------
    _ : bool
        wether the new data is valid (True) or the threhsold is exceeded (False)
    """
    if benzene_val < floor:
        raise LowBenzene(benzene_val, floor)
    elif cap < benzene_val:
        raise HighBenzene(benzene_val, cap)

    return True


# post run checks
def check_final_product(original_final_product: float, optimized_final_product: float) -> bool:
    """
    Function to ensure the optimization improved the final product
    
    Parameters
    ----------
    original_final_product : float
        original final product value
    optimized_final_product : float
        optimized final product value
        
    Returns
    -------
    _ : bool
        whether the final product improved (True) or not (False)
    """
    if original_final_product > optimized_final_product:
        raise DecreasingProduct(original_final_product, optimized_final_product)

    return original_final_product < optimized_final_product


def check_final_byproduct(original_byproduct: float, optimized_byproduct: float) -> bool:
    """
    Function to ensure the optimization decreased the byproduct
    
    Parameters
    ----------
    original_byproduct : float
        original byproduct value
    optimized_byproduct : float
        optimized byproduct value
        
    Returns
    -------
    _ : bool
        whether the byproduct decreased or not (True) or not (False)
    """
    if original_byproduct < optimized_byproduct:
        raise IncreasingByproduct(original_byproduct, optimized_byproduct)
        
    return optimized_byproduct < original_byproduct


def connection_check(new_data: pd.DataFrame) -> None:
    """
    Function to check if the connection is stable or not (all zeros or missing)

    Parameters
    ----------
    new_data : pd.DataFrame | pd.Series
        new data

    Returns
    """
    if (new_data == 0).all().all() or (new_data == 1).all().all() or new_data.empty:
        raise NOConnectionError("No Connection Error: All the data is missing or zero")

    # TODO: how do I check the edge case where some are zero and the other's are missing ... this needs to be done ...

    return
