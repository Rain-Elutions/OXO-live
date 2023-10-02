from functools import wraps
import logging
import numpy as np
import pandas as pd
from scipy.optimize import dual_annealing
from typing import List, Protocol

from utils import read_json


class Model(Protocol):
    """Protocol to represent a model --- must implement a predict function"""
    def predict(self, new_data: np.ndarray) -> np.ndarray:
        """Function to predict on new data"""


def objective(controls:np.ndarray, noncontrollable:np.ndarray, final_model:Model) -> float:
    """
    Objective function for the hf process

    Parameters
    ----------
    controls : np.ndarray | List[float]
        values of the controllable variables for the current row (controlled by the optmization function)
    noncontrollable : np.ndarray | List[float]
        values of the controllable variables for the current row

    Returns
    -------
    _ : float
        value of the objective funtion for this row
    """
    all_variables = np.array(np.concatenate([controls, noncontrollable]).reshape(1, -1))

    #return -(final_model.predict(all_variables)[0] - product_model.predict(all_variables)[0])
    return -(final_model.predict(all_variables)[0])

# @add_validation
#def run_model(controllable: np.ndarray, noncontrollable: np.ndarray, bounds: List[List[float]], final_model:Model, byproduct_model:Model, **optimizer_kwargs):
def run_model(controllable: np.ndarray, noncontrollable: np.ndarray, bounds: List[List[float]], final_model:Model, **optimizer_kwargs):
    """
    Function to run the model and the optimization procedure on the new data (assumes new data is a single row)

    Parameters
    ----------
    controllable : np.ndarray | List[float]
        values of the control variables
    noncontrollable : np.ndarray | List[float]
        values of the noncontrollable variables
    bounds : List[List[float]]
        list of bounds for each variable (lower, upper)
    final_model : Model
        final model to predict the kpi denominator
    byproduct_model : Model
        byproduct model to predict the amount of byproduct
    optimizer_kwargs : Dict[str, ?]
        additional keyword arguments to dual annealing

    Returns
    -------
    result.x : List[float]
        optimal values for the control variables
    result.success : bool
        whether the optimzation procedure succeeded or not
    """
    #result = dual_annealing(objective, bounds, args=(noncontrollable, final_model, byproduct_model), x0=controllable, **optimizer_kwargs)
    result = dual_annealing(objective, bounds, args=(noncontrollable, final_model), x0=controllable, **optimizer_kwargs)

    return result.x, result.success

# TODO: to create a validation procedure for applying the model we need:
# 1. To check if the new data is within the historical data (or close)
# 2. confirm the model converges 
def check_bounds(new_data: pd.Series, bounds: np.ndarray, tol:float = .1) -> bool:
    """
    Function to check the whether the controllable variables are near the bounds of what it was developed on historically

    Parameters
    ----------
    new_data : np.ndarray | List[float]
        values of the new data
    historical_bounds : np.ndarray | List[List[float]]
        historical bounds for each variable
    tol : float, default = .1
        tolerance --- if new data is outside bounds but only by x% then still apply the model
    
    Returns
    -------
    safe : bool
        whether it is safe to apply the model or not
    """
    upper_factor = tol + 1
    lower_factor = 1 - tol

    # TODO: check that the indexing of the numpy array is right
    lower_bounds = bounds[0, :] * lower_factor
    upper_bounds = bounds[1, :] * upper_factor

    each_var = np.logical_and(new_data > lower_bounds, new_data < upper_bounds)

    return each_var.all()

def add_validation(func):
    """Function to add validation checks to the run_model function"""
    @wraps(func)
    def wrapper(new_row:pd.DataFrame, control_vars, noncontrol_vars, **kwargs):
        """Wrapper function"""
        historical_bounds: np.ndarray
        logger = logging.getLogger(__name__)
        
        timestamp = new_row.index[0]

        safe = check_bounds(new_row, historical_bounds)
        if not safe:
            logger.warning(f"New Data at timestamp: {timestamp} is not close enough to historical data --- skipping")
            return new_row

        controls = new_row.loc[:, control_vars].copy()
        noncontrols = new_row.loc[:, noncontrol_vars].copy()

        optimal_controls, converged =  run_model(controllable=controls, noncontrollable=noncontrols, **kwargs)

        if not converged: 
            logger.warning(f"New Data at timestamp: {timestamp} Failed to converge --- returning original data")
            return controls

        return optimal_controls
    return wrapper


