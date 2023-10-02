from dataclasses import dataclass
import numpy as np
from scipy.optimize import dual_annealing
from typing import List, Protocol
import datetime
from .Data import IncomingData
import pandas as pd
import numpy as np

class Model(Protocol):
    """Protocol to represent a model --- must implement a predict function"""
    def predict(self, new_data: np.ndarray) -> np.ndarray:
        """Function to predict on new data"""


@dataclass(frozen=True)
class Dual_Annealing_Optimization:
    '''
    class to run dual_annealing optimization
    
    '''
    nomissing_data: IncomingData
    final_model: Model
    # denominator_list: list
    # time_stamps: list
    
    @staticmethod
    def objective(controls_vals:np.ndarray, noncontrols_vals:np.ndarray, final_model: Model, denominator) -> float:
        """
        Objective function for dual annealing optimization.

        Parameters
        ----------
        Ingest all the parameters from Method: run_optimization.
        
        controls_vals : np.ndarray | List[float]
            values of the controllable variables for the current row (controlled by the optmization function)
        noncontrols_vals : np.ndarray | List[float]
            values of the controllable variables for the current row
        
        Returns
        -------
        _ : float
            value of the objective funtion for this row
        """

        all_variables = np.array(np.concatenate([controls_vals, noncontrols_vals]).reshape(1, -1))

        # minimize the negative of the predicted value, since dual_annealing is a minimization algorithm
        # and we want to maximize the predicted value, which is the LNG production
        # return -final_model.predict(all_variables)[0]

        # minimize the target variable
        # return final_model.predict(all_variables)[0]

        pred = final_model.predict(all_variables)[0]
        kpi1 = pred / denominator

        if kpi1 < 0.84:
            error = (0.84 - kpi1)**2
        elif kpi1 > 0.9:
            error = (kpi1 - 0.9)**2
        else:
            error = 0

        # print(f'kpi1: {kpi1}, error: {error}')
        return error
        

    def run_optimization(self, timestamp: datetime.datetime, bound: List[float]) -> np.ndarray:
        '''
        method to run dual_annealing optimization

        Parameters
        bound: the min and max value of the controllable variables
        
        Return the optimized controllable value
        '''
        controls_vals  = self.nomissing_data.get_control_vals(timestamp).values.flatten()
        noncontrols_vals = self.nomissing_data.get_noncontrol_vals(timestamp).values.flatten()
        denominator = self.nomissing_data.calculate_denomitor(timestamp)
        # print(f'denominator: {denominator}')

        # get the index of timestamp in time_stamps
        # index = self.time_stamps.index(timestamp)
        # denominator = self.denominator_list[index]

        result = dual_annealing(self.objective, bound, args=(noncontrols_vals, self.final_model, denominator), x0=controls_vals, maxiter=20)
        
        optimal_all = np.array(np.concatenate([result.x, noncontrols_vals]).reshape(1, -1))
        optimized_product = self.final_model.predict(optimal_all)

        return result.x, optimized_product
