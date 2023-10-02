from Data import IncomingData
from dataclasses import dataclass
import numpy as np
from scipy.optimize import dual_annealing
from typing import List, Protocol
import datetime

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
    
    @staticmethod
    def objective(controls_vals:np.ndarray, noncontrols_vals:np.ndarray, final_model: Model) -> float:
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

        return -(final_model.predict(all_variables)[0])

    def run_optimization(self, timestamp: datetime.datetime, bound: List[float]) -> np.ndarray:
        '''
        method to run dual_annealing optimization
        
        Return the optimized controllable value & success or not
        '''
        controls_vals  = self.nomissing_data.get_control_vals(timestamp).values.flatten()
        noncontrols_vals = self.nomissing_data.get_noncontrol_vals(timestamp).values.flatten()
        result = dual_annealing(self.objective, bound, args=(noncontrols_vals, self.final_model), x0=controls_vals, maxiter=50)
        optimal_controls_vals = result.x
        return optimal_controls_vals


