from typing import Protocol
import datetime
import numpy as np
from reporter import AdditionalInfoChecker
from Dual_Annealing_Optimization import Dual_Annealing_Optimization

class Add_checker_value(Protocol):
    '''
    Protocal to add values to checker
    '''
    def get_checker_value(self,):
        '''
        function to get the checker value
        '''
        ...
        
    def add_value(self,):
        '''
        function to add the value the checker
        '''
        ...

class Np_KPI_checker:
    
    def __init__(self, data, timestamp: datetime.datetime, checker: AdditionalInfoChecker):
        self.data = data
        self.timestamp = timestamp
        self.checker = checker
    
    def get_checker_value(self,):
        '''
        method to calculate np consumption
        '''
        np_consumption = 100 * self.data.at[self.timestamp, 'T8.PACOL2.6FIC-403 Augusta___Value'] / (self.data.at[self.timestamp, 'T8.PACOL2.6FI-474 Augusta___Value']*self.data.at[self.timestamp, 'V406-TOTALI N-OLEFINE Augusta___Value'])
        return np_consumption
    
    def add_value(self, ):
        '''
        method to add ng value to the ng_checker
        '''
        self.checker.add_value(self.timestamp.timestamp(), self.get_checker_value())

class Ng_KPI_checker:
    
    def __init__(self, data, timestamp: datetime.datetime, checker: AdditionalInfoChecker):
        self.data = data
        self.timestamp = timestamp
        self.checker = checker
    
    def get_checker_value(self,):
        '''
        method to calculate np consumption
        '''
        checker_value = self.data.at[self.timestamp, 'T8.PACOL2.6FI-120_51 Augusta___Value'] / self.data.at[self.timestamp, 'T8.PACOL2.6FI-440 Augusta___Value']
        return checker_value
    
    def add_value(self, ):
        '''
        method to add ng value to the ng_checker
        '''
        self.checker.add_value(self.timestamp.timestamp(), self.get_checker_value())

class Original_product_checker:
    
    def __init__(self, data, timestamp: datetime.datetime, checker: AdditionalInfoChecker, optimization: Dual_Annealing_Optimization):
        self.data = data
        self.timestamp = timestamp
        self.checker = checker
        self.optimization = optimization
    
    def get_checker_value(self,):
        '''
        method to calculate np consumption
        '''
        controls_vals = self.optimization.nomissing_data.get_control_vals(self.timestamp).values.flatten()
        noncontrols_vals = self.optimization.nomissing_data.get_noncontrol_vals(self.timestamp).values.flatten()
        all_variables = np.array(np.concatenate([controls_vals, noncontrols_vals]).reshape(1, -1))
        checker_value = self.optimization.final_model.predict(all_variables)[0]
        return checker_value
    
    def add_value(self, ):
        '''
        method to add ng value to the ng_checker
        '''
        self.checker.add_value(self.timestamp.timestamp(), self.get_checker_value())    
        
class Optimized_product_checker:
    
    def __init__(self, data, timestamp: datetime.datetime, checker: AdditionalInfoChecker, optimization: Dual_Annealing_Optimization, optimal_controls_vals: np.ndarray):
        self.data = data
        self.timestamp = timestamp
        self.checker = checker
        self.optimization = optimization
        self.optimal_controls_vals = optimal_controls_vals
    
    def get_checker_value(self,):
        '''
        method to calculate np consumption
        '''
        noncontrols_vals = self.optimization.nomissing_data.get_noncontrol_vals(self.timestamp).values.flatten()
        all_variables = np.array(np.concatenate([self.optimal_controls_vals, noncontrols_vals]).reshape(1, -1))
        checker_value = self.optimization.final_model.predict(all_variables)[0]
        return checker_value
    
    def add_value(self, ):
        '''
        method to add ng value to the ng_checker
        '''
        self.checker.add_value(self.timestamp.timestamp(), self.get_checker_value())    