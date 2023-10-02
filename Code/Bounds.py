from Data import IncomingData
from Features import get_feature_full_name
from dataclasses import dataclass, field
import pandas as pd
from typing import List, Tuple

@dataclass(frozen=True)
class Bounds:
    '''
    class to create correct bounds for incoming live data.
    Should consider both max changing rates(dynamic bounds) and Historical bounds(global bounds). 
    '''
    incoming_data: IncomingData
    controllable_info: dict = field(init=False)
    
    def __post_init__(self, ) -> None:
        '''
        get controllable information in AllFeature
        '''
        object.__setattr__(self, 'controllable_info', self.incoming_data.features.controllable)
    
    
    def current_controllable_val(self, ) -> pd.DataFrame:
        '''
        method the get all the current controllable values.
        '''
        current_controllable_value = self.incoming_data.value.loc[:,get_feature_full_name(self.incoming_data.features.controllable.keys())]
        
        return current_controllable_value
    
    def dynamic_bounds(self, ) -> pd.DataFrame:
        '''
        method to calculate max bounds based on current controllable values. 
        '''
        current_controllable_value = self.current_controllable_val()
        max_rate_change = [item['rate'] for item in self.controllable_info.values()]
        
        min_dyn_bounds = current_controllable_value - (current_controllable_value*max_rate_change)
        max_dyn_bounds = current_controllable_value + (current_controllable_value*max_rate_change)
        
        return min_dyn_bounds, max_dyn_bounds
    
    def global_bounds(self, ) -> List:
        '''
        method to get the Historical bounds(global bounds).
        '''
        min_glo_bounds = [item['bounds'][0] for item in self.controllable_info.values()]
        max_glo_bounds = [item['bounds'][1] for item in self.controllable_info.values()]
        
        return min_glo_bounds, max_glo_bounds
    
    def final_bounds(self, ) -> List[List[Tuple[float]]]:
        '''
        method to create the final bounds for controllables.
        '''
        min_dyn_bounds, max_dyn_bounds = self.dynamic_bounds()
        min_glo_bounds, max_glo_bounds = self.global_bounds()
        
        controllable_full_name = get_feature_full_name(self.controllable_info.keys())
        
        for i, cont in enumerate(controllable_full_name):
            min_dyn_bounds.loc[min_dyn_bounds[cont] < min_glo_bounds[i], cont] = min_glo_bounds[i]
            min_dyn_bounds.loc[min_dyn_bounds[cont] > max_glo_bounds[i], cont] = max_glo_bounds[i] - 0.0001
            max_dyn_bounds.loc[max_dyn_bounds[cont] < min_glo_bounds[i], cont] = min_glo_bounds[i] + 0.0001
            max_dyn_bounds.loc[max_dyn_bounds[cont] > max_glo_bounds[i], cont] = max_glo_bounds[i]
        
        bounds = [
        [(min_dyn_bounds.at[timestamp, col+'___Value'], max_dyn_bounds.at[timestamp, col+'___Value']) for col in self.controllable_info.keys()] 
        for timestamp in self.incoming_data.value.index.tolist()
        ]
        
        return bounds
    
    def final_bound(self, index) -> List[Tuple[float]]:
        '''
        get the controllable bound at a single timestamp.
        
        Parameters:
        ----------
        index: int
            index value during iteration
            May fix this, because only this thing use index, instead of timestamp. Really high coupling!!
        '''
        bound = self.final_bounds()[index]
        
        return bound