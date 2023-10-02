from typing import List
from dataclasses import dataclass,field

def get_feature_full_name(to_convert: List[str]) -> List[str]:
    '''
    add suffix '___Value' to the short name to get the full name of features
    '''
    
    return [short_name+'___Value' for short_name in to_convert]

@dataclass(frozen=True)
class Feature_Info:
    '''
    class to include all controllable, noncontrollable, additional & Lims feature information.

    Parameters
    ----------
    optimizer: dict
        read from optimizer.info, must have controllable & noncontrollable; additional & Lims are optional.
        controllable is a dict of dicts, including min & max changing rates, bounds.
        nonctrollable is a list of features(strings).
    '''
    optimizer: dict
    controllable: dict = field(init=False)
    noncontorllable: list = field(init=False)
    additional: list = field(init=False)
    Lims: list = field(init=False)
    
    def __post_init__(self) -> None:
        '''
        method to generate all the sub features
        '''
        object.__setattr__(self, 'controllable', self.optimizer['controllable'])
        object.__setattr__(self, 'noncontorllable', self.optimizer['noncontrollable'])
        object.__setattr__(self, 'additional', self.optimizer.get('additional', []))
        object.__setattr__(self, 'Lims', self.optimizer.get('Lims', []))
