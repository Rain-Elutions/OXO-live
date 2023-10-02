from dataclasses import dataclass
from .Features import Feature_Info, get_feature_full_name
from .reporter import MissingDataChecker
import pandas as pd
import datetime
import logging


@dataclass(frozen=True)
class IncomingData:
    '''
    class to include IncomingData information
    '''
    
    value: pd.DataFrame
    features: Feature_Info
    
    def slicing(self, timestamp: datetime.datetime, columns: list) -> pd.DataFrame:
        '''
        get the all the value at a single timestamp.
        '''
        return self.value.loc[[timestamp], columns]
    
    def get_control_vals(self, timestamp: datetime.datetime) -> pd.DataFrame:
        '''
        get the controllable values at a single timestamp
        '''
        return self.slicing(timestamp, get_feature_full_name(self.features.controllable.keys()))
    
    def get_noncontrol_vals(self, timestamp: datetime.datetime) -> pd.DataFrame:
        '''
        get the noncontrollable values at a single timestamp
        '''
        return self.slicing(timestamp, get_feature_full_name(self.features.noncontorllable))    

    def feature_engineering(self, ) -> pd.DataFrame:
        '''
        interface to do feature engineering in the specific process
        '''
        pass

    def calculate_denomitor(self, timestamp: datetime.datetime) -> pd.DataFrame:
        '''
        get the denominator value in the kpi1 formula at a single timestamp
        '''
        tag2 = self.slicing(timestamp, get_feature_full_name(self.features.additional['tag2'])).values[0][0]
        tag3 = self.slicing(timestamp, get_feature_full_name(self.features.additional['tag3'])).values[0][0]
        tag4 = self.slicing(timestamp, get_feature_full_name(self.features.additional['tag4'])).values[0][0]
        tag5 = self.slicing(timestamp, get_feature_full_name(self.features.additional['tag5'])).values[0][0]

        return (tag2 + tag3)*1000 + tag4 + tag5


@dataclass(frozen=True)
class Missing_info:
    '''
    class to include Missingvalue operations
    '''
    incoming_data: IncomingData
    
    def missing_count(self, timestamp: datetime.datetime) -> pd.Series:
        '''
        method to record missing count of controllable & noncontrollable for a single timestamp.
        
        Return a pandas Series including the missing feature. 
        '''
        controllable_noncontrollable = list(self.incoming_data.features.controllable.keys()) \
                                       + self.incoming_data.features.noncontorllable
        controllable_noncontrollable = get_feature_full_name(controllable_noncontrollable)
        
        # Count Missing
        missing_place = self.incoming_data.slicing(timestamp, controllable_noncontrollable).isnull().sum()
        missing_place = missing_place[missing_place>0]

        return missing_place
    
    def missing_log(self, timestamp: datetime.datetime, missing_tag_checker: MissingDataChecker) -> None:
        '''
        method to log the missing features into the logger and set the missing checker value.
        '''
        logger = logging.getLogger(__name__)

        missing_place = self.missing_count(timestamp)
        
        missing_features = [feature for feature in missing_place.index]
        for missing_feature in missing_features:
            logger.warning(missing_feature.replace('___Value', ''))
        
        # Record the total missing count for the time-stamp
        if total_missing := missing_place.sum() > 0:
            missing_tag_checker.add_value(timestamp.timestamp(), total_missing)
    
    def missing_filling(self, previous_data: pd.DataFrame) -> IncomingData:
        '''
        method to fill the missing value for a single timestamp.
        
        Change from use the mean of previous 4 value to ffill, which I think makes more sense.
        '''
        nomissing_value = pd.concat([previous_data, self.incoming_data.value]).ffill().iloc[-self.incoming_data.value.shape[0]:]
        return IncomingData(nomissing_value, self.incoming_data.features)