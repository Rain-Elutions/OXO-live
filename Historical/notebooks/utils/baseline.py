# local imports
import pandas as pd

# XGboost
import xgboost as xgb
from xgboost import cv
from xgboost import XGBRegressor

def get_controllable_ind_tag(tag_df: pd.DataFrame, selected_tags: list) -> list:
    '''
    Function to get controllable and indicator tags

    Parameters
    ----------
    tag_df_path: entire tag dataframe
    selected_tags: selected tag list from feature selection 
    
    Returns
    -------
    ctrl_tag : list of selected controllable tags
    ind_tag :  list of selected controllable tags
    '''


    Controllable_tags = tag_df[tag_df['Class']=='C'].Tag.values.flatten().tolist()
    ctrl_tag = [x for x in selected_tags if x in Controllable_tags]
    ind_tag = [x for x in selected_tags if x not in ctrl_tag]
    return ctrl_tag, ind_tag

def model_building() -> XGBRegressor:
    '''
    Function to build xgb regression model

    Parameters
    ----------
    ctrl_tag: list of selected controllable tags
    ind_tag:  list of selected controllable tags
    training_X: dataframe for generating bounds

    Returns
    -------
    XGB model
    '''

    my_model = XGBRegressor(colsample_bytree=0.2,
                            gamma=0.0,
                            learning_rate=0.01,
                            max_depth=4,
                            min_child_weight=1.5,
                            n_estimators=7200,                                                                  
                            reg_alpha=0.9,
                            reg_lambda=0.6,
                            subsample=0.2,
                            random_state=123)
    return my_model

