from Features import get_feature_full_name, Feature_Info
from utils import read_json, read_pickle
import pandas as pd
import numpy as np
import pytest

@pytest.fixture
def features():
    optimizer_info_path = '../Model/p2_optimizer.json'
    optimizer_info = read_json(optimizer_info_path)
    return Feature_Info(optimizer_info)

@pytest.fixture
def final_model():
    final_model_path = '../Model/xgb_pacol2_sasol_main_product_30mins.pkl'
    final_model = read_pickle(final_model_path)
    return final_model

# @pytest.fixture
# def feature_short_to_long_map():
#     feature_short_to_long_map_path = '../Model/short_to_long_name_map.json'
#     feature_short_to_long_map = read_json(feature_short_to_long_map_path)
#     return feature_short_to_long_map

def test_get_feature_full_name(features) -> None:
    to_convert = features.noncontorllable
    assert get_feature_full_name(to_convert) == [tag+'___Value' for tag in to_convert]


def test_optimizer_info_match_model(features, final_model) -> None:
    '''
    test if the number of controllable & noncontrollable matches the shape in the final model.
    '''
    control_noncontrol_features = list(features.controllable.keys()) \
                                  + features.noncontorllable
    # High coupling here! Bad!
    short_control_noncontrol_features = [long_feature.replace('T8.PACOL2.','').replace(' Augusta','') for long_feature in control_noncontrol_features]
    model_features = final_model.get_booster().feature_names
    assert short_control_noncontrol_features == model_features
    

def test_pull_features(features) -> None:
    '''
    test if the UsedDatapointNames.csv matches the features used in optimizer.info
    '''
    pull_feature = pd.read_csv('../p2_UsedDatapointNames.csv', header=None).values.flatten()
    optimizer_tags = list(features.controllable.keys()) \
                    + features.noncontorllable \
                    + features.additional \
                    + features.Lims
    return np.testing.assert_array_equal(pull_feature,optimizer_tags)


    