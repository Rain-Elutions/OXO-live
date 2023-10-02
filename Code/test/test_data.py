from Data import IncomingData, Missing_info
from Features import Feature_Info, get_feature_full_name
import pytest
import pandas as pd
import datetime
from utils import read_json

@pytest.fixture
def raw_data():
    raw_data_path = '../Model/data_unit_test.csv'
    raw_data = pd.read_csv(raw_data_path, parse_dates=['Date'], index_col='Date')
    return raw_data

@pytest.fixture
def previous_data():
    previous_data_path = '../Model/previous_tag_vals.csv'
    previous_data = pd.read_csv(previous_data_path, parse_dates=['Date'], index_col='Date')
    return previous_data

@pytest.fixture
def features():
    optimizer_info_path = '../Model/p2_optimizer.json'
    optimizer_info = read_json(optimizer_info_path)
    return Feature_Info(optimizer_info)


def test_slicing(raw_data: pd.DataFrame, features: Feature_Info) -> None:
    '''
    test if slicing works well
    '''
    piece_of_sliced_data = IncomingData(raw_data).slicing(raw_data.index[0], get_feature_full_name(features.controllable.keys()))
    piece_of_raw_date = raw_data[get_feature_full_name(features.controllable.keys())]
    assert piece_of_raw_date.equals(piece_of_sliced_data)

def test_missing_count(raw_data, previous_data, features) -> None:
    '''
    test if missing count method works well
    '''
    missing_info = Missing_info(IncomingData(raw_data), features)

    assert missing_info.missing_count(raw_data.index[0]).sum() == 1

def test_missing_log() ->None:
    '''
    test if missing log method works well
    '''
    pass

def test_missing_filling(raw_data, previous_data, features) -> None:
    '''
    test if missing filling works well
    '''
    missing_info = Missing_info(IncomingData(raw_data), features)
    no_missing_data = missing_info.missing_filling(previous_data)

    correct_filled_data = raw_data.copy()
    correct_filled_data['T8.PACOL2.6AIC-401A Augusta___Value'] = 10
    correct_filled_data['V406-TOTALI N-OLEFINE Augusta___Value'] = 8.914886
    correct_filled_data['INGRESSO R473-AROMATICI UOP Augusta___Value'] = 0.105783
    correct_filled_data['K401-HYDROGEN Augusta___Value'] = 88.371111
    correct_filled_data['INGRESSO R490-DIOLEFINE Augusta___Value'] = 0.6748

    assert correct_filled_data[no_missing_data.value.columns].equals(no_missing_data.value)


