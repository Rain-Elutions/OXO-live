import os.path as osp
import json
import pickle
import pandas as pd
import numpy as np
import yaml

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            # 👇️ alternatively use str()
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def write_json(data: dict, *args, indent:int = 4, **kwargs) -> None:
    """
    Function to write a dictionary out to a json file --- according to the json standards, the dictionary keys must be strings

    Parameters 
    ----------
    data : dict
        dictionary to be written to a json file
    args : str
        path for the file to be written to
    kwargs : dict
        additional keyword arguments to json.dump
    """
    #print(args)
    out_path = osp.join(args[0])
    #out_path = args
    with open(out_path, 'w') as fp:
        json.dump(data, fp, cls=NpEncoder, indent=indent, **kwargs)
    return

def read_json(*args):
    """
    Function to load a json file as a dictionary

    Parameters
    ----------
    args: List[str]
        input path to the json file to be read, separate arguments will be combined in to a single path: ie foo, bar, test.json -> foo/bar/test.json    

    Returns
    -------
    data: Dict[str, ?]
        json file as a dictionary
    """
    path = osp.join(*args)
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def read_yaml(*args):
    """
    Function to load a json file as a dictionary

    Parameters
    ----------
    args: List[str]
        input path to the json file to be read, separate arguments will be combined in to a single path: ie foo, bar, test.json -> foo/bar/test.json    

    Returns
    -------
    data: Dict[str, ?]
        yaml file as a dictionary
    """
    path = osp.join(*args)
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    return data

    
def read_pickle(*args:str):
    """
    Function to read pickle files

    Parameters
    ----------
    args: List[str] | str
        input path to the json file to be read, separate arguments will be combined in to a single path: ie foo, bar, test.json -> foo/bar/test.pkl   

    Returns
    -------
    data : ?
        contents of the pickle file    
    """
    file_path = osp.join(*args)

    with open(file_path, 'rb') as fp:
        data = pickle.load(fp)

    return data

def reduce_mem_usage(df: pd.DataFrame, verbose=True) -> pd.DataFrame:
    """
    Function to trim the data types to reduce memory consumption

    Parameters
    ----------
    df : pd.DataFrame
        data of interest
    verbose : bool, default=True
        whether to print extra info out

    Returns
    -------
    df : pd.DataFrame
        reduced data frame
    """
    # NOTE: warning: this function applies all the operations directly to df so the input df will be changed --- make sure this is intended
    # ie: df2 = reduce_mem_usage(df1, True) will result in df2 __and__ df1 being changed. 
    # if this is not the intended behavior then make a copy df_copy = df.copy() to remove the reference issue and then modify df_copy

    numerics = ['int8','int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtypes

        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
 
    return df