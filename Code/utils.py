import json
import os.path as osp
import pickle
import pandas as pd

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

    out_path = osp.join(*args)

    with open(out_path, 'w') as fp:
        json.dump(data, fp, indent=indent, **kwargs)

    return
