"""
This module is to give some utility functions to help with converting data that was downloaded historically to the live deployment format. This process consists of the
following steps:

This procedure assumes the starting data is the historical data where the columns are the object/datapoint names for each tag (index is the date) that match the mapper file
1. Get the object/datapoint ids from the object/datapoint names (the columns)
2. Identify the needed property names/ids from for live deployment (this must be done manually ...)
3. Use the MapperHandler to create mapping from names to ids
    3.1: create an instance of a MapperHandler (mh)
    3.2.a: Subset the mapper using mh.filter_mapper using the mapper, object_ids, and property_ids
    3.3.a: Create the mappings (name_to_id and id_to_name) based on the subsetted mapper using mh.create_mappings

    3.2.b: Use mh.read_filter_create_extract to get the mappings (name_to_id, id_to_name)

4. Convert the dataframe columns to the live deployment format (column_names_to_ld_format)
5. Use put_data to write out a single row in the live deployment format to a file
6. Repeat 5 for as many as many rows as you'd like

NOTE: there is a higher level function will will run steps 3 and 4 and it is called convert_to_ld
"""

def get_ids_from_names(mapper, names, variable='Datapoint'):
    """
    Function to get the ids from the names based on the mapper file

    Parameters
    ----------
    mapper : pd.DataFrame
        mapper file
    names : List[str]
        (object/property) names to map to ids
    variable : str
        whether the names are object names (true) or property names (false)

    Returns
    -------
    ids : List[int]
        (object or property names)
    """
    variable_name = variable + ' Name'
    variable_id = variable + ' Id'
    ids = mapper.loc[mapper[variable_name].isin(names), variable_id].tolist()
    return ids


def column_names_to_ld_format(df, object_names, property_names, sep='___'):
    """
    Function to convert column names of historical data to live deployment data

    Parameters
    ----------
    df : pd.DataFrame
        data in the historical format --- assumes all the columns are tags and are currently labeled as object_names
    object_names : List[str]
        object/datapoint names
    property_names : List[str]
        property names
    sep : str
        separator for the object and property names: object_name{sep}property_name

    Returns
    -------
    df_copy : pd.DataFrame
        data with the columns in the live deployment format (this will be a copy of the original data not a reference)
    """
    # NOTE: could have a check to make sure the same number of object and property names are passed
    new_column_names = [object_name+sep+property_name for object_name, property_name in zip(object_names, property_names)]

    # make explicit deep copy
    df_copy = df.copy()

    # make sure the order is correct by creating a map
    new_name_map = {}

    for name in df_copy.columns.tolist():
        this_index = [i for i, new_name in enumerate(new_column_names) if name in new_name]
        new_name_map[name] = new_column_names[this_index[0]]

    return df_copy.rename(columns=new_name_map)

def convert_to_ld(df, mapper, mh, property_ids, object_variable_name='Datapoint', sep='___'):
    """
    Function to use a mapperhandler to filter the mapper, create the mappings (name_to_id and id_to_name) and format the data into a format to be written into live deployment files

    Parameters
    ----------
    df : pd.DataFrame
        data of interest where the columns are the object/datapoint names
    mapper : pd.DataFrame
        mapper file
    mh : MapperHandler
        initialized mapperhandler
    property_ids : List[int]
        property ids
    object_variable_name : str, default = Datapoint
        One of Datapoint or Object --- corresponding to the object/datapoint whichever variable is used in the mapper
    sep : str, default = ___
        separator for object and property name: object_name{sep}property_name

    Returns
    -------
    final : pd.DataFrame
        data that is ready to be written to live deployment using live deployment files --- columns are now object_name{sep}property_name
    sub : pd.DataFrame
        subsetted mapper --- you can also get the final object_names/object_ids/property_names/property_ids from this by just converting the column to a list
    name_to_id : Dict[Tuple[str, str]] -> Tuple[int, int]
        mapping from from (object name, property name) to (object id, property id)
    id_to_name : Dict[Tuple[int, int]] -> Tuple[str, str]
        mapping from (object id, property id) to (object name, property name)
    """
    # save the object names and get the object_ids
    initial_object_names = df.columns.tolist()
    object_ids = get_ids_from_names(mapper, initial_object_names, variable=object_variable_name)

    sub = mh.filter_mapper(mapper, object_ids=object_ids, property_ids=property_ids)
    name_to_id, id_to_name = mh.create_mappings(sub)
    object_names = sub[object_variable_name + ' Name'].tolist()
    property_names = sub['Property Name']

    final = column_names_to_ld_format(df, object_names, property_names, sep=sep)

    return final, sub, name_to_id, id_to_name

