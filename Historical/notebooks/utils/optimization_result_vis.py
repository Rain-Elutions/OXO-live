# Plotly
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objs as go
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler


def Controllable_Value_Plot(final_df: pd.DataFrame, controllable_tags: list) -> go.Figure:
    """
    Return plots showing the trends of controllable before and after optimization, with legends group.

    Parameters
    ----------
    final_df: dataframe including optimized & actual controllable values. Columns need to be tag_Optimized & tag
    controllable_tag: list of all the controllable name
    
    Returns
    -------
    fig : go.Figure
          plot of 
    """
    
    

    fig = make_subplots(rows = len(controllable_tags), cols=2,
                        specs = [[{'colspan':2}, None]]*len(controllable_tags), 
                        shared_xaxes = True,
                        subplot_titles=(controllable_tags)
                       )
    
    final_df = final_df.resample('12h').mean()
    result_df = final_df[[tag+'_Optimized' for tag in controllable_tags]]
    X_raw = final_df[controllable_tags]
    
    for i in range(1):
        result_optimized = result_df.iloc[:, i]
        raw_data = X_raw.iloc[:, i]
        fig.add_trace(go.Scatter(x=result_optimized.index, y=result_optimized,
                                 legendgroup="Optimized", 
                                 showlegend=True, 
                                 mode='lines',
                                 marker=dict(color=px.colors.sequential.Viridis[0]),
                                 name=f'Controllable Optimized'
                                ),
                      row=i+1, col=1)
        fig.add_trace(go.Scatter(x=raw_data.index, y=raw_data,
                                 legendgroup="Actual", 
                                 showlegend=True,
                                 mode='lines',
                                 marker=dict(color=px.colors.sequential.Viridis[5]),
                                 name=f'Controllable Acutal'
                                ),
                      row=i+1, col=1)
        
    for i in range(1, len(controllable_tags)):
        result_optimized = result_df.iloc[:, i]
        raw_data = X_raw.iloc[:, i]
        fig.add_trace(go.Scatter(x=result_optimized.index, y=result_optimized,
                                 legendgroup="Optimized", 
                                 showlegend=False, 
                                 mode='lines',
                                 marker=dict(color=px.colors.sequential.Viridis[0]),
                                 name=f'Controllable Optimized'
                                ),
                      row=i+1, col=1)
        fig.add_trace(go.Scatter(x=raw_data.index, y=raw_data,
                                 legendgroup="Actual", 
                                 showlegend=False,
                                 mode='lines',
                                 marker=dict(color=px.colors.sequential.Viridis[5]),
                                 name=f'Controllable Acutal'
                                ),
                      row=i+1, col=1)
        
    # General Styling
    fig.update_layout(height=1000, bargap=0.2,
                      margin=dict(b=50,r=30,l=100),
                      title = "<span style='font-size:36px; font-family:Times New Roman'>LNG CO2 Removal - Actual vs Optimized</span>",                  
                      plot_bgcolor='rgb(242,242,242)',
                      paper_bgcolor = 'rgb(242,242,242)',
                      font=dict(family="Times New Roman", size= 14),
                      hoverlabel=dict(font_color="floralwhite"),
                      showlegend=True)
    return fig


# Client's demand, for each time stamp of the optimized data, we use euclidean_dist algorithm to find the most similar point (validation point) in the training data
def find_nearest_point(row, validation_df: pd.DataFrame, ctrl_tag: list,TRAINING_TESTING_CUTOFF_DATE: str) -> pd.DataFrame:
    '''
    Function to find four validation points to support directive
    
    Parameters:
    row: row of the directive Dataframe
    validation_df: Training Data; Data to find the validation point
    ctrl_tag: list of the controllable tags, need to be included in both row and validation_df
    TRAINING_TESTING_CUTOFF_DATE: string of the cutoff of the training and testing date
    
    '''

    reference_df = validation_df[ctrl_tag].loc[:TRAINING_TESTING_CUTOFF_DATE, :].dropna()
    reference_df['Nearest_TimeStamp'] = reference_df.index
    reference_df['Euclidean_Dist'] = 0
    
    euclidean_dist_controllable = np.square(reference_df.iloc[:, :len(ctrl_tag)]-row[ctrl_tag].values).sum(axis=1)
    reference_df['Euclidean_Dist'] = euclidean_dist_controllable
    
    ind = np.argpartition(euclidean_dist_controllable, 4)[:4] # select the first 4 cloest time stamp
    four_row_result = reference_df.iloc[ind, :]
    single_row_df = pd.concat([four_row_result[col] for col in four_row_result.columns], axis=0).reset_index(drop=True)
    column_names = [f'{col}_Validation{i+1}' for col in four_row_result.columns for i in range(4)]
    validation_result = pd.DataFrame(single_row_df.values.reshape(1, -1), columns=column_names)

    return validation_result


def scale_data(final_df: pd.DataFrame, train_df: pd.DataFrame, ctrl_tag: list, TRAINING_TESTING_CUTOFF_DATE: str) -> np.ndarray:
    '''
    Function to scale the data
    
    Parameters:
    - final_df: dataframe including optimized & actual controllable values
    - original_df: original dataframe
    - ctrl_tag: list of the controllable tags, need to be included in both row and validation_df
    - TRAINING_TESTING_CUTOFF_DATE: string of the cutoff of the training and testing date
    
    Returns:
    - train_df_scaled: scaled training data
    - optimized_df: scaled optimization data

    '''
    # scale the train_df
    scaler = MinMaxScaler()
    train_df_scaled = scaler.fit_transform(train_df)
    
    # scale the optimized_df
    optimized_df = final_df.iloc[:, :len(ctrl_tag)]
    optimized_df = scaler.transform(optimized_df)

    return train_df_scaled, optimized_df


def find_similar_datapoints_knn(train_df_scaled: np.ndarray, optimized_df: np.ndarray, num_neighbors=4, metric='euclidean') -> np.ndarray:
    '''
    Function to find similar datapoints using KNN
    
    Parameters:
    - train_df_scaled: scaled training data
    - optimized_df: scaled optimized data

    Returns:
    - indices: indices of the similar datapoints

    '''

    # Implement KNN
    nn = NearestNeighbors(n_neighbors=num_neighbors, metric=metric)
    nn.fit(train_df_scaled)
    distances, indices = nn.kneighbors(optimized_df)
    
    return indices


def generate_validation_result(final_df: pd.DataFrame, original_df: pd.DataFrame, ctrl_tag: list, TRAINING_TESTING_CUTOFF_DATE: str)-> pd.DataFrame:
    '''
    Function to generate validation result

    Parameters:
    - final_df: dataframe including optimized & actual controllable values
    - original_df: original dataframe
    - ctrl_tag: list of the controllable tags, need to be included in both row and validation_df
    - TRAINING_TESTING_CUTOFF_DATE: string of the cutoff of the training and testing date

    Returns:
    - result: a dataframe of the validation result
    '''
    train_df = original_df[ctrl_tag].loc[:TRAINING_TESTING_CUTOFF_DATE]
    train_df_scaled, optimized_df = scale_data(final_df, train_df, ctrl_tag, TRAINING_TESTING_CUTOFF_DATE)
    indices = find_similar_datapoints_knn(train_df_scaled, optimized_df)

    train_df['Nearest_TimeStamp'] = train_df.index
    result = pd.DataFrame()
    # iterate through all time stamps
    for index in indices:
        four_row_result = train_df.iloc[index, :]
        single_row_df = pd.concat([four_row_result[col] for col in four_row_result.columns], axis=0).reset_index(drop=True)
        column_names = [f'{col}_Validation{i+1}' for col in four_row_result.columns for i in range(indices.shape[1])]
        validation_result = pd.DataFrame(single_row_df.values.reshape(1, -1), columns=column_names)
        result = pd.concat([result, validation_result])
    
    # drop rows with missing values in the final_df
    # new_final_df = final_df[~np.isnan(final_df.iloc[:, :21]).any(axis=1)]
    # result.index = new_final_df.index
    
    # new_final_df = pd.concat([final_df, result], axis=1)
    
    return result
    # return new_final_df #, indices
    
