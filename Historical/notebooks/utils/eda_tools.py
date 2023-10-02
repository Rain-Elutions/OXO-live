import numpy as np
import pandas as pd

# Plotly
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objs as go

def select_good_period(df: pd.DataFrame, col_num: int) -> go.Figure:
    """
    Function to select the good time period from a dataframe

    Parameters
    ----------
    df : pd.DataFrame
        data of interest
    col_num : int
        column number of interest
    
    Returns
    -------
    fig : go.Figure
        plot of ?
    """
    
    # col_num is the number you want to see in the graph i<=10
    
    selected_cols = np.random.choice(df.shape[1], size=col_num, replace=False)
    df_selected = df.iloc[:,selected_cols]

    fig = make_subplots(rows=1, cols=1)
    
    # NOTE: move the resampling outside of the for loop so the resampling happens once instead of multiple times ... 
    for i, col in enumerate(df_selected.columns):
        df_day = df_selected.groupby(df.index.date)[col].mean()
        fig.add_trace(go.Scatter(x=df_day.index, y=df_day,
                                 mode='lines',
                                 marker=dict(color=px.colors.sequential.Viridis[i]),
                                 name=f'{col}'),
                      row=1, col=1)
        
    # General Styling
    fig.update_layout(height=600, bargap=0.2,
                      margin=dict(b=50,r=30,l=100),
                      title = "<span style='font-size:36px; font-family:Times New Roman'>Feature Pattern</span>",                  
                      plot_bgcolor='rgb(242,242,242)',
                      paper_bgcolor = 'rgb(242,242,242)',
                      font=dict(family="Times New Roman", size= 14),
                      hoverlabel=dict(font_color="floralwhite"),
                      showlegend=False)
    return fig


def single_column_vis(df: pd.DataFrame, col_name: str, title_name: str) -> go.Figure:
    """
    Helper function to visualize a single column

    Parameters
    ----------
    df : pd.DataFrame
        data of interest
    col_name : str
        column to visualize
    title_name : str
        title for the plot
    
    Returns
    -------
    fig : go.Figure
        visualization for a single variable
    """
    
    fig = make_subplots(rows=2, cols=2,
                        specs=[
                               [{'colspan':2}, None],
                               [{'type':'histogram'}, {'type':'bar'}]],
                        column_widths=[0.5,0.5],
                        vertical_spacing=0.1, horizontal_spacing=0.1,
                        subplot_titles=(
                                        f'Daily {col_name} Trend',
                                        f'{col_name} Distribution',
                                        f'{col_name} Box')
                       )

    # Top
    df_day = df.groupby(df.index.date)[col_name].mean()
    fig.add_trace(go.Scatter(x=df_day.index, y=df_day,
                             mode='lines',
                             marker=dict(color=px.colors.sequential.Viridis[5]),
                             name='Daily Trend'),
                  row=1, col=1)

    # Left Bottom Chart
    fig.add_trace(go.Histogram(x=df[col_name], 
                               name='Distribution', 
                               marker = dict(color = px.colors.sequential.Viridis[3])
                               ), 
                  row = 2, col = 1)

    fig.update_xaxes(showgrid = False, showline = True, 
                     linecolor = 'gray', linewidth = 2, 
                     row = 2, col = 1)
    fig.update_yaxes(showgrid = False, gridcolor = 'gray', 
                     gridwidth = 0.5, showline = True, 
                     linecolor = 'gray', linewidth = 2, 
                     row = 2, col = 1)

    # Right Bottom Chart
    fig.add_trace(go.Box(y=df[col_name],
                         name=f'{col_name}Box'

                             ),
                             row=2, col=2)

    fig.update_xaxes(showgrid = False, linecolor='gray', 
                     linewidth = 2, zeroline = False, 
                     row=2, col=2)
    fig.update_yaxes(showgrid = False, linecolor='gray',
                     linewidth=2, zeroline = False, 
                     row=2, col=2)


    # General Styling
    # NOTE: again consider doing this else where ...
    fig.update_layout(height=700, bargap=0.2,
                      margin=dict(b=50,r=30,l=100),
                      title = f"<span style='font-size:36px; font-family:Times New Roman'>{col_name} {title_name}</span>",                  
                      plot_bgcolor='rgb(242,242,242)',
                      paper_bgcolor = 'rgb(242,242,242)',
                      font=dict(family="Times New Roman", size= 14),
                      hoverlabel=dict(font_color="floralwhite"),
                      showlegend=False)
    return fig


def lines_plot(df: pd.DataFrame, list_of_col: list) -> go.Figure:
    """
    Function to make two line plots

    Parameters
    ----------
    df : pd.DataFrame
        data of interest
    col_one : str
        name of the first variable of interest
    col_two : str
        name of the second variable of interest

    Returns
    -------
    fig : go.Figure
        two line plots
    """
    
    fig = make_subplots(rows=1, cols=2,
                        specs=[
                               [{'colspan':2}, None]
                               ],

                        vertical_spacing=0.1, horizontal_spacing=0.1,
                        # subplot_titles=(
                        #                 'Daily Trend',
                        #                 )
                       )


    # Top Chart
    for col in list_of_col:
        
        df_day = df.groupby(df.index.date)[col].mean()
        fig.add_trace(go.Scatter(x=df_day.index, y=df_day,
                                 mode='lines',
                                 name=f'{col}'),
                      row=1, col=1)

    # General Styling
    # NOTE: could also do this elsewhere
    fig.update_layout(width=900, height=600, bargap=0.2,
                      margin=dict(b=50,r=30,l=100),
                      title_x=0.5,
                      yaxis_title="5FI696 Level",
                      title = {'text':"<span style='font-size:36px; font-family:Times New Roman'>LNG Level Before & After Optimization</span>"},                  
                      plot_bgcolor='rgb(242,242,242)',
                      paper_bgcolor = 'rgb(242,242,242)',
                      font=dict(family="Times New Roman", size= 14),
                      hoverlabel=dict(font_color="floralwhite"),
                      showlegend=True)
    return fig

def secondary_y_line_plot(df: pd.DataFrame, first_axis_tag: list, second_axis_tag: list) -> go.Figure:
    """
    Function to make line secondary_y_line_plot

    Parameters
    ----------
    df: pd.DataFrame
        should have first_axis_tag & second_axis_tag in the columns
        should have the time-stamps as the index
         
    first_axis_tag: list,
                    list of tags correspond to first axis
                    
    second_axis_tag: list,
                     list of tags correspond to second axis
                     
    Returns
    -------
    fig: go.Figure

    """
    
    fig = make_subplots(rows=1, cols=1,
                        specs=[
                               [{"secondary_y": True}]
                               ],

                        vertical_spacing=0.1, horizontal_spacing=0.1,
                        # subplot_titles=(
                        #                 'Daily Trend',
                        #                 )
                       )


    for column in first_axis_tag:
        df_day = df.groupby(df.index.date)[column].mean()
        fig.add_trace(go.Scatter(x=df_day.index, y=df_day,
                                 mode='lines',
                                 #marker=dict(color=px.colors.sequential.Viridis[2]),
                                 name=f'{column}'),
                      secondary_y=False,
                      row=1, col=1)
    for column in second_axis_tag:
        df_day = df.groupby(df.index.date)[column].mean()
        fig.add_trace(go.Scatter(x=df_day.index, y=df_day,
                                 mode='lines',
                                 #marker=dict(color=px.colors.sequential.Viridis[2]),
                                 name=f'{column}'),
                      secondary_y=True,                      
                      row=1, col=1)

    # Set y-axes titles
    fig.update_yaxes(title_text="<b>LNG & Output</b> Level", secondary_y=False)
    fig.update_yaxes(title_text="<b>Delta, Butane & Propane</b> Level", secondary_y=True)
    # General Styling
    # NOTE: could also do this elsewhere
    fig.update_layout(width=900, height=600, bargap=0.2,
                      margin=dict(b=50,r=30,l=100),
                      title_x=0.5,
                      #yaxis_title="LNG Level",
                      title = {'text':"<span style='font-size:36px; font-family:Times New Roman'>All LNG Product Level</span>"},                  
                      plot_bgcolor='rgb(242,242,242)',
                      paper_bgcolor = 'rgb(242,242,242)',
                      font=dict(family="Times New Roman", size= 14),
                      hoverlabel=dict(font_color="floralwhite"),
                      showlegend=True)
    return fig
    
