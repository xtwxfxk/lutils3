# -*- coding: utf-8 -*-
__author__ = 'xtwxfxk'

from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go

def bbands(price, window_size=10, num_of_std=5):
    rolling_mean = price.rolling(window=window_size).mean()
    rolling_std  = price.rolling(window=window_size).std()
    upper_band = rolling_mean + (rolling_std*num_of_std)
    lower_band = rolling_mean - (rolling_std*num_of_std)
    return rolling_mean, upper_band, lower_band



def plot_stock(df, name='', index=None):

    bb_avg, bb_upper, bb_lower = bbands(df.close)

    colors = np.zeros(df.close.shape).astype(np.str)

    colors[df.open >= df.close] = 'red'
    colors[df.open < df.close] = 'green'

    df['ma5'] = df.close.rolling(window=5, min_periods=1).mean()
    df['ma10'] = df.close.rolling(window=10, min_periods=1).mean()

    fig = go.Figure(data=[
        go.Candlestick(
            x=index,
            open=df.open,
            high=df.high,
            low=df.low,
            close=df.close,
            name=name,
            yaxis='y2'),
        go.Scatter(y=df.ma5, line=dict(color='orange', width=1), name='MA 5', yaxis='y2'),
        # go.Scatter(y=df.ma10, line=dict(color='green', width=1), name='MA 10', yaxis='y2'),
        
    #     go.Scatter(y=bb_avg, line=dict(color='#ccc', width=1), name='MA 10', yaxis='y2'),
        go.Scatter(y=bb_upper, line=dict(color='#ccc', width=1), legendgroup='Bollinger Bands', name='Bollinger Bands', yaxis='y2'),
        go.Scatter(y=bb_lower, line=dict(color='#ccc', width=1), legendgroup='Bollinger Bands', showlegend=False, yaxis='y2'),
        
        go.Bar(y=df.volume, yaxis='y', name='volume', marker=dict(color=colors))
    ])

    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=.8,
            xanchor="right",
            x=1
        ), yaxis=dict(
            domain = [0, 0.2],
            showticklabels = False
        ), yaxis2=dict(
            domain = [0.2, 0.8]
        ), margin=dict(
            l=10,
            r=10,
            b=10,
            t=10,
            pad=1
        ), xaxis=dict(
            rangeselector = dict(
                visible = True,
                buttons = list([
                    dict(count=1,
                        label='reset',
                        step='all'),
                ])
            )
        )
    )

    return fig







