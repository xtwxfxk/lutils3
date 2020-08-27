# -*- coding: utf-8 -*-
__author__ = 'xtwxfxk'

from datetime import datetime
import numpy as np
import pandas as pd
import chart_studio.plotly as py

INCREASING_COLOR = '#17BECF'
DECREASING_COLOR = '#7F7F7F'


def plot_stock(df, name=''):
    data = [dict(
        type = 'candlestick',
        open = df.open,
        high = df.high,
        low = df.low,
        close = df.close,
        x = df.index,
        yaxis = 'y2',
        name = name,
        increasing = dict( line = dict(color = INCREASING_COLOR)),
        decreasing = dict( line = dict(color = DECREASING_COLOR)),
    )]

    layout=dict()

    fig = dict(data=data, layout=layout)

    fig['layout'] = dict()
    fig['layout']['plot_bgcolor'] = 'rgb(250, 250, 250)'
    fig['layout']['xaxis'] = dict( rangeselector = dict( visible = True ) )
    fig['layout']['yaxis'] = dict( domain = [0, 0.2], showticklabels = False )
    fig['layout']['yaxis2'] = dict( domain = [0.2, 0.8] )
    fig['layout']['legend'] = dict( orientation = 'h', y=0.9, x=0.3, yanchor='bottom' )
    fig['layout']['margin'] = dict( t=40, b=40, r=40, l=40 )


    rangeselector=dict(
        visibe = True,
        x = 0, y = 0.9,
        bgcolor = 'rgba(150, 200, 250, 0.4)',
        font = dict( size = 13 ),
        buttons=list([
            dict(count=1,
                 label='reset',
                 step='all'),
            dict(count=1,
                 label='1yr',
                 step='year',
                 stepmode='backward'),
            dict(count=3,
                label='3 mo',
                step='month',
                stepmode='backward'),
            dict(count=1,
                label='1 mo',
                step='month',
                stepmode='backward'),
            dict(step='all')
        ]))
        
    fig['layout']['xaxis']['rangeselector'] = rangeselector


    mv_y = df.close.rolling(window=5, min_periods=1).mean()
    mv_x = list(df.index)

    # Clip the ends
    mv_x = mv_x[5:-5]
    mv_y = mv_y[5:-5]

    fig['data'].append( dict( x=mv_x, y=mv_y, type='scatter', mode='lines', 
                             line = dict( width = 1 ),
                             marker = dict( color = '#E377C2' ),
                             yaxis = 'y2', name='Moving Average' ) )


    colors = []
    for i in range(len(df.close)):
        if i != 0:
            if df.close[i] > df.close[i-1]:
                colors.append(INCREASING_COLOR)
            else:
                colors.append(DECREASING_COLOR)
        else:
            colors.append(DECREASING_COLOR)


    fig['data'].append( dict( x=df.index, y=df.volume,                         
                         marker=dict( color=colors ),
                         type='bar', yaxis='y', name='Volume' ) )

    def bbands(price, window_size=10, num_of_std=5):
        rolling_mean = price.rolling(window=window_size).mean()
        rolling_std  = price.rolling(window=window_size).std()
        upper_band = rolling_mean + (rolling_std*num_of_std)
        lower_band = rolling_mean - (rolling_std*num_of_std)
        return rolling_mean, upper_band, lower_band

    bb_avg, bb_upper, bb_lower = bbands(df.close)

    fig['data'].append( dict( x=df.index, y=bb_upper, type='scatter', yaxis='y2', 
                             line = dict( width = 1 ),
                             marker=dict(color='#ccc'), hoverinfo='none', 
                             legendgroup='Bollinger Bands', name='Bollinger Bands') )

    fig['data'].append( dict( x=df.index, y=bb_lower, type='scatter', yaxis='y2',
                             line = dict( width = 1 ),
                             marker=dict(color='#ccc'), hoverinfo='none',
                             legendgroup='Bollinger Bands', showlegend=False ) )



    # py.iplot( fig, filename = 'candlestick-test-3', validate = False )
    return fig















