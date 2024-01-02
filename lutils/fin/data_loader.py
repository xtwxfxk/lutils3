# -*- coding: utf-8 -*-
__author__ = 'xtwxfxk'

import os, datetime
import numpy as np
import pandas as pd

TQ_ROOT = 'Z:/tq_data/ticks'
CTP_ROOT_1 = 'Y:/fin_data'
CTP_ROOT_2 = 'Y:/ctp_data'


def load_tq(exchange, symbol):
    path = os.path.join(TQ_ROOT, '%s.%s.h5' % (exchange, symbol))
    if os.path.exists(path):
        print('load %s' % path)
        df = pd.read_hdf(path)
        cs = []
        for c in df.columns.values:
            cs.append(c.rsplit('.', 1)[-1])
        columns = dict(zip(df.columns.values, cs))
        df = df.rename(columns=columns)


        df['datetime'] = pd.to_datetime(df.datetime, format='%Y-%m-%d %H:%M:%S.%f')

        return df[['datetime', 'last_price', 'volume', 'amount', 'open_interest',
        'bid_price1', 'bid_volume1',
        'ask_price1', 'ask_volume1',
        # 'bid_price2', 'bid_volume2',
        # 'ask_price2', 'ask_volume2',
        # 'bid_price3', 'bid_volume3',
        # 'ask_price3', 'ask_volume3',
        # 'bid_price4', 'bid_volume4',
        # 'ask_price4', 'ask_volume4',
        # 'bid_price5', 'bid_volume5',
        # 'ask_price5', 'ask_volume5'
        ]]

    else:
        return None


def load_ctp(exchange, symbol, ctp_root, start_date='2023-05-01'):

    dfs = []
    lists = os.listdir(ctp_root)
    lists.sort()
    for dp in lists:
        data_path = os.path.join(ctp_root, dp)
        if os.path.isdir(data_path) and dp > start_date and not dp.endswith('json'): # 目录 必须日期格式 后续修改
            dp = os.path.join(data_path, '%s.%s.h5' % (exchange, symbol))
            if not os.path.exists(dp): # and datetime.datetime.strptime(dp, '%Y-%m-%d').weekday() <:
                continue # break
            print('load %s' % dp)

            _df = pd.read_hdf(dp)

            # 大连商品交易所 夜盘 ActionDay 为下一个交易日, 修改为当前日期。 夜盘未过0点......
            # 郑州商品交易所 夜盘 TradingDay 为当前交易日。 夜盘未过0点......
            if exchange == 'DCE': 
                _df = _df[(_df['UpdateTime'] >= '09:00:00') & (_df['UpdateTime'] <= '23:00:00')] # 非9-23点 存在难处理数据
                _df.loc[((_df['UpdateTime'] > '15:15:00') & (_df['UpdateTime'] <= '23:59:59')), 'ActionDay'] = _df[_df['UpdateTime'] >= '09:00:00'].iloc[0]['ActionDay']

            dfs.append(_df)

    if len(dfs) > 0:
        df = pd.concat(dfs)[['Volume', 'Turnover', 'LastPrice', 'TradingDay', 'ActionDay', 'UpdateTime', 'UpdateMillisec', 'OpenInterest', 
        'BidPrice1', 'BidVolume1',
        'AskPrice1', 'AskVolume1',
        # 'BidPrice2', 'BidVolume2',
        # 'AskPrice2', 'AskVolume2',
        # 'BidPrice3', 'BidVolume3',
        # 'AskPrice3', 'AskVolume3',
        # 'BidPrice4', 'BidVolume4',
        # 'AskPrice4', 'AskVolume4',
        # 'BidPrice5', 'BidVolume5',
        # 'AskPrice5', 'AskVolume5',
        ]]

        df['datetime'] = df['ActionDay'] + ' ' + df['UpdateTime'] + '.' + df['UpdateMillisec'].astype(str)
        # df['tradetime'] = df['TradingDay'] + ' ' + df['UpdateTime'] + '.' + df['UpdateMillisec'].astype(str)
        
        df = df[['datetime', 'LastPrice', 'Volume', 'Turnover', 'OpenInterest', 'BidPrice1', 'BidVolume1', 'AskPrice1', 'AskVolume1',]].rename(columns={'datetime': 'datetime', 
            'LastPrice': 'last_price', 
            'Volume': 'volume', 
            'Turnover': 'amount', 
            'OpenInterest': 'open_interest',
            'BidPrice1': 'bid_price1', 
            'BidVolume1': 'bid_volume1', 
            'AskPrice1': 'ask_price1', 
            'AskVolume1': 'ask_volume1', 
            # 'BidPrice2': 'bid_price2', 
            # 'BidVolume2': 'bid_volume2', 
            # 'AskPrice2': 'ask_price2', 
            # 'AskVolume2': 'ask_volume2', 
            # 'BidPrice3': 'bid_price3', 
            # 'BidVolume3': 'bid_volume3', 
            # 'AskPrice3': 'ask_price3', 
            # 'AskVolume3': 'ask_volume3', 
            # 'BidPrice4': 'bid_price4', 
            # 'BidVolume4': 'bid_volume4', 
            # 'AskPrice4': 'ask_price4', 
            # 'AskVolume4': 'ask_volume4', 
            # 'BidPrice5': 'bid_price5', 
            # 'BidVolume5': 'bid_volume5', 
            # 'AskPrice5': 'ask_price5', 
            # 'AskVolume5': 'ask_volume5',
            }) # 'tradetime': 'tradetime'

        df.volume = df.volume.diff(1).fillna(0)
        df['datetime'] = pd.to_datetime(df.datetime, format='%Y%m%d %H:%M:%S.%f')

        return df
    else:
        return None


def _load(exchange, symbol, start_date=None):

    if start_date is None: start_date = '2023-05-01'

    dfs = []
    if start_date <= '2023-05-01':
        df_tq = load_tq(exchange, symbol)
        if df_tq is not None and not df_tq.empty:
            dfs.append(df_tq)

    df_ctp = load_ctp(exchange, symbol, CTP_ROOT_1, start_date)
    if df_ctp is not None and not df_ctp.empty:
        dfs.append(df_ctp)

    df_ctp = load_ctp(exchange, symbol, CTP_ROOT_2, start_date)
    if df_ctp is not None and not df_ctp.empty:
        dfs.append(df_ctp)

    if len(dfs) > 0:
        df = pd.concat(dfs)
        return df
    else:
        return None

def load(exchange, symbol, start_date=None, resample=None, **kwargs):
    
    
    df = _load(exchange, symbol, start_date)
    if df is not None and not df.empty:
        df.index = df.datetime

        if resample is None:
            return df
        else: # resample is not None
            resamples = []
            between_times = [['09:00', '10:15'], ['10:30', '11:30'], ['13:30', '15:00'], ['21:00', '23:00']]
            # between_times = [['09:05', '10:10'], ['10:35', '11:25'], ['13:35', '14:55'], ['21:05', '22:55']]

            for day in np.unique(df.index.date):
                df_day = df[df.index.date == day]
                
                for (start, end) in between_times:
                    df_hour = df_day.between_time(start, end)

                    df_resample = df_hour.resample(resample).last()
                    df_resample = df_resample.ffill()
                    resamples.append(df_resample)


            return pd.concat(resamples)


    else:
        return None

    # _df = df[(df.tradetime.dt.hour >= 8) | (df.tradetime.dt.hour <= 1)]
    
    # tradetime = df.loc[((df.tradetime.dt.dayofweek >= 4) & (df.tradetime.dt.hour >= 20)) | (df.tradetime.dt.dayofweek > 4)].tradetime + datetime.timedelta(days=2)
    # _df.loc[tradetime.index, 'tradetime'] = tradetime.values
    
    # _df.index = _df.tradetime
    # resample_ohlc = _df['last_price'].resample('1Min', closed='left', label='right').ohlc(_method='ohlc')
    
    # return resample_ohlc


def load_ohlc(exchange, symbol, start_date=None, resample='1T', fillna='ffill'):

    df = _load(exchange, symbol, start_date)
    if df is not None and not df.empty:
        df.index = df.datetime

        between_times = [['09:00', '10:15'], ['10:30', '11:30'], ['13:30', '15:00'], ['21:00', '23:00']]

        dfs = []
        for day in np.unique(df.index.date):
            df_day = df[df.index.date == day]
            for (start, end) in between_times:
                df_hour = df_day.between_time(start, end)

                if not df_hour.empty:
                    resample_ohlc = df_hour.last_price.resample(resample, closed='left', label='right').ohlc(_method='ohlc')
                    resample_volume = df_hour.volume.resample(resample, closed='left', label='right').sum()
                    resample_amount = df_hour.amount.resample(resample, closed='left', label='right').sum()

                    resample_df = pd.concat([resample_ohlc.fillna(method=fillna), resample_volume.fillna(method=fillna), resample_amount.fillna(method=fillna)], axis=1)
                    dfs.append(resample_df)

        return pd.concat(dfs)
    else:
        return None


    # df = pd.read_hdf(file_path)
    # df.index = pd.to_datetime(df.datetime, format='%Y-%m-%d %H:%M:%S.%f')

    # cs = []
    # for c in df.columns.values:
    #     cs.append(c.rsplit('.', 1)[-1])
    # columns = dict(zip(df.columns.values, cs))
    # df = df.rename(columns=columns)

    # resample_ohlc = df['last_price'].resample(resample, offset=21).ohlc(_method='ohlc')
    # resample_volume = df['volume'].resample(resample, offset=21).sum() # .to_frame().rename(columns={'volume': 'volume'})
    # resample_amount = df['amount'].resample(resample, offset=21).sum() # .to_frame().rename(columns={'amount': 'amount'})

    # df = pd.concat([resample_ohlc, resample_volume, resample_amount], axis=1)

    # # df = df[df.apply(in_time, axis=1)]
    # # if fillna:
    # #     df = df.fillna(method="ffill")
    # df = df.dropna()

    # return df

# def tick_2_kline(file_path, resampe='60s'):
#     _df = pd.read_hdf(file_path)
#     _df.index = pd.to_datetime(df.datetime, format='%Y-%m-%d %H:%M:%S.%f')
#     _df = _df[_df.apply(in_time, axis=1)]

#     _df.rolling(resampe, min_periods=1)


