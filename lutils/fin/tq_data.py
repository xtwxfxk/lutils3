# -*- coding: utf-8 -*-
__author__ = 'xtwxfxk'

import pandas as pd


def in_time(row):
    if row.name.weekday() < 5:
        hour_second = row.name.strftime('%H:%M')
        if (hour_second >= '09:00' and hour_second <= '10:15') or (hour_second >= '10:30' and hour_second <= '11:30') or (hour_second >= '13:30' and hour_second <= '15:00') or (hour_second >= '21:00' and hour_second <= '23:00'):
            return True
        return False
    return False

def in_day(row):
    if row.name.weekday() < 5:
        return True
    return False

def load_hdf_ohlc(file_path, resample='1Min', fillna='ffill'):

    df = pd.read_hdf(file_path)
    df.index = pd.to_datetime(df.datetime, format='%Y-%m-%d %H:%M:%S.%f')

    cs = []
    for c in df.columns.values:
        cs.append(c.rsplit('.', 1)[-1])
    columns = dict(zip(df.columns.values, cs))
    df = df.rename(columns=columns)

    resample_ohlc = df['last_price'].resample(resample, offset=21).ohlc(_method='ohlc')
    resample_volume = df['volume'].resample(resample, offset=21).sum() # .to_frame().rename(columns={'volume': 'volume'})
    resample_amount = df['amount'].resample(resample, offset=21).sum() # .to_frame().rename(columns={'amount': 'amount'})

    df = pd.concat([resample_ohlc, resample_volume, resample_amount], axis=1)

    # df = df[df.apply(in_time, axis=1)]
    # if fillna:
    #     df = df.fillna(method="ffill")
    df = df.dropna()

    return df

def tick_2_kline(file_path, resampe='60s'):
    _df = pd.read_hdf(file_path)
    _df.index = pd.to_datetime(df.datetime, format='%Y-%m-%d %H:%M:%S.%f')
    _df = _df[_df.apply(in_time, axis=1)]

    _df.rolling(resampe, min_periods=1)


