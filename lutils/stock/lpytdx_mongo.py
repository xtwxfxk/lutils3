# -*- coding: utf-8 -*-
__author__ = 'xtwxfxk'

import time, datetime, random, functools
import pandas as pd
import pymongo
from pymongo import MongoClient


def reindex_date(func):

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        datas = func(*args, **kwargs)
        datas = datas.set_index(pd.to_datetime(datas['date']), drop=True, inplace=False)
        datas = datas.drop(['date', 'datetime'], axis=1)

        return datas

    return wrapped

def reindex_date_datetime(func):

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        datas = func(*args, **kwargs)
        datas = datas.set_index([pd.to_datetime(datas['date']), pd.to_datetime(datas['datetime'])], drop=True, inplace=False)
        datas = datas.drop(['date', 'datetime'], axis=1)
        return datas

    return wrapped

class LTdxMongo():

    

    def __init__(self):
        pass


    def stock_list(self):
        data = pd.concat([pd.concat([self.to_df(self.get_security_list(j, i * 1000)).assign(sse='sz' if j == 0 else 'sh') for i in range(int(self.get_security_count(j) / 1000) + 1)], axis=0, sort=False) for j in range(2)], axis=0, sort=False)
        return data

    def get_k_data(self, code, start, end, **kwargs):
        def __select_market_code(code):
            code = str(code)
            if code[0] in ['5', '6', '9'] or code[:3] in ["009", "126", "110", "201", "202", "203", "204"]:
                return 1
            return 0

        category = kwargs.get('category', Category.KLINE_TYPE_RI_K)
        market = kwargs.get('market', __select_market_code(code))
        qfq = kwargs.get('qfq', True)
        end = end if end is not None else datetime.date.today().strftime('%Y-%m-%d')

        return self._get_k_data(code, category, market, start, end, qfq)

    def _get_k_data(self, code, category, market, start, end, qfq, **kwargs):
        start_time = datetime.datetime.strptime(start, '%Y-%m-%d') + datetime.timedelta(hours=9, minutes=30)
        end_time = datetime.datetime.strptime(end, '%Y-%m-%d') + datetime.timedelta(hours=15)

        start_date = datetime.datetime.strftime(start_time, '%Y-%m-%d %H:%M')
        end_date = datetime.datetime.strftime(end_time, '%Y-%m-%d %H:%M')
        if start_time > end_time:
            return None

        _start = 0
        _count = 800
        dfs = []
        while True:
            _df = self.to_df(self.get_security_bars(category, market, code, _start, _count))
            if _df.shape[0] < 1:
                break
            dfs.append(_df)
            _start = _start + _count
            if datetime.datetime.strptime(_df.head(1).at[0, 'datetime'], '%Y-%m-%d %H:%M') < start_time:
                break

        df = pd.concat(dfs, axis=0).sort_values(by='datetime', ascending=True)

        df['datetime'] = df['datetime'] + ':00'

        df = df.assign(date=df[['year', 'month', 'day']].apply(lambda x: '{0}-{1:02d}-{2:02d}'.format(x[0], x[1], x[2]), axis=1))
        df = df.drop(['year', 'month', 'day', 'hour', 'minute'], axis=1)
        df = df.loc[(df['datetime'] >= start_date) & (df['datetime'] < end_date)]
        df = df.rename(columns={'vol': 'volume'})

        ############################## qfq #############################
        if qfq:
            xdxr = self.to_df(self.get_xdxr_info(market, code))
            xdxr = xdxr.assign(date=xdxr[['year', 'month', 'day']].apply(lambda x: '{0}-{1:02d}-{2:02d}'.format(x[0], x[1], x[2]), axis=1))
            # xdxr = xdxr.drop(['year', 'month', 'day'], axis=1)
            xdxr = xdxr[xdxr['category'] == 1]
            data = df.groupby('date').first().join(xdxr[['category', 'fenhong', 'peigu', 'peigujia', 'songzhuangu', 'date']].set_index('date'), how='left', on='date')
            data = df.set_index('datetime').join(data.reset_index().set_index('datetime')[['category', 'fenhong', 'peigu', 'peigujia', 'songzhuangu']], how='left', on='datetime').reset_index()

            data.category = data.category.fillna(method='ffill')

            data = data.fillna(0)
            data['preclose'] = (data['close'].shift(1) * 10 - data['fenhong'] + data['peigu'] * data['peigujia']) / (10 + data['peigu'] + data['songzhuangu'])

            data['adj'] = (data['preclose'].shift(-1) / data['close']).fillna(1)[::-1].cumprod()

            for col in ['open', 'high', 'low', 'close', 'preclose']:
                data[col] = data[col] * data['adj']

            decimals = pd.Series([2, 2, 2, 2], index=['open', 'close', 'high', 'low'])
            data = data.round(decimals)
            data['volume'] = data['volume']  if 'volume' in data.columns else data['vol']
            try:
                data['high_limit'] = data['high_limit'] * data['adj']
                data['low_limit'] = data['high_limit'] * data['adj']
            except:
                pass
            df = data.drop(['fenhong', 'peigu', 'peigujia', 'songzhuangu', 'category', 'preclose', 'adj'], axis=1, errors='ignore')
        return df


    @reindex_date_datetime
    def get_k_data_1min(self, code, start='2000-01-01', end=None, **kwargs):
        return self.get_k_data(code=code, start=start, end=end, type='1min', **kwargs)

    @reindex_date_datetime
    def get_k_data_5min(self, code, start='2000-01-01', end=None, **kwargs):
        return self.get_k_data(code=code, start=start, end=end, type='5min', **kwargs)

    @reindex_date_datetime
    def get_k_data_15min(self, code, start='2000-01-01', end=None, **kwargs):
        return self.get_k_data(code=code, start=start, end=end, type='15min', **kwargs)

    @reindex_date_datetime
    def get_k_data_30min(self, code, start='2000-01-01', end=None, **kwargs):
        return self.get_k_data(code=code, start=start, end=end, type='30min', **kwargs)

    @reindex_date_datetime
    def get_k_data_1hour(self, code, start='2000-01-01', end=None, **kwargs):
        return self.get_k_data(code=code, start=start, end=end, type='1hour', **kwargs)

    @reindex_date
    def get_k_data_daily(self, code, start='2000-01-01', end=None, **kwargs):
        return self.get_k_data(code=code, start=start, end=end, type='daily', **kwargs)

    # @reindex_date
    # def get_k_data_weekly(self, code, start='2000-01-01', end=None, **kwargs):
    #     return self.get_k_data(code=code, start=start, end=end, type='weekly', **kwargs)

    # @reindex_date
    # def get_k_data_monthly(self, code, start='2000-01-01', end=None, **kwargs):
    #     return self.get_k_data(code=code, start=start, end=end, type='monthly', **kwargs)

    # def to_qfq(self, code, df):
    #     xdxr = self.to_df(self.get_xdxr_info(1, code))

    #     if xdxr.shape[0] < 1:
    #         return df

    #     xdxr = xdxr.assign(date=xdxr[['year', 'month', 'day']].apply(lambda x: '{0}-{1:02d}-{2:02d}'.format(x[0], x[1], x[2]), axis=1))
    #     xdxr = xdxr.drop(['year', 'month', 'day'], axis=1)
    #     xdxr = xdxr.set_index('date', drop=False, inplace=False)

    #     info = xdxr[xdxr['category'] == 1]

    #     if info.shape[0] > 0:

    #         data = df.join(info[['category', 'fenhong', 'peigu', 'peigujia', 'songzhuangu']], how="left")

    #         data.category = data.category.fillna(method='ffill')

    #         data = data.fillna(0)
    #         data['preclose'] = (data['close'].shift(1) * 10 - data['fenhong'] + data['peigu'] * data['peigujia']) / (10 + data['peigu'] + data['songzhuangu'])

    #         data['adj'] = (data['preclose'].shift(-1) / data['close']).fillna(1)[::-1].cumprod()

    #         for col in ['open', 'high', 'low', 'close', 'preclose']:
    #             data[col] = data[col] * data['adj']

    #         data['volume'] = data['volume']  if 'volume' in data.columns else data['vol']
    #         data = data.drop(['fenhong', 'peigu', 'peigujia', 'songzhuangu', 'category', 'preclose', 'adj'], axis=1, errors='ignore')

    #         return data
    #     else:
    #         return df


    def get_hosts(self):
        return hq_hosts

if __name__ == '__main__':
    # https://rainx.gitbooks.io/pytdx/content/
    # ltdxhq = LTdxHq(heartbeat=True)
    # # print(lpytdx.get_hosts())
    # ltdxhq.connect('119.147.212.81', 7709)
    # df = ltdxhq.get_k_data(code='603636', start='2013-01-01', category=Category.KLINE_TYPE_RI_K)
    # ltdxhq.disconnect()
    # print(df)


    ltdxhq = LTdxMongo() # heartbeat=True)
    ltdxhq.get_k_data_1min(code='603636', start='2013-01-01')



    # import tushare as ts
    # ts.set_token('xxxxx')
    # pro = ts.pro_api()

    # data = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')
    # data = pro.query('stock_basic', exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')