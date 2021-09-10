# -*- coding: utf-8 -*-
__author__ = 'xtwxfxk'

import time, datetime, random, functools
import pandas as pd
from enum import Enum, unique

from pytdx.hq import TdxHq_API, TDXParams
from pytdx.config.hosts import hq_hosts

'''
category K线种类:
    0 5分钟K线 1 15分钟K线 2 30分钟K线 3 1小时K线 4 日K线
    5 周K线
    6 月K线
    7 1分钟
    8 1分钟K线 9 日K线
    10 季K线
    11 年K线

'''


class Market():
    SH = TDXParams.MARKET_SH    # 深圳
    SZ = TDXParams.MARKET_SZ    # 上海

class Category():
    KLINE_TYPE_5MIN = 0         # 5 分钟K 线
    KLINE_TYPE_15MIN = 1        # 15 分钟K 线
    KLINE_TYPE_30MIN = 2        # 30 分钟K 线
    KLINE_TYPE_1HOUR = 3        # 1 小时K 线
    KLINE_TYPE_DAILY = 4        # 日K 线
    KLINE_TYPE_WEEKLY = 5       # 周K 线
    KLINE_TYPE_MONTHLY = 6      # 月K 线
    KLINE_TYPE_EXHQ_1MIN = 7    # 1 分钟
    KLINE_TYPE_1MIN = 8         # 1 分钟K 线
    KLINE_TYPE_RI_K = 9         # 日K 线
    KLINE_TYPE_3MONTH = 10      # 季K 线
    KLINE_TYPE_YEARLY = 11      # 年K 线

# @unique
# class CategoryDailyCount(Enum):
#     KLINE_TYPE_5MIN = 48         # 5 分钟K 线
#     KLINE_TYPE_15MIN = 16        # 15 分钟K 线
#     KLINE_TYPE_30MIN = 8        # 30 分钟K 线
#     KLINE_TYPE_1HOUR = 4        # 1 小时K 线
#     KLINE_TYPE_DAILY = 1        # 日K 线
#     KLINE_TYPE_WEEKLY = 5       # 周K 线
#     KLINE_TYPE_MONTHLY = 6      # 月K 线
#     KLINE_TYPE_EXHQ_1MIN = 7    # 1 分钟
#     KLINE_TYPE_1MIN = 8         # 1 分钟K 线
#     KLINE_TYPE_RI_K = 9         # 日K 线
#     KLINE_TYPE_3MONTH = 10      # 季K 线
#     KLINE_TYPE_YEARLY = 11      # 年K 线


def reindex_datetime(func):

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        datas = func(*args, **kwargs)
        datas = datas.set_index(datas['datetime'], drop=True, inplace=False)
        datas = datas.drop(['date', 'datetime'], axis=1)

        return datas

    return wrapped

def reindex_date_datetime(func):

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        datas = func(*args, **kwargs)
        datas = datas.set_index([datas['date'], datas['datetime']], drop=True, inplace=False)
        datas = datas.drop(['date', 'datetime'], axis=1)
        return datas

    return wrapped

class LTdxHq(TdxHq_API):


    def __init__(self, multithread=False, heartbeat=False, auto_retry=False, raise_exception=False, random_connect=False):
        super(LTdxHq, self).__init__(multithread=multithread, heartbeat=heartbeat, auto_retry=auto_retry, raise_exception=raise_exception)
        if random_connect:
            host = random.choice(ltdxhq.get_hosts())
            self.connect(host[1], host[2])
        else:
            self.connect('119.147.212.81', 7709)

    # def __del__(self):
    #     print("del")
    #     # self.disconnect()
    #     if self.heartbeat_thread and self.heartbeat_thread.is_alive():
    #         self.stop_event.set()

    #     if self.client:
    #         print("disconnecting")
    #         try:
    #             self.client.shutdown(socket.SHUT_RDWR)
    #             self.client.close()
    #             self.client = None
    #         except Exception as e:
    #             print(str(e))
    #             if self.raise_exception:
    #                 raise TdxConnectionError("disconnect err")
    #         print("disconnected")

    # def __exit__(self, *args):
    #     pritn('exit')
    #     self.disconnect()

    def get_k_data(self, code, start, end, **kwargs):
        def __select_market_code(code):
            code = str(code)
            if code[0] in ['5', '6', '9'] or code[:3] in ["009", "126", "110", "201", "202", "203", "204"]:
                return 1
            return 0

        category = kwargs.get('category', Category.KLINE_TYPE_RI_K)
        market = kwargs.get('market', __select_market_code(code))
        end = end if end is not None else datetime.date.today().strftime('%Y-%m-%d')

        return self._get_k_data(code, category, market, start, end)

    def _get_k_data(self, code, category, market, start, end):
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

        df = pd.concat(dfs, axis=0).sort_values(by="datetime", ascending=True)
        # df = df.set_index('datetime', drop=True, inplace=False)\
        #     .drop(['year', 'month', 'day', 'hour', 'minute'], axis=1)[start_date:end_date]

        # df = df.set_index('datetime', drop=True, inplace=False).drop(['year', 'month', 'day', 'hour', 'minute'], axis=1)[start_date:end_date]
        df['date'] = df[['year', 'month', 'day']].apply(lambda x: '{0}-{1:02d}-{2:02d}'.format(x[0], x[1], x[2]), axis=1)
        df = df.drop(['year', 'month', 'day', 'hour', 'minute'], axis=1)
        df = df.loc[(df['datetime'] >= start_date) & (df['datetime'] < end_date)]

        return df

    # KLINE_TYPE_5MIN = 0         # 5 分钟K 线
    # KLINE_TYPE_15MIN = 1        # 15 分钟K 线
    # KLINE_TYPE_30MIN = 2        # 30 分钟K 线
    # KLINE_TYPE_1HOUR = 3        # 1 小时K 线
    # KLINE_TYPE_DAILY = 4        # 日K 线
    # KLINE_TYPE_WEEKLY = 5       # 周K 线
    # KLINE_TYPE_MONTHLY = 6      # 月K 线
    # KLINE_TYPE_EXHQ_1MIN = 7    # 1 分钟
    # KLINE_TYPE_1MIN = 8         # 1 分钟K 线
    # KLINE_TYPE_RI_K = 9         # 日K 线
    # KLINE_TYPE_3MONTH = 10      # 季K 线
    # KLINE_TYPE_YEARLY = 11      # 年K 线
    @reindex_date_datetime
    def get_k_data_1min(self, code, start='2000-01-01', end=None):
        return self.get_k_data(code=code, start=start, end=end, category=Category.KLINE_TYPE_1MIN)

    @reindex_date_datetime
    def get_k_data_5min(self, code, start='2000-01-01', end=None):
        return self.get_k_data(code=code, start=start, end=end, category=Category.KLINE_TYPE_5MIN)

    @reindex_date_datetime
    def get_k_data_15min(self, code, start='2000-01-01', end=None):
        return self.get_k_data(code=code, start=start, end=end, category=Category.KLINE_TYPE_15MIN)

    @reindex_date_datetime
    def get_k_data_30min(self, code, start='2000-01-01', end=None):
        return self.get_k_data(code=code, start=start, end=end, category=Category.KLINE_TYPE_30MIN)

    @reindex_date_datetime
    def get_k_data_1hour(self, code, start='2000-01-01', end=None):
        return self.get_k_data(code=code, start=start, end=end, category=Category.KLINE_TYPE_1HOUR)

    @reindex_datetime
    def get_k_data_daily(self, code, start='2000-01-01', end=None):
        return self.get_k_data(code=code, start=start, end=end, category=Category.KLINE_TYPE_DAILY)

    @reindex_datetime
    def get_k_data_weekly(self, code, start='2000-01-01', end=None):
        return self.get_k_data(code=code, start=start, end=end, category=Category.KLINE_TYPE_WEEKLY)

    @reindex_datetime
    def get_k_data_monthly(self, code, start='2000-01-01', end=None):
        return self.get_k_data(code=code, start=start, end=end, category=Category.KLINE_TYPE_MONTHLY)


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


    ltdxhq = LTdxHq() # heartbeat=True)
    ltdxhq.get_k_data_1min(code='603636', start='2013-01-01')



    # import tushare as ts
    # ts.set_token('xxxxx')
    # pro = ts.pro_api()

    # data = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')
    # data = pro.query('stock_basic', exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')