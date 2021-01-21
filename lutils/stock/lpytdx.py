# -*- coding: utf-8 -*-
__author__ = 'xtwxfxk'

import time, datetime
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

class LTdxHq(TdxHq_API):


    def get_k_data(self, code, start, **kwargs):
        def __select_market_code(code):
            code = str(code)
            if code[0] in ['5', '6', '9'] or code[:3] in ["009", "126", "110", "201", "202", "203", "204"]:
                return 1
            return 0

        category = kwargs.get('category', Category.KLINE_TYPE_RI_K)
        market = kwargs.get('market', __select_market_code(code))
        end = kwargs.get('end', datetime.date.today().strftime('%Y-%m-%d'))

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
            dfs.append(_df)
            _start = _start + _count

            if datetime.datetime.strptime(_df.head(1).at[0, 'datetime'], '%Y-%m-%d %H:%M') < start_time:
                break

        df = pd.concat(dfs, axis=0).sort_values(by="datetime", ascending=True)
        df = df.assign(date=df['datetime']).assign(code=str(code))\
            .set_index('date', drop=False, inplace=False)\
            .drop(['year', 'month', 'day', 'hour', 'minute', 'datetime'], axis=1)[start_date:end_date]

        return df


    def get_hosts(self):
        return hq_hosts

if __name__ == '__main__':
    # https://rainx.gitbooks.io/pytdx/content/
    ltdxhq = LTdxHq(heartbeat=True)
    # print(lpytdx.get_hosts())
    ltdxhq.connect('119.147.212.81', 7709)

    df = ltdxhq.get_k_data(code='603636', start='2019-02-01', category=Category.KLINE_TYPE_15MIN)
    ltdxhq.disconnect()
    print(df)
