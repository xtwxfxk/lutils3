# -*- coding: utf-8 -*-
__author__ = 'xtwxfxk'

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


@unique
class Market(Enum):
    SH = TDXParams.MARKET_SH    # 深圳
    SZ = TDXParams.MARKET_SZ    # 上海

@unique
class Category(Enum):
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

class LPytdx(object):

    def __init__(self):
        
        self.init_pytdx()

    def init_pytdx(self):
        self.api = TdxHq_API(heartbeat=True)

    def connect(self, ip, port=7709): # ip='119.147.212.81'
        self.api.connect(ip, port)

    def calc_daily_count(self, category):
        daily_count = 1
        if category == Category.KLINE_TYPE_5MIN:
            daily_count = 1
        elif category == Category.KLINE_TYPE_5MIN:
            daily_count = 1
        elif category == Category.KLINE_TYPE_15MIN:
            daily_count = 1
        elif category == Category.KLINE_TYPE_30MIN:
            daily_count = 1
        elif category == Category.KLINE_TYPE_1HOUR:
            daily_count = 1
        elif category == Category.KLINE_TYPE_DAILY:
            daily_count = 1
        elif category == Category.KLINE_TYPE_WEEKLY:
            daily_count = 1
        elif category == Category.KLINE_TYPE_MONTHLY:
            daily_count = 1
        elif category == Category.KLINE_TYPE_EXHQ_1MIN:
            daily_count = 1
        elif category == Category.KLINE_TYPE_1MIN:
            daily_count = 1
        elif category == Category.KLINE_TYPE_RI_K:
            daily_count = 1
        elif category == Category.KLINE_TYPE_3MONTH:
            daily_count = 1
        elif category == Category.KLINE_TYPE_YEARLY:
            daily_count = 1
        else:
            daily_count = 1
        return daily_count

    def get_k_data(self, code, cateogry, market, start, end):
        daily_count = 1
        # cateogry

    def get_hosts(self):
        return hq_hosts

if __name__ == '__main__':
    # https://rainx.gitbooks.io/pytdx/content/
    lpytdx = LPytdx()
    # print(lpytdx.get_hosts())
    # lpytdx.connect('119.147.212.81', 7709)

    print(lpytdx.calc_daily_count())
