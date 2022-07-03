# -*- coding: utf-8 -*-
__author__ = 'xtwxfxk'

import os
import sys
import time
import re
import socket
import socks
import threading
import random
import gzip
import datetime
import http.client as httpclient
import logging
import requests # requesocks

from http import cookiejar
from urllib.parse import urlparse
import io, urllib


#from scrapy.selector import Selector
from lxml import html
from bs4 import BeautifulSoup
from lutils.bitvise import Bitvise
from lutils import read_random_lines, LUTILS_ROOT
from lutils.lrequest import free_port, getaddrinfo
from hyper.contrib import HTTP20Adapter

__all__ = ['LRequests']

#socket.setdefaulttimeout(200)

logger = logging.getLogger('lutils')

NOT_REQUEST_CODE = [404, ]

header_path = os.path.join(LUTILS_ROOT, 'header')
USER_AGENT_DIR = os.path.join(LUTILS_ROOT, 'user_agent')

def generator_header():
    user_agent = read_random_lines(USER_AGENT_DIR, 5)[0]
    return {
            'user-agent': user_agent,
            'accept': '*/*',
            'accept-language': 'zh-CN,zh;q=0.9,en;q=0.8',
            
            # 'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
            # 'accept-encoding': 'gzip, deflate, br',
            # 'accept-language': 'zh-CN,zh;q=0.9',

            # 'Content-Type': 'text/plain;charset=UTF-8',
            # 'Connection': 'keep-alive',

            }


class LRequests(object):
    def __init__(self, string_proxy=None, headers=None, timeout=90, debuglevel=0, **kwargs):
        self.debuglevel = debuglevel
        self.timeout = timeout

        if headers is not None:
            self.headers = headers # {'key': 'value'}
        else:
            self.headers = generator_header()

        # self.session = requesocks.session(headers=self.headers, timeout=timeout)
        self.session = requests.Session()
        self.session.headers = self.headers
        self.session.mount('https://', HTTP20Adapter())
        # self.session = requests.session()

#        self.session.headers = self.headers


        if string_proxy:
            socket.getaddrinfo = getaddrinfo
            urlinfo = urlparse(string_proxy)

            if urlinfo.scheme == 'ssh':
                self.bitvise = Bitvise(urlinfo.hostname, urlinfo.port, username=urlinfo.username, password=urlinfo.password)
                forwarding_ip, forwarding_port = self.bitvise.start()

                string_proxy = 'socks5://%s:%s' % (forwarding_ip, forwarding_port)

            self.session.proxies = {'http': string_proxy, 'https': string_proxy}

        self._body = None


    def open(self, url, method='GET', data=None, timeout=socket._GLOBAL_DEFAULT_TIMEOUT, isdecode=False, repeat=3, is_xpath=True, stream=False):
        while True:
            try:

                if isinstance(url, str):
                    logger.info('Load URL: %s' % url)
                    if(method.upper() == 'POST'):
                        response = self.session.post(url=url, data=data, timeout=self.timeout, allow_redirects=True, headers=self.headers, stream=stream)
                    else:
                        response = self.session.get(url=url, data=data, timeout=self.timeout, allow_redirects=True, headers=self.headers, stream=stream)

                elif isinstance(url, urllib.request.Request):
                    logger.info('Load URL: %s' % url.get_full_url())
                    response = self.session.request(method=url.get_method(), url=url.get_full_url(), data=url.get_data(), timeout=self.timeout, allow_redirects=True, headers=self.headers, stream=stream)

                self.body = response, is_xpath, stream
                self.current_url = response.url
                return response
            except (urllib.error.HTTPError, urllib.error.URLError, httpclient.BadStatusLine, socket.timeout, socket.error, IOError, httpclient.IncompleteRead, socks.ProxyConnectionError, socks.SOCKS5Error) as e:
                repeat = repeat - 1
                if isinstance(e, urllib.error.HTTPError):
                    if e.code in NOT_REQUEST_CODE:
                        raise
                time.sleep(random.randrange(5, 10))
                if not repeat:
                    raise

            # except:
            #     repeat = repeat - 1
            #     if not repeat:
            #         raise

    def get(self, url, method='GET', data=None, timeout=socket._GLOBAL_DEFAULT_TIMEOUT, is_xpath=True, stream=False):
        return self.load(url, method=method, data=data, timeout=timeout, is_xpath=is_xpath, stream=stream)

    def load(self, url, method='GET', data=None, timeout=socket._GLOBAL_DEFAULT_TIMEOUT, is_xpath=True, stream=False):

        return self.open(url, method=method, data=data, timeout=timeout, is_xpath=is_xpath, stream=stream)

    def load_img(self, url, method='GET', data=None, timeout=socket._GLOBAL_DEFAULT_TIMEOUT, stream=True):
        if self.debuglevel:
            logger.info('Load Image: %s' % url)
        return self.open(url, method=method, data=data, timeout=timeout, is_xpath=False, stream=stream)

    def load_file(self, file_path):
        self.loads(open(file_path, 'r').read())

    def loads(self, page_str, url=''):
        self.current_url = url
        self.body = page_str, True, False

    def getForms(self, url, data=None, timeout=socket._GLOBAL_DEFAULT_TIMEOUT, isdecode=False, repeat=3, is_xpath=False):
        return self.get_forms_by_url(url, data, timeout, isdecode, repeat, is_xpath)

    # def get_forms_by_url(self, url, data=None, timeout=socket._GLOBAL_DEFAULT_TIMEOUT, isdecode=False, repeat=3, is_xpath=False):
    #     try:
    #         if timeout is socket._GLOBAL_DEFAULT_TIMEOUT:
    #             timeout = self.timeout
    #         response = None
    #         response = self.open(url, data, timeout, isdecode, repeat, is_xpath)
    #         return ParseFile(io.StringIO(str(BeautifulSoup(self.body, 'lxml')).replace('<br/>', '').replace('<hr/>', '')), response.geturl(), backwards_compat=False)
    #     except:
    #         raise
    #     finally:
    #         if response:
    #             del response

    # def get_forms(self):
    #     return ParseFile(io.StringIO(str(BeautifulSoup(self.body, 'lxml'))), self.current_url, backwards_compat=False) # .replace('<br/>', '').replace('<hr/>', '')


    @property
    def body(self):
        return self._body

    @body.setter
    def body(self, value):
        try:
            response, is_xpath, stream = value
            self._body = ''
            if stream:
                self._body = response.raw.data
            else:
                if isinstance(response.text, str):
                    self._body = response.text
                else:
                    self._body = str(response.text) # .encode('utf-8')
                if is_xpath:
                    self.tree = html.fromstring(str(BeautifulSoup(self.body, 'lxml')))
        except :
            raise

    def post(self, url, headers=None, data=None):
        self.session.post(url, data=data)

    def xpath(self, xpath):
        eles = self.tree.xpath(xpath)
        if eles and len(eles) > 0:
            return eles[0]

        return None


    def xpaths(self, xpath):
        return self.tree.xpath(xpath)


    def __del__(self):
        pass


if __name__ == '__main__':

#    l = LRequests(string_proxy='socks5://192.168.1.195:1072')
#     l = LRequests()
#     l.load('http://image.tiancity.com/article/UserFiles/Image/luoqi/2010/201009/29/3/4.jpg', is_xpath=False, stream=True)


#    print l.body

    # import shutil
    # shutil.copyfileobj(l.body, open('D:\\xxx.jpg', 'wb'))

    lr = LRequests(string_proxy='socks5://192.168.1.188:1080')
    lr.load('http://www.google.com')

    print(lr.body)

