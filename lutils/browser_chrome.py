# -*- coding: utf-8 -*-
__author__ = 'xtwxfxk'

import os
import sys
import re
import time
import json
import copy
import shutil
import socket
import tempfile
import zipfile
import logging
from urllib.parse import urlparse
import random
import traceback
import pickle
from urllib import request
import lxml
import functools
from io import BytesIO
from lxml import html
from PIL import Image
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException

from lutils import read_random_lines, LUTILS_ROOT, _clean
from lutils.bitvise import Bitvise
from lutils import free_port
from lutils import conf

logger = logging.getLogger('lutils')


def deco_sync_local(func):
    @functools.wraps(func)
    def wrapper(self, url, *args, **kwargs):
        func(self, url)
        self.sync_local()

    return wrapper

class BrowserMixin(object):


    def sync_local(self):
        self.html = self.page_source


    @property
    def html(self):
        return self._html

    @html.setter
    def html(self, source):
        self._html = _clean(source)
        self.tree = html.fromstring(self._html)

    @deco_sync_local
    def load(self, url):
        self.get(url)

    def scroll_down(self, click_num=5):
        body = self.xpath('//body')
        if body:
            for _ in range(click_num):
                body.send_keys(Keys.PAGE_DOWN)
                time.sleep(self.wait_time)
            time.sleep(self.wait_time)

    def scroll_up(self, click_num=5):
        body = self.xpath('//body')
        if body:
            for _ in range(click_num):
                body.send_keys(Keys.PAGE_UP)
                time.sleep(self.wait_time)
            time.sleep(self.wait_time)

    def fill(self, name_a, *value):
        ele = self.find_name(name_a)
        if ele:
            ele.clear()
            ele.send_keys(*value)
            time.sleep(self.wait_time)
        else: raise NoSuchElementException('%s Element Not Found' % name_a)

    # def find_ids(self, id, ignore=False):
    #     try:
    #         return self.find_elements_by_id(id)
    #     except NoSuchElementException as e:
    #         if ignore: return []
    #         else: raise NoSuchElementException(id)

    def find_id(self, id, ignore=False):
        try:
            return self.find_element_by_id(id)
        except NoSuchElementException as e:
            if ignore: return None
            else: raise NoSuchElementException(id)

    def find_names(self, name_b, ignore=False):
        try:
            return self.find_elements_by_name(name_b)
        except NoSuchElementException as e:
            if ignore: return []
            else: raise NoSuchElementException(name_b)

    def find_name(self, name, ignore=False):
        try:
            return self.find_element_by_name(name)
        except NoSuchElementException as e:
            if ignore: return None
            else: raise NoSuchElementException(name)

    def csss(self, css, ignore=False):
        try:
            return self.find_elements_by_css_selector(css)
        except NoSuchElementException as e:
            if ignore: return []
            else: raise NoSuchElementException(css)

    def css(self, css, ignore=False):
        try:
            return self.find_element_by_css_selector(css)
        except NoSuchElementException as e:
            if ignore: return None
            else: raise NoSuchElementException(css)

    def xpaths(self, xpath, ignore=False):
        try:
            return self.find_elements_by_xpath(xpath)
        except NoSuchElementException as e:
            if ignore: return []
            else: raise NoSuchElementException(xpath)

    def xpath(self, xpath, ignore=False):
        try:
            return self.find_element_by_xpath(xpath)
        except NoSuchElementException as e:
            if ignore: return None
            else: raise NoSuchElementException(xpath)

    def xpath_local(self, xpath):
        eles = self.tree.xpath(xpath)
        if eles and len(eles) > 0:
            return eles[0]
        return None

    def xpaths_local(self, xpath):
        return self.tree.xpath(xpath)

    def fill_id(self, id, *value):
        ele = self.find_id(id)
        if ele:
            ele.clear()
            ele.send_keys(*value)
            time.sleep(self.wait_time)
        else: raise NoSuchElementException('%s Element Not Found' % id)

    def wait_xpath(self, xpath):
        self.wait.until(lambda driver: driver.xpath(xpath))

    def down_until(self, xpath, stop=200, jump=20):
        _same_count = 0
        _count = 0
        while stop == -1 or (_count < stop):
            self.scroll_down()
            _c = len(self.xpaths(xpath))
            if _c == 0:
                break
            if _count ==_c:
                _same_count += 1
                time.sleep(self.wait_time)
            else:
                _count = _c
                _same_count = 0

            if _same_count > jump:
                break
            time.sleep(self.wait_time)

    def down_bottom(self):
        self.execute_script("window.scrollTo(0, document.body.scrollHeight);")

    def click_xpaths(self, xpath, num=-1):
        _eles = []
        if isinstance(xpath, str):
            _eles = self.xpaths(xpath)
        elif isinstance(xpath, list):
            for _x in xpath:
                _eles.extend(self.xpaths(xpath))

        if len(_eles) > 0:
            if num != -1:
                start = random.randrange(3)
                end = start + random.randint(1, num)
                __eles = []
                for _e in range(start, end):
                    if len(_eles) < 1: break
                    __eles.append(_eles.pop(random.randrange(len(_eles))))
                _eles = __eles
            for _e in _eles:
                _e.click()
                time.sleep(self.wait_time) # random.randrange(1000, 5555, 50)/1000.0)

    def wait_xpath(self, xpath, timeout=None):
        if time is None: timeout = self.timeout
        self.wait.until(EC.presence_of_element_located((By.XPATH, xpath)))


    def highlight_xpath(self, xpath, ignore=True):
        ele = self.xpath(xpath, ignore)
        if ele:
            ele.send_keys(Keys.NULL)
            self.highlight(ele)

    def highlight_xpaths(self, xpath, ignore=True):
        eles = self.xpaths(xpath, ignore)
        for ele in eles:
            self.highlight(ele)

    def highlight(self, element):
        driver = element._parent
        driver.execute_script("arguments[0].setAttribute('style', arguments[1]);", element, "background: yellow; border: 1px solid red;")
        element.send_keys(Keys.NULL)


    def hover(self, element):
        hov = ActionChains(self).move_to_element(element)
        hov.perform()
        time.sleep(self.wait_time)

    def save_exe(self, exe_path):
        pickle.dump({'command_executor': self.command_executor._url, 'session_id': self.session_id}, open(exe_path, 'wb'))

    def ele_xpath(self, element, xpath, ignore=False):
        try:
            return element.find_element_by_xpath(xpath)
        except NoSuchElementException as e:
            if ignore: return None
            else: raise NoSuchElementException(xpath)

    def screenshot_eles(self, xpaths):
        ims = []
        eles = self.xpaths(xpaths)
        for ele in eles:
            ims.append(Image.open(BytesIO(ele.screenshot_as_png)))
            
        return ims




class Browser(webdriver.Chrome, webdriver.Remote, BrowserMixin):

    def __init__(self, profile_dir=None, string_proxy=None, timeout=180, capabilities=None, profile_preferences={}, **kwargs):
        self.timeout = timeout + 2
        self.wait_timeout = kwargs.get('wait_timeout', self.timeout)
        self.script_timeout = kwargs.get('script_timeout', self.timeout)

        self._init_instance(profile_dir=profile_dir, string_proxy=string_proxy, timeout=timeout, capabilities=capabilities, profile_preferences=profile_preferences, **kwargs)

        self.wait_time = 0.5
        self.wait = WebDriverWait(self, self.timeout)

        self._html = ''

    def _init_instance(self, profile_dir=None, string_proxy=None, timeout=180, capabilities=None, profile_preferences={}, **kwargs):

        options = Options()
        opts.add_argument("user-data-dir=%s" % profile_dir)
        if string_proxy is not None:
            opts.add_argument('proxy-server=%s' % string_proxy)

        webdriver.Chrome.__init__(self, options=options)

        self.set_page_load_timeout(self.timeout)
        self.implicitly_wait(self.wait_timeout)
        self.set_script_timeout(self.script_timeout)



if __name__ == '__main__':
    # import time
    # profile = LFirefoxProfile(profile_directory='K:\\xx\\fff') #, is_temp=False)
    # browser = Browser(firefox_profile=profile, timeout=30)
    # browser.implicitly_wait(5)
    # browser.set_script_timeout(10)
    #
    # browser.get('http://www.baidu.com')
    #
    # e = browser.xpath('//div[@class="xxxxx"]')
    # print e
    #
    # print 'ssssssssss'
    # time.sleep(10)
    # browser.quit()


    # ######
    # b = Browser(string_proxy='socks5://127.0.0.1:1080')
    # b.execute_script('arguments[0].setAttribute("style", "color: transparent;text-shadow: #111 0 0 5px;")', ele)

    pass