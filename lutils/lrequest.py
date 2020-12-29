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
import logging
import traceback

from http import cookiejar
from urllib.parse import urlparse
import io, urllib
from urllib import request
import http.client

#from scrapy.selector import Selector
# from lxml import html
from lxml import etree
from bs4 import BeautifulSoup
from lutils.ClientForm import ParseFile
from lutils.socksipyhandler import SocksiPyHandler, SocksiPysHandler
from lutils.bitvise import Bitvise
from lutils import read_random_lines, LUTILS_ROOT, _clean, free_port


__all__ = ['LRequest', 'LRequestCookie']

#socket.setdefaulttimeout(200)

logger = logging.getLogger('lutils')

NOT_REQUEST_CODE = [404, ]

header_path = os.path.join(LUTILS_ROOT, 'header')
USER_AGENT_DIR = os.path.join(LUTILS_ROOT, 'user_agent')

def generator_header():
    user_agent = read_random_lines(USER_AGENT_DIR, 5)[0]

    return [('User-Agent', user_agent),
    ('Accept', 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'),
    ('Accept-Language', 'en-us,en;q=0.7,zh-cn;q=0.3'),
    ('Accept-Encoding', 'gzip,deflate'),
    ('Accept-Charset', 'ISO-8859-1,utf-8;q=0.7,*;q=0.7'),
    ('Connection', 'keep-alive')]


class LCookieJar():
    def _cookie_from_cookie_tuple(self, tup, request):
        name, value, standard, rest = tup

        domain = standard.get("domain", Absent)
        path = standard.get("path", Absent)
        port = standard.get("port", Absent)
        expires = standard.get("expires", Absent)

        # set the easy defaults
        version = standard.get("version", None)
        if version is not None:
            try:
                version = int(version)
            except ValueError:
                return None  # invalid version, ignore cookie
        secure = standard.get("secure", False)
        # (discard is also set if expires is Absent)
        discard = standard.get("discard", False)
        comment = standard.get("comment", None)
        comment_url = standard.get("commenturl", None)

        # set default path
        if path is not Absent and path != "":
            path_specified = True
            path = escape_path(path)
        else:
            path_specified = False
            path = request_path(request)
            i = path.rfind("/")
            if i != -1:
                if version == 0:
                    # Netscape spec parts company from reality here
                    path = path[:i]
                else:
                    path = path[:i+1]
            if len(path) == 0: path = "/"

        # set default domain
        domain_specified = domain is not Absent
        # but first we have to remember whether it starts with a dot
        domain_initial_dot = False
        if domain_specified:
            domain_initial_dot = bool(domain.startswith("."))
        if domain is Absent:
            req_host, erhn = eff_request_host(request)
            domain = erhn
        elif not domain.startswith("."):
            domain = "."+domain

        # set default port
        port_specified = False
        if port is not Absent:
            if port is None:
                # Port attr present, but has no value: default to request port.
                # Cookie should then only be sent back on that port.
                port = request_port(request)
            else:
                port_specified = True
                port = re.sub(r"\s+", "", port)
        else:
            # No port attr present.  Cookie can be sent back on any port.
            port = None

        # set default expires and discard
        if expires is Absent:
            expires = None
            discard = True
        elif expires <= self._now:
            expires = self._now + 3600

        return Cookie(version,
            name, value,
            port, port_specified,
            domain, domain_specified, domain_initial_dot,
            path, path_specified,
            secure,
            expires,
            discard,
            comment,
            comment_url,
            rest)


class DumpCookieJar(cookiejar.CookieJar): # LCookieJar
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_cookies_lock']
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self._cookies_lock = threading.RLock()



class LMozillaCookieJar(LCookieJar, cookiejar.MozillaCookieJar):
    pass

def find_open_port(min_port=5000, max_port=10000):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    while 1:
        port = random.randint(min_port, max_port)
        try:
            s.bind(('127.0.0.1', port))
            s.close()
            del s
            return port
        except:
            time.sleep(0.5)
            port = port + random.randint(1, 300)




def getaddrinfo(*args):
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, '', (args[0], args[1]))]

class LRequest(object):
    '''
    form = lr.get_forms()[0]
    form.find_control('name').get(label='option label').selected = True
    form.find_control(id='id').get(label='option label').selected = True

    a = form.find_control('url')
    for item in form.find_control("url").items:
        print '%s' % (item.name)

    '''

    current_url = ''

    def __init__(self, string_proxy=None, request_header=None, timeout=90, delay=0, debuglevel=0, **kwargs):
        self.debuglevel = debuglevel
        if self.debuglevel > 0:
            logger.info('%s: %s %s: begin init' % (datetime.datetime.now().isoformat(), os.getpid(), id(self)))

        handers = kwargs.get('handers', [])

        self._timeout = timeout
        self._body = ''
        self._opener = None
        self.tree = None
        self.delay = delay

        if string_proxy:
            socket.getaddrinfo = getaddrinfo
            urlinfo = urlparse.urlparse(string_proxy)
            if urlinfo.scheme == 'ssh':
                # forwarding_ip = socket.gethostbyname(socket.gethostname())
                # forwarding_port = free_port()
                self.bitvise = Bitvise(urlinfo.hostname, urlinfo.port, username=urlinfo.username, password=urlinfo.password) #, forwarding_ip=forwarding_ip, forwarding_port=forwarding_port)
                forwarding_ip, forwarding_port = self.bitvise.start()

                # if self.tunnel.login(urlinfo.hostname, urlinfo.username, urlinfo.password, port=urlinfo.port, proxyport=localprot):
                socksiPyHandler = SocksiPyHandler(proxyargs=(socks.PROXY_TYPE_SOCKS5, forwarding_ip, forwarding_port, True, None, None), debuglevel=debuglevel)
                socksiPysHandler = SocksiPysHandler(proxyargs=(socks.PROXY_TYPE_SOCKS5, forwarding_ip, forwarding_port, True, None, None), debuglevel=debuglevel)
            elif urlinfo.scheme == 'socks5':
                socksiPyHandler = SocksiPyHandler(proxyargs=(socks.PROXY_TYPE_SOCKS5, urlinfo.hostname, urlinfo.port, True, urlinfo.username, urlinfo.password), debuglevel=debuglevel)
                socksiPysHandler = SocksiPysHandler(proxyargs=(socks.PROXY_TYPE_SOCKS5, urlinfo.hostname, urlinfo.port, True, urlinfo.username, urlinfo.password), debuglevel=debuglevel)
            elif urlinfo.scheme == 'socks4':
                socksiPyHandler = SocksiPyHandler(proxyargs=(socks.PROXY_TYPE_SOCKS4, urlinfo.hostname, urlinfo.port, True, urlinfo.username, urlinfo.password), debuglevel=debuglevel)
                socksiPysHandler = SocksiPysHandler(proxyargs=(socks.PROXY_TYPE_SOCKS4, urlinfo.hostname, urlinfo.port, True, urlinfo.username, urlinfo.password), debuglevel=debuglevel)
            elif urlinfo.scheme == 'http':
                socksiPyHandler = SocksiPyHandler(debuglevel=debuglevel)
                socksiPysHandler = SocksiPysHandler(debuglevel=debuglevel)
                httpProxyHandler = request.ProxyHandler({'http': string_proxy})
                handers.append(httpProxyHandler)

        else:
            socksiPyHandler = SocksiPyHandler(debuglevel=debuglevel)
            socksiPysHandler = SocksiPysHandler(debuglevel=debuglevel)

        handers.append(socksiPyHandler)
        handers.append(socksiPysHandler)

        self.init_opener(request_header, handers)

        if self.debuglevel > 0:
            logger.info('%s: %s %s: end init' % (datetime.datetime.now().isoformat(), os.getpid(), id(self)))

    # def write_log(self, l):
    #     if hasattr(self, 'f') and self.f:
    #         open(self.f, 'a').write('%s\n' % l)

    def init_opener(self, request_header=None, handers=[]):
        self.cookie = DumpCookieJar()
        handers.append(request.HTTPCookieProcessor(self.cookie))

        self._opener = request.build_opener(*handers)
        self._opener.addheaders = generator_header()

#            [('User-Agent', 'Mozilla/5.0 (Windows NT 5.1; rv:5.0) Gecko/20100101 Firefox/5.0'),
#            ('Accept', 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'),
#            ('Accept-Language', 'en-us,en;q=0.7,zh-cn;q=0.3'),
#            ('Accept-Encoding', 'gzip,deflate'),
#            ('Accept-Charset', 'ISO-8859-1,utf-8;q=0.7,*;q=0.7'),
#            ('Connection', 'keep-alive'), ]

        if os.path.exists(header_path):
            header = []
            headerFile = open(header_path, 'r').read()
            headerFile = '\n'.join([x for x in headerFile.split("\n") if x.strip() != ''])
            headerList = filter(lambda x: len(x.strip()) > 0, filter(None, headerFile.split('#')))
            headerContent = random.choice(list(headerList))
            header_lines = headerContent.splitlines()
            for header_line in header_lines:
                if len(header_line.strip()) > 0:
                    header_key, header_value = header_line.strip().split(':', 1)
                    header.append((header_key.strip(), header_value.strip()))
            self._opener.addheaders = header

        if request_header:
            self._opener.addheaders = request_header



    def open(self, url, data=None, timeout=socket._GLOBAL_DEFAULT_TIMEOUT, append_header=[], is_decode=False, repeat=3, is_xpath=True):
        try:
            self._body = ''
            if self.debuglevel > 0:
                logger.info('%s: %s %s: begin open' % (datetime.datetime.now().isoformat(), os.getpid(), id(self)))
            if timeout is socket._GLOBAL_DEFAULT_TIMEOUT:
                timeout = self._timeout
            if isinstance(url, str):
                logger.info('Load URL: %s' % url)

                # url = url.replace(' ', '%20')
                # and_reg = re.compile('&amp;')
                # while len(re.findall(and_reg, url)) > 0:
                    # url = url.replace('&amp;', '&')

            ### logger.info('Load URL: %s' % url.get_full_url())

            for header in append_header:
                self._opener.addheaders.append(header)

            if self.debuglevel > 0:
                for h in self._opener.addheaders:
                    logger.info('lutils header: %s' % (str(h)))
            while True:
                try:
                    # if self.debuglevel > 0:
                    #     logger.info('%s: %s %s: begin opener open' % (datetime.datetime.now().isoformat(), os.getpid(), id(self)))

                    response = None
                    if data:
                        response = self._opener.open(url, data=urllib.urlencode(data, doseq=True), timeout=timeout) # data: {'name': 'value'}
                    else:
                        response = self._opener.open(url, timeout=timeout)

                    self.current_url = response.geturl()
                    self.body = response, is_decode, is_xpath
                    return response
                except (request.HTTPError, request.URLError, http.client.BadStatusLine, socket.timeout, socket.error, IOError, http.client.IncompleteRead, socks.ProxyConnectionError, socks.SOCKS5Error) as e:
                    repeat = repeat - 1
                    if isinstance(e, request.HTTPError):
                        if e.code in NOT_REQUEST_CODE:
                            raise
                    time.sleep(random.randrange(10, 30))
                    if not repeat:
                        raise
                finally:
                    if self.debuglevel > 0:
                        logger.info('%s: %s %s: end opener open' % (datetime.datetime.now().isoformat(), os.getpid(), id(self)))
                    if response:
                        del response
        except :
            raise
        finally:
            if self.debuglevel > 0:
                logger.info('%s: %s %s: end open' % (datetime.datetime.now().isoformat(), os.getpid(), id(self)))

            for header in append_header:
                self._opener.addheaders.remove(header)

    def load(self, url, data=None, timeout=socket._GLOBAL_DEFAULT_TIMEOUT, append_header=[], is_decode=False, repeat=3, is_xpath=True):
        try:
            if isinstance(self.delay, int) and self.delay > 0:
                time.sleep(self.delay)
            elif isinstance(self.delay, list) and len(self.delay) == 2:
                time.sleep(random.randrange(*self.delay))

            if timeout is socket._GLOBAL_DEFAULT_TIMEOUT:
                timeout = self._timeout
            return self.open(url, data, timeout, append_header, is_decode, repeat, is_xpath)
        except :
            if self.debuglevel:
                logger.error(traceback.format_exc())
            raise

    def load_img(self, url, method='GET', data=None, timeout=socket._GLOBAL_DEFAULT_TIMEOUT,):

        logger.info('Load Image: %s' % url)
        return self.load(url, method=method, data=data, timeout=timeout, is_xpath=False)


    def load_file(self, file_path):
        '''
        Load local file

        Args:
          file_path: file path
        '''
        self.loads(open(file_path, 'r').read())

    def loads(self, page_str, url=''):
        self.current_url = url
        self.body = page_str, '', True


    def getForms(self, url, data=None, timeout=socket._GLOBAL_DEFAULT_TIMEOUT, is_decode=False, repeat=3, is_xpath=False):
        return self.get_forms_by_url(url, data, timeout, is_decode, repeat, is_xpath)

    def get_forms_by_url(self, url, data=None, timeout=socket._GLOBAL_DEFAULT_TIMEOUT, is_decode=False, repeat=3, is_xpath=False):
        try:
            if timeout is socket._GLOBAL_DEFAULT_TIMEOUT:
                timeout = self._timeout
            response = None
            response = self.open(url, data, timeout, is_decode, repeat, is_xpath)
            return ParseFile(io.StringIO(str(BeautifulSoup(self.body, 'lxml')).replace('<br/>', '').replace('<hr/>', '')), response.geturl(), backwards_compat=False)
        except:
            raise
        finally:
            if response:
                del response

    def get_forms(self):
        return ParseFile(io.StringIO(str(BeautifulSoup(self.body, 'lxml'))), self.current_url, backwards_compat=False) # .replace('<br/>', '').replace('<hr/>', '')


    @property
    def body(self):
        return self._body

    @body.setter
    def body(self, params):
        try:
            response, is_decode, is_xpath = params
            body = ''
            self._body = ''
            if isinstance(response, http.client.HTTPResponse): # urllib.addinfourl
                if response.info().get('Content-Encoding') in ('gzip', 'x-gzip'):
                    # body = gzip.GzipFile('', 'r', 0, io.BytesIO(response.read())).read() # .decode('utf-8') # todo
                    body = gzip.decompress(response.read()) # .decode('utf-8', 'ignore')
                    # body = str(body)
                else:
                    body = response.read()

                if is_decode:
                    content_type = response.info().get('Content-Type', '')
                    charset = None
                    if content_type.find('charset=') > -1:
                        charset = content_type[content_type.find('charset=') + 8:]
                    if charset:
                        self._body = body.decode(charset, 'ignore')
                    else:
                        self._body = body
                else:
                    self._body = body
            else:
                self._body = response

            # todo
            if is_xpath:
                if isinstance(self._body, bytes): # todo
                    self._body = self._body.decode('ISO-8859-1')

                self._body = _clean(self._body)
                # self.tree = Selector(text=str(BeautifulSoup(self.body, 'lxml')))
                # self.tree = html.fromstring(str(BeautifulSoup(self.body, 'lxml')))
                # self.tree = html.fromstring(self._body)
                # print(self._body)
                # self.tree = etree.fromstring(self._body)
                parser = etree.HTMLParser()
                self.tree = etree.parse(io.StringIO(self._body), parser)
        except :
            raise
        finally:
            del response
            del body

    # def delBody(self):
    #     del self._body

    # body = property(getBody, setBody, delBody, "http response text property.")

    def xpath(self, xpath):
        eles = self.tree.xpath(xpath)
        if eles and len(eles) > 0:
            return eles[0]

        return None


    def xpaths(self, xpath):
        return self.tree.xpath(xpath)

#    def css(self, css):
#        return self.tree.css(css)

    def get_ele_text(self, ele):
        return "".join(ele.itertext()).strip()

    # def _clean(self, html, remove=['br', 'hr']):
    #     self.remove = remove
    #     html = re.compile('<!--.*?-->', re.DOTALL).sub('', html) # remove comments
    #     if remove:
    #         # XXX combine tag list into single regex, if can match same at start and end
    #         for tag in remove:
    #             html = re.compile('<' + tag + '[^>]*?/>', re.DOTALL | re.IGNORECASE).sub('', html)
    #             html = re.compile('<' + tag + '[^>]*?>.*?</' + tag + '>', re.DOTALL | re.IGNORECASE).sub('', html)
    #             html = re.compile('<' + tag + '[^>]*?>', re.DOTALL | re.IGNORECASE).sub('', html)
    #     return html

    def __del__(self):
        if self._opener:
            del self._opener
        del self._body
        del self._timeout

class LRequestMozillaCookie(LRequest):

    def __init__(self, request_header=None, string_proxy=None, timeout=90, cookie_path=None, debuglevel=0, **kwargs):
        self.cookie_path = cookie_path

        super(LRequestMozillaCookie, self).__init__(request_header=request_header, string_proxy=string_proxy, timeout=timeout, debuglevel=0, **kwargs)

    def init_opener(self, request_header=None, handers=[]):
        self.cookie = LMozillaCookieJar(self.cookie_path)
        handers.append(request.HTTPCookieProcessor(self.cookie))

        self._opener = request.build_opener(*handers)
        self._opener.addheaders = generator_header()

#            [('User-Agent', 'Mozilla/5.0 (Windows NT 5.1; rv:5.0) Gecko/20100101 Firefox/5.0'),
#            ('Accept', 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'),
#            ('Accept-Language', 'en-us,en;q=0.7,zh-cn;q=0.3'),
#            ('Accept-Encoding', 'gzip,deflate'),
#            ('Accept-Charset', 'ISO-8859-1,utf-8;q=0.7,*;q=0.7'),
#            ('Connection', 'keep-alive'), ]

        if os.path.exists(header_path):
            header = []
            headerFile = open(header_path, 'r').read()
            headerFile = '\n'.join([x for x in headerFile.split("\n") if x.strip() != ''])
            headerList = filter(lambda x: len(x.strip()) > 0, filter(None, headerFile.split('#')))
            headerContent = random.choice(headerList)
            header_lines = headerContent.splitlines()
            for header_line in header_lines:
                if len(header_line.strip()) > 0:
                    header_key, header_value = header_line.strip().split(':', 1)
                    header.append((header_key.strip(), header_value.strip()))
            self._opener.addheaders = header

        if request_header:
            self._opener.addheaders = request_header

    def save_cookies(self):
        if self.cookie_path:
            self.cookie.save()

    def load_cookies(self):
        if self.cookie_path and os.path.exists(self.cookie_path):
            self.cookie.load()



LRequestCookie = LRequestMozillaCookie


if __name__ == '__main__':
    # lr = LRequestCookie(cookie_path='d:\\cc.cookies')
    lr = LRequest()
    # f = lr.getForms('http://mail.163.com/')[0]
    #
    # f['username'] = 'xx'
    # f['password'] = 'xx'
    # lr.load(f.click())
    #
    # lr.save_cookies()
    #
    # generator_header()
    #
    # lr = LRequest()
    # lr.load('http://www.baidu.com', data={'aaa': 'bbb'})
    # print lr.body

    # BITVISE_HOME = 'D:/tools/Bitvise SSH Client'
    # lr = LRequest(string_proxy='ssh://xx:xx@96.44.185.84:22')

    # lr.load('http://www.google.com')
    lr.load('http://www.baidu.com')
    # print(lr.body)

    # import conf, time
    # from bitvise import Bitvise
    # conf.BITVISE_HOME = 'D:/tools/Bitvise SSH Client'
    #
    # b = Bitvise('150.95.130.30', 22, username='xx', password='xx', forwarding_ip='192.168.1.101', forwarding_port=1081)
    # b.start()
    # print 'sssss'
    # while 1:
    #     time.sleep(1)







