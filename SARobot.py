# -*- coding=utf-8 -*-
import os
import sys
import re
import time
import json
import requests
import logging
import traceback
import itertools
import multiprocessing
from logging.handlers import TimedRotatingFileHandler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from lxml import etree
from datetime import datetime,timedelta,date
from operator import itemgetter, div, sub
from StockList import stock
# import pushybullet as pb
# from PyFetion import *
from PushList import Targets

RuleFolders = [u'RuleDmacrs',u'RuleDmakis',u'RuleGoldbar',u'RuleDblQaty',u'RuleTwine',u'RuleMultiArr']
Headers = {'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/40.0.2214.111 Safari/537.36'}
fromDate = '20150612'
toDate = date.today().strftime('%Y%m%d')


class StockType():
    """
    股票的类型
    """
    SH = 'SH' # 上海
    SZ = 'SZ' # 深圳
    CY = 'CY' # ？

class FigureConf():
    Font = FontProperties(fname=r"C:\Windows\WinSxS\amd64_microsoft-windows-font-truetype-simsun_31bf3856ad364e35_10.0.10240.16384_none_030990b90a5c1c08\simsun.ttc", size=14) 
    M5clr   = '#0000CC'
    M10clr  = '#FFCC00'
    M20clr  = '#CC6699'
    M30clr  = '#009966'
    DDDclr  = '#000000'
    AMAclr  = '#FF0033'
    DMAclr  = '#0066FF'
    VARclr  = '#3300FF'
    EXP1clr = '#FF00FF'
    EXP2clr = '#3300CC'

def GetIndustry(_stockid):
    """
    通过股票ID获取其所属行业Industry
    """
    data =  pd.read_csv(os.path.join(BaseDir(), r'data\all.csv'), dtype={'code':'object'}, encoding='GBK')
    industry = data.ix[data.code==_stockid,['industry']].values[0][0]
    return industry

def BaseDir():
    currentpath  = os.path.realpath(__file__)
    basedir = os.path.dirname(currentpath)
    return basedir
    
def CreateFolder():
    dailyfolder = os.path.join(BaseDir(), 'daily')
    if not os.path.exists(dailyfolder):
        os.mkdir(dailyfolder)   
    folder = datetime.today().strftime('%Y%m%d')
    folderpath = os.path.join(dailyfolder, folder)
    if not os.path.exists(folderpath):
        os.mkdir(folderpath)
    for item in RuleFolders:
        subfolder = os.path.join(folderpath, item)
        if not os.path.exists(subfolder):
            os.mkdir(subfolder)
    return folderpath,folder

def SetLogger(logger, dirStr, logName):
    if not os.path.exists(dirStr):
        os.makedirs(dirStr)
    logFileName = os.path.join(dirStr, logName)
    logHandler = TimedRotatingFileHandler(logFileName, when="midnight")
    logHandler.suffix = "%Y%m%d_%H%M%S.log"
    logFormatter = logging.Formatter('%(asctime)-12s:%(message)s')
    logHandler.setFormatter(logFormatter)
    streamHandle = logging.StreamHandler()
    streamHandle.setFormatter(logFormatter)
    logger.addHandler(logHandler)
    logger.addHandler(streamHandle)
    logger.setLevel(logging.WARNING)
    return logger

def PushwithMail(_msglist, _sendto):
    global logger
    import smtplib  
    from email.MIMEText import MIMEText  
    from email.Utils import formatdate  
    from email.Header import Header 
    smtpHost = 'smtp.qq.com'
    fromMail = username = '18982025433@qq.com'  
    password = 'stock888'
    subject  = u'[%s] 自动推荐'%datetime.today().strftime('%Y/%m/%d')
    body     = '\n'.join(_msglist) 
    mail = MIMEText(body,'plain','utf-8')  
    mail['Subject'] = Header(subject,'utf-8')  
    mail['From'] = fromMail  
    mail['To'] = _sendto
    mail['Date'] = formatdate() 
    try:  
        smtp = smtplib.SMTP_SSL(smtpHost)  
        smtp.ehlo()  
        smtp.login(username,password)
        smtp.sendmail(fromMail,_sendto,mail.as_string())  
        smtp.close()  
        logger.warning('Push to %s successfully.'%_sendto)
    except Exception as e:  
        logger.warning(str(e) + ' when pushing the msg with Mail.')
        
def CheckDate(_date):
    today = date.today()
    yesterday = today - timedelta(days=1)
    flag = _date in [today.strftime('%Y-%m-%d'),yesterday.strftime('%Y-%m-%d')]
    return flag

def ConvStrToDate(_str):
    ymd = time.strptime(_str,'%Y%m%d')
    return date(*ymd[0:3])
    
def ConvDateToStr(_date):
    return _date.strftime('%Y%m%d')
    
def GrabRealTimeStock(_stockid):
    """
    获取股票实时行情数据
    """
    url = 'http://hq.sinajs.cn/list=%s'%_stockid
    r = requests.get(url, headers = Headers)
    regx = re.compile(r'\=\"(.*)\"\;');
    m =  regx.search(r.text)
    info = m.group(0)
    infos = info.split(',')
    # Return Open/Close/High/Low/volume/Date
    return [eval(infos[1]),eval(infos[3]),eval(infos[4]),eval(infos[5]),eval(infos[8])/100,infos[30]] 
    
def GrabStock(_stockid, _begin , _end, _grabReal = False):
    Url = 'http://biz.finance.sina.com.cn/stock/flash_hq/kline_data.php?symbol=%s&end_date=%s&begin_date=%s' 
    _url = Url%(_stockid, _end, _begin)
    r = requests.get(_url, headers = Headers)
    page = etree.fromstring(r.text.encode('utf-8'))
    contents = page.xpath(u'/control/content')
    items = [[eval(content.attrib['o']),eval(content.attrib['c']),eval(content.attrib['h']),eval(content.attrib['l']),eval(content.attrib['v']),content.attrib['d']] for content in contents]
    todaydate = date.today().strftime('%Y-%m-%d')
    if todaydate != items[-1][-1] and _grabReal:
        latest = GrabRealTimeStock(_stockid)
        if latest[-1] == todaydate:
            items.append(latest)        
    return items

def StockQuery(_stockname): 
    m = re.match(r'\d{6}', _stockname)
    stockid = _stockname if m else stock.get(_stockname)
    stockname = stock.get(_stockname) if m else _stockname
    stockloc = StockType.SH if stockid[0] == '6' else StockType.SZ
    stockid = stockloc + stockid
    return stockname.decode('utf-8'), stockid   
    
def GetColumn(_array, _column):
    return [itemgetter(_column)(row) for row in _array]

def GetPart(_indices, _list):
    return [_list[idx] for idx in _indices]
    
def MovingAverage(_array, _idx, _width):
    length = len(_array)
    if length < _width:
        raise Exception("The width exceeds maximum length of stocks ")
    else:
        if type(_array[0]) == type([]):
            return [sum([itemgetter(_idx)(elem) for elem in _array[i-_width+1:i+1]])/float(_width) if i >= _width-1 else _array[i][_idx] for i in xrange(length)]
        else: 
            return [sum( _array[i-_width+1:i+1] if i >= _width-1 else (_array[0:i]+[_array[i]]*(_width-i)))/float(_width) for i in xrange(length)]
    
def CalcExpMA(_list, _period):
    length = len(_list)
    def ExpMA(_list, n, N):
        return _list[0] if n == 0 else (_list[n]*2.0 + (N - 1.0)*ExpMA(_list, n-1,N))/( N +1.0)
    ExpMA = [ExpMA(_list,i,_period) for i in xrange(length)]
    return ExpMA
        
def RisingPercent(_array):
    length = len(_array)
    return [100.0*(itemgetter(1)(_array[i]) - itemgetter(1)(_array[i-1]))/itemgetter(1)(_array[i-1]) if i >=1 else 0 for i in xrange(length)]
    
def CalcVar(_array):
    varvalue = [np.var(GetColumn(_array, i)) for i in xrange(len(_array[0]))]
    return varvalue
    
def NormVol(_list):
    return [10.0*elem/max(_list) for elem in _list]

def FindZero(_list):
    L_S1 = _list[1:]+[_list[-1]]
    npList = np.array(_list)
    npL_S1 = np.array(L_S1)
    multi = npList*npL_S1
    indices = list((multi < 0).nonzero()[0])
    indices = [index if abs(_list[index])<abs(_list[index+1]) else index+1 for index in indices]
    return indices
    
def FindClose(_list):
    eps = 0.02
    npList = np.array(_list)
    indices = list((abs(npList)<eps).nonzero()[0])
    return indices
    
def CalcDiff(_list):
    L1 = _list[1:]+[_list[-1]]
    L2 = [_list[0]]+_list[0:-1]
    Diff = list(np.array(map(sub, L1, L2))*5.0)
    return Diff

def CalcInteg(_list):
    return [sum(_list[0:i+1]) for i,elem in enumerate(_list)]
    
def GetStockList():     
    return [id for id, sname in stock.items() if re.match(r'\d{6}', id)]

def CalcMA(_array):
    MA1 = MovingAverage(_array, 1, 1)
    MA5 = MovingAverage(_array, 1, 5)
    MA10 = MovingAverage(_array, 1, 10)
    MA20 = MovingAverage(_array, 1, 20)
    MA30 = MovingAverage(_array, 1, 30)
    VAR = CalcVar( [MA1,MA5, MA10, MA20, MA30] )    
    MACluster = {'MA1':MA1, 'MA5':MA5, 'MA10':MA10, 'MA20':MA20, 'MA30':MA30, 'VAR':VAR}
    return MACluster

def CalcDMA(_close, Short = 5, Long = 89, Middle = 34):
    DDD = map(sub, MovingAverage(_close, 0, Short) , MovingAverage(_close, 0, Long))
    AMA = MovingAverage(DDD, 0, Middle)
    DMA = map(sub, DDD , AMA)
    # DMACluster = {'DMA':DMA, 'AMA':AMA, 'DIF':DIF}
    return DDD, AMA, DMA

def RuleGoldBar(_prices, _volumes, _date, _check = True):
    RecentP = _prices[-5:]
    RecentV = _volumes[-5:]
    C0 = CheckDate(_date) if _check else True
    C1 = RecentP[4]>RecentP[3]>RecentP[2]>RecentP[1]
    C2 = RecentV[4]<RecentV[3]<RecentV[2]<RecentV[1]
    C3 = (RecentP[1]-RecentP[0])/RecentP[0]>0.09
    Rule = False not in [C0,C1,C2,C3]
    return Rule
    
def RuleGoldCross(DDD, AMA, _zeroNdxs, _lastNdx, _date, _check = True):
    DMA = map(sub, DDD , AMA)
    DIFF = CalcDiff(DMA)
    C0 = CheckDate(_date) if _check else True
    C1 = DMA[_lastNdx]>0
    C2 = sum(DMA[_zeroNdxs[0]:_zeroNdxs[1]])>0
    C3 = sum(DMA[_zeroNdxs[1]:_zeroNdxs[2]])<=0
    C4 = sum(DMA[_zeroNdxs[0]:_zeroNdxs[2]])>0
    C5 = _lastNdx - _zeroNdxs[2] < 2
    C6 = ((_zeroNdxs[1] - _zeroNdxs[0]) - 2*(_zeroNdxs[2] - _zeroNdxs[1]))>0
    C7 = (_zeroNdxs[2] - _zeroNdxs[1]) < 8
    C8 = AMA[_zeroNdxs[2]] - AMA[_zeroNdxs[1]] >= 0 or AMA[_zeroNdxs[2]] - AMA[_zeroNdxs[0]] > 0
    Rule = False not in [C0,C1,C2,C3,C4,C5,C6,C7,C8]
    return Rule 
    
def RuleGoldKiss(DDD, AMA, _zeroNdx, Close, _lastNdx, _date, _check = True):
    DMA = map(sub, DDD , AMA)
    DIFF = CalcDiff(DMA)
    AMADIFF = CalcDiff(AMA)
    DFZeros = FindZero(DIFF)
    DMAAfterZero = DMA[_zeroNdx:]
    MaxDMA, MaxIndx = max( (v, i) for i, v in enumerate(DMAAfterZero) )     
    C0 = CheckDate(_date) if _check else True
    C1 = 0<DMA[_lastNdx]<0.15*Close[_lastNdx] # Last day DMA Less than Close_price*1.5%
    C2 = 0<DMA[DFZeros[-1]]<0.1*Close[DFZeros[-1]] # Kiss day DMA Less than Close_price*10%
    C3 = MaxDMA > 0.03*Close[_zeroNdx+MaxIndx]
    C4 = 5<=(_lastNdx - _zeroNdx)<=120 and (_lastNdx - DFZeros[-1])<=1 # Last DMA Cross day within 9 weeks, Kiss day within 1 week
    C5 = DIFF[_zeroNdx] > 0
    C6 = DIFF[_lastNdx] >= 0
    C7 = AMADIFF[_lastNdx] > 0 or AMA[DFZeros[-1]]-AMA[_zeroNdx] >0
    C8 = sum(DMA[_zeroNdx:]) > 0
    Rule = False not in [C0,C1,C2,C3,C4,C5,C6,C7,C8]    
    return Rule 

def RuleGoldTwine(DDD, AMA, Close, _date, _check=True):
    DMA = map(sub, DDD, AMA)
    Recent = DMA[-10:]
    threshold = 0.02
    C0 = CheckDate(_date) if _check else True
    C1 = False not in [abs(item) < threshold*price for item, price in zip(Recent,Close)]
    Rule = False not in [C0, C1]
    return Rule
    
def RuleGoldWave(DDD, AMA, _zeroNdx, Close, _lastNdx, _date, _check = True):
    pass    

def RuleDoubleQuantity(_prices, _volumes, _date, _check = True):
    RecentP = _prices[-4:]
    RecentV = _volumes[-33:]
    MeanV = np.mean(RecentV[0:30])
    MaxIndx = max( (v, i) for i, v in enumerate(RecentV) )[1]   
    C0 = CheckDate(_date) if _check else True
    C1 = False not in [ R>2.0*MeanV for R in RecentV[-3:]]
    C2 = MaxIndx >= 30
    C3 = RecentP[1] > RecentP[0] and RecentP[3] > RecentP[1]
    C4 = 0.07<= (RecentP[3] - RecentP[0])/RecentP[0] <=0.1
    Rule = False not in [C0,C1,C2,C3,C4]    
    return Rule 
    
def RuleEXPMA(_list, _lastNdx, _date):
    EXP1 = CalcExpMA(_list,10)
    EXP2 = CalcExpMA(_list,50)
    DIFEXP = map(sub, EXP1, EXP2)
    EXPZeros = FindZero(DIFEXP) 
    C0 = CheckDate(_date)
    C1 = DIFEXP[_lastNdx]>0
    C2 = (_lastNdx - EXPZeros[-1])<5
    Rule = False not in [C0,C1,C2]
    return Rule

def RuleMultiArrange(_close, _date):
    MA5  = MovingAverage(_close, 0, 5 )
    MA13 = MovingAverage(_close, 0, 13)
    MA21 = MovingAverage(_close, 0, 21)
    MA34 = MovingAverage(_close, 0, 34)
    MA55 = MovingAverage(_close, 0, 55)
    C0 = CheckDate(_date)
    C1 = MA5[-1] > MA13[-1] > MA21[-1] > MA34[-1] > MA55[-1]
    C2 = MA5[-2] > MA13[-2] > MA21[-2] > MA34[-2] > MA55[-2]
    C3 = MA5[-3] > MA13[-3] > MA21[-3] > MA34[-3] > MA55[-3]
    Rule = False not in [C0, C1,not C2, not C3]
    return Rule 
    
def Rule135(_close, _date):
    MA13 = MovingAverage(_close, 0, 13)
    MA34 = MovingAverage(_close, 0, 34)
    MA55 = MovingAverage(_close, 0, 55)
    DIFF = CalcDiff(MA13)
    C0 = CheckDate(_date)
    C1 = MA55[-1]>MA34[-1]>MA13[-1]
    C2 = DIFF[-1] >= 0
    C3 = _close[-1]>MA13[-1]
    Rule = False not in [C0,C1,C2,C3]
    return Rule

def CalcBoll(Close,N=89, k=2):
    # Bollinger Bands consist of:
    # an N-period moving average (MA)
    # an upper band at K times an N-period standard deviation above the moving average (MA + Kσ)
    # a lower band at K times an N-period standard deviation below the moving average (MA − Kσ)
    # %b = (last − lowerBB) / (upperBB − lowerBB)
    # Bandwidth tells how wide the Bollinger Bands are on a normalized basis. Writing the same symbols as before, and middleBB for the moving average, or middle Bollinger Band:
    # Bandwidth = (upperBB − lowerBB) / middleBB
    length = len(Close)
    MA = MovingAverage(Close,0,N)
    # MA = CalcExpMA(Close, N)
    SM = map(lambda x,y:(x-y)**2, Close, MA)
    MD = [(sum(SM[i-N+1:i+1] if i >= N-1 else (SM[0:i]+[SM[i]]*(N-i)))/float(N))**0.5 for i in xrange(length)]
    UP = map(lambda x,y:x+y*k, MA, MD)
    DN = map(lambda x,y:x-y*k, MA, MD)
    b = map(lambda x,y,z:(x-z)/(float(y-z) if y!=z else 1.0), Close, UP, DN)
    Band = map(lambda x,y,z:(x-z)/(float(y) if y!=0 else 1.0), UP, MD, DN)
    return MA, UP, DN, b, Band

def GrabHFQPrice(_stockid):
    hfqUrl = 'http://vip.stock.finance.sina.com.cn/api/json.php/BasicStockSrv.getStockFuQuanData'
    payload = {'symbol': _stockid,'type': 'hfq'}
    r = requests.get(hfqUrl, headers = Headers, params = payload)
    text = r.text[1:-1]
    text = text.replace('{_', '{"')
    text = text.replace('total', '"total"')
    text = text.replace('data', '"data"')
    text = text.replace(':"', '":"')
    text = text.replace('",_', '","')
    text = text.replace('_', '-')
    jdata = json.loads(text, encoding = 'utf-8')
    return jdata['data']

def GenerateFigure(_open, _close, _items):
    # GenerateFigure(Open, Close, items)
    #==================Ignore Figure==========#
    Percent = RisingPercent(items)
    Rise = map(sub, Close , Open)
    rise_index = [i for i,per in enumerate(Rise) if per>=0]
    fall_index = [i for i,per in enumerate(Rise) if per<0]

    step = 5
    lookback = 55
    id_start = idx[-1]-lookback if idx[-1]>lookback else idx[0]
    plt.subplot(3, 1, 1)            
    
    # Draw K-fig 
    rise_index = [i for i,per in enumerate(Rise) if per>=0]
    fall_index = [i for i,per in enumerate(Rise) if per<0]
    plt.vlines(rise_index, GetPart(rise_index,Low), GetPart(rise_index,High), edgecolor='red', linewidth=1, label='_nolegend_') 
    plt.vlines(rise_index, GetPart(rise_index,Open), GetPart(rise_index,Close), edgecolor='red', linewidth=4, label='_nolegend_')
    plt.vlines(fall_index, GetPart(fall_index,Low), GetPart(fall_index,High), edgecolor='green', linewidth=1, label='_nolegend_') 
    plt.vlines(fall_index, GetPart(fall_index,Open), GetPart(fall_index,Close), edgecolor='green', linewidth=4, label='_nolegend_') 
    plt.title(stockname, fontproperties = FigureConf.Font)  
    
    plt.grid(True, 'major', color='0.3', linestyle='solid', linewidth=0.2)      
    ax = plt.gca()      
    ax.autoscale(enable=True, axis='both', tight=True)
    ax.set_xticklabels( emp[0::step], rotation=75, fontsize='small')
    ax.set_xlim([id_start,idx[-1]])             
    ax.set_ylim(min(Close[id_start:]), max(Close[id_start:]))
    
    plt.subplot(3, 1, 2)
    plt.stem(idx, MACluster['VAR'], linefmt=VARclr, markerfmt=" ", basefmt=" ")
    plt.plot(idx,DDD, DDDclr, AMA, AMAclr ,DMA, DMAclr)
    plt.plot(zero_ndx[-3:], zero_pts[-3:], 'ro')            
    plt.grid(True, 'major', color='0.3', linestyle='solid', linewidth=0.2)              
    ax = plt.gca()
    ax.autoscale(enable=True, axis='both', tight=True)          
    ax.set_xticklabels( emp[0::step], rotation=75, fontsize='small')
    ax.set_xlim([id_start,idx[-1]])             
    ax.set_ylim(min(DMA[id_start:] + AMA[id_start:] + DDD[id_start:]),\
    max(DMA[id_start:] + AMA[id_start:]+ DDD[id_start:]))
    
    plt.subplot(3, 1, 3)
    plt.bar(rise_index, GetPart(rise_index,Vol),bottom=-20,color='r',edgecolor='r',align="center")
    plt.bar(fall_index, GetPart(fall_index,Vol),bottom=-20,color='g',edgecolor='g',align="center")              
    plt.grid(True, 'major', color='0.3', linestyle='solid', linewidth=0.2)      
    plt.xticks(np.arange(len(idx))[0::step], emp[0::step])
    ax = plt.gca()  
    ax.autoscale(enable=True, axis='both', tight=True)
    ax.set_xticklabels(datex[0::step], rotation=75, fontsize='small')
    ax.set_xlim([id_start,idx[-1]])             
    # # plt.show()
    try:
        plt.savefig('%s/%s/%s%s.png'%(baseFolder,RuleFolder,stockid+stockname,datex[zero_ndx[-1]]), dpi=100)
    except:
        plt.savefig('%s/%s/%s%s.png'%(baseFolder,RuleFolder,stockid+stockname[1:],datex[zero_ndx[-1]]), dpi=100)
    plt.clf()

def GoldSeeker(_id, _fromDate, _toDate, _num, _figure = False):
    global logger   
    Result = ()
    try:
        stockname, stockid = StockQuery(_id)
        items = GrabStock(stockid, _fromDate, _toDate)  
        datex = GetColumn(items, 5)
        if not CheckDate(datex[-1]):
            logger.warning('Suspension%4s:%4s:%s'%(_num, stockname+(4-len(stockname))*'  ', stockid))
            return ()
        HfqPrice = GrabHFQPrice(_id)            
        length = len(items)
        idx = xrange(length)
        emp = ['']*length
        Open = GetColumn(items, 0)          
        Close = GetColumn(items, 1)
        HfqClose = map(lambda d: float(HfqPrice[d]), datex)
        [DDD, AMA, DMA] = CalcDMA(HfqClose)     
        zero_ndx = FindZero(DMA)
        # zero_pts = GetPart(zero_ndx, DMA)         
        
        High = GetColumn(items, 2)
        Low = GetColumn(items, 3)
        Volumes = GetColumn(items, 4)
        Cross = RuleGoldCross(DDD, AMA, zero_ndx[-3:], idx[-1], datex[-1])
        Kiss = RuleGoldKiss(DDD, AMA, zero_ndx[-1], HfqClose, idx[-1], datex[-1])       
        GoldBar = RuleGoldBar(HfqClose, Volumes, datex[-1])
        DbleQuty = RuleDoubleQuantity(HfqClose, Volumes, datex[-1])
        Twine = RuleGoldTwine(DDD, AMA, HfqClose, datex[-1])
        MultiArr = RuleMultiArrange(HfqClose, datex[-1])
        category = ['']
        for ndx,item in enumerate([Cross, Kiss, GoldBar, DbleQuty, Twine, MultiArr]):
            if item:
                RuleFolder = RuleFolders[ndx]
                if _figure:
                    GenerateFigure(Open, Close, items)
                goldstock = '{0:8}- {1} {2}({3})'.format(RuleFolder[4:],stockid[2:],stockname.encode('utf-8'),GetIndustry(stockid[2:]).encode('utf-8'))
                category.append(RuleFolder[4:])
                Result = (goldstock,AMA[-1])
        logger.warning('Complete%6s:%4s:%s %s'%(_num, stockname+(4-len(stockname))*'  ', stockid, '@'.join(category)))
    except Exception as e:
        logger.error('Error: %s\n%s'%(e, traceback.format_exc()))
    return Result

def SortList(_tupleList):
    OrderedResult = sorted(_tupleList, key=itemgetter(1))
    AMAOrderedResult = map(itemgetter(0), OrderedResult)
    OrderedResult = sorted(AMAOrderedResult,key=lambda x:x[0:4])
    return OrderedResult

def PushStocks(_stockList, _targets):
    if _stockList:
        for target in _targets:
            if   target['type'] == 'A':
                toPush = _stockList
                # PushwithPb(toPush,folder)
            elif target['type'] == 'D':
                toPush = [item for item in _stockList if item[0]=='D']
            elif target['type'] == 'm':
                toPush = [item for item in _stockList if item[1]=='m']
            else:
                toPush = [':( Sorry. Keep you money in your pocket safely. No stocks to push today.']       
            PushwithMail(toPush, target['mail'])
            # PushwithFetion(toPush, target['phone'])
            time.sleep(2)
    pass

def ClassifyStocks(stocks):
    SHstocks = [item for item in stocks if item[0]=='6']
    SZstocks = [item for item in stocks if item[0]=='0']
    CYstocks = [item for item in stocks if item[0]=='3']
    StockField = [StockType.SH, StockType.SZ, StockType.CY]
    return dict(zip(StockField, [SHstocks, SZstocks, CYstocks]))

def AsyncGrab( stocks ):
    result = []
    pool = multiprocessing.Pool(processes = 4)
    for ndx,stock in enumerate(stocks):
        result.append(pool.apply_async(GoldSeeker, (stock, fromDate, toDate, ndx,)))
    pool.close()
    pool.join()
    return [res.get() for res in result if res.get()]

def MapGrab( stocks ):
    def GoldSeekerWrapper(zipitems):
        return GoldSeeker(*zipitems)
    pool = multiprocessing.Pool(processes = 4)
    indices = xrange(1, len(stocks))
    result = pool.map(GoldSeekerWrapper, itertools.izip(stocks, itertools.repeat(fromDate), itertools.repeat(toDate), indices))
    pool.close()
    pool.join()
    return [res for res in result if res]

logTime = datetime.now().strftime('%Y%m%d_%H%M')
logFile = 'StockData_%s.log'%logTime
logDir = os.path.join(BaseDir(), "Logs")
logger = logging.getLogger('spider_stock')
logger = SetLogger(logger, logDir, logFile) 
    
if __name__ == '__main__':
    baseFolder, folder = CreateFolder()
    stocks = GetStockList()
    if len(sys.argv) > 1 and sys.argv[1] in [StockType.SH, StockType.SZ, StockType.CY]:
        classified = ClassifyStocks(stocks)
        stocks = classified[sys.argv[1]]
    finalResults = AsyncGrab(stocks)
    OrderedResult = SortList(finalResults)
    PushStocks(OrderedResult, Targets)
