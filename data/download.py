import sys
import os
sys.path.append(os.path.abspath(__file__+"../../../../"))
import pandas as pd
import numpy as np
import os
import pickle
from multiprocessing import Pool
import time
import multiprocessing
from functools import partial
import concurrent.futures as futures
import datetime
import itertools
import h5py
import time
import matplotlib.pyplot as plt
from multiprocessing.dummy import Pool as ThreadPool
import calendar
import hf.tools.tools as tools
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import math
import logging
from binance.um_futures import UMFutures
from binance.lib.utils import config_logging
from binance.spot import Spot as SpotClient
import traceback
import threading
import yaml
from pandarallel import pandarallel
import random
import numba
from numba import njit
from multiprocessing import Manager, Process, Queue
import hf.data.um_client as um_client
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option("display.max_colwidth", 100)
pd.set_option('display.width', 10000)


def readzip(path, sids, suffix):
    dd={}
    def loadsymboldata(symbol, suffix=suffix, dd=dd, path=path):
        fpath=path+"/"+symbol
        for fname in sorted(os.listdir(fpath)):
            if fname.endswith(suffix+".zip"):
                try:
                    df = pd.read_csv(fpath+"/"+fname, compression='zip') 
                    dd[symbol]=df
                except Exception as ex:
                    print("load zip err", fname, flush=True)
                return
    pool = ThreadPool(64)
    pool.map(loadsymboldata, sids)
                
    return dd

def downloadday(date, validsymbols, symbol):
    cmd=f'python {g_cfg["code_path"]}/download-aggTrade.py -t um -s {symbol} -d {date} -skip-monthly 1 -c 1'
    if symbol in validsymbols:
        maxcnt=10
        while maxcnt>0:
            res=os.popen(cmd)
            resinfo=res.read()
            checkstring=f'{symbol}-aggTrades-{date}.zip\n\nfile already exists'
            checkstr_notfound='File not found:'
            if checkstring in resinfo:
                print("download succ:",symbol, date, maxcnt, flush=True)
                return 1
        
            maxcnt-=1
            time.sleep(10)
            print("download error:",symbol, date, maxcnt, flush=True)
    else:
        return 1

    return 0

def wget_downloadday(date, save_path, ftype, symbol):
    if ftype == 'klines':
        fname = f'1m/{symbol}-1m-{date}.zip '
    else:
        fname = f'{symbol}-{ftype}-{date}.zip'
    cmd=f'wget https://data.binance.vision/data/futures/um/daily/{ftype}/{symbol}/{fname} -O {save_path}/{ftype}/{symbol}/{fname}'
    target_path = f'{save_path}/{ftype}/{symbol}/{fname}'
    os.system(f'mkdir -p {save_path}/{ftype}/{symbol}' + '/1m'if ftype == 'klines' else "")
    maxcnt=3
    while maxcnt>0:
        if not os.path.exists(target_path) or os.path.getsize(target_path) <= 0:
            res=os.system(cmd)
        else:
            res=0
        if res ==0:
            print("download succ:",symbol, ftype, date, maxcnt, flush=True)
            return 1
    
        maxcnt-=1
        time.sleep(2)
        print("download error:",symbol, ftype, date, maxcnt, flush=True)


    return 1

def waitdownloaday(sids, date, validsymbols):
    # wget_downloadday(date, g_cfg["save_path"], "klines", sids[0])
    pool = ThreadPool(32)
    diff=5
    # while True:
    #     ftype="aggTrades"
    #     succnts=pool.map(partial(wget_downloadday, date, g_cfg["save_path"], ftype), validsymbols)
    #     if sum(succnts) >= len(validsymbols)-diff:
    #         print(f"download all symbols {ftype} succ:", date, flush=True)
    #         break
    #     print(f"download all symbols {ftype} fail:", date, flush=True)
    # while True:
    #     ftype="bookDepth"
    #     succnts=pool.map(partial(wget_downloadday, date, g_cfg["save_path"], ftype), validsymbols)
    #     if sum(succnts) >= len(validsymbols)-diff:
    #         print(f"download all symbols {ftype} succ:", date, flush=True)
    #         break
    #     print(f"download all symbols {ftype} fail:", date, flush=True)
        
    while True:
        ftype="klines"
        succnts=pool.map(partial(wget_downloadday, date, g_cfg["save_path"], ftype), validsymbols)
        if sum(succnts) >= len(validsymbols)-diff:
            print(f"download all symbols {ftype} succ:", date, flush=True)
            break
        print(f"download all symbols {ftype} fail:", date, flush=True)
    # while True:
    #     ftype="bookTicker"
    #     succnts=pool.map(partial(wget_downloadday, date, g_cfg["save_path"], ftype), validsymbols)
    #     if sum(succnts) == len(validsymbols):
    #         print(f"download all symbols {ftype} succ:", date, flush=True)
    #         break
    #     print(f"download all symbols {ftype} fail:", date, flush=True)
    #     time.sleep(300)
    return
       
def process_alldays(endday):
    symbol_infos=pd.read_csv(tools.cpath(__file__, g_cfg["meta_path"]))
    symbols=symbol_infos["symbol"].values
    endtm=int(tools.string_toTimestamp(endday+"T08:00:00")*1000)
    tmstart=g_start_tm
    for tm in range(tmstart, endtm, 24*3600*1000):
        day=tools.tmu2s_utc(tm, "%Y-%m-%d")
        tmi=tools.tmu2i(tm)
        validsymbols=symbol_infos[symbol_infos["onboardDate"]<tmi]["symbol"].values
        
        waitdownloaday(symbols, day, validsymbols)
    return
       

def retry_fetch_ohlcv(symbol, timeframe, endTime, max_retries=1000, limit=1):
    #     [
    #   [
    #     1607444700000,    	// 开盘时间
    #     "18879.99",       	// 开盘价
    #     "18900.00",       	// 最高价
    #     "18878.98",       	// 最低价
    #     "18896.13",       	// 收盘价(当前K线未结束的即为最新价)
    #     "492.363",  		// 成交量
    #     1607444759999,   	// 收盘时间
    #     "9302145.66080",    // 成交额 idx=7
    #     1874,               // 成交笔数
    #     "385.983",    		// 主动买入成交量
    #     "7292402.33267",    // 主动买入成交额
    #     "0" 				// 请忽略该参数
    #   ]
    # ]
    um_futures_clients=um_client.UmClient(g_cfg["um_client"], UMFutures)
    num_retries = 0
    while num_retries< max_retries:
        try:
            num_retries += 1
            client=um_futures_clients.get_um_client()
            ohlcv = client.client.continuous_klines(symbol, "PERPETUAL", timeframe, limit=limit, endTime=endTime)
            um_futures_clients.recycle(client)
            if len(ohlcv) <=0 :
                print("error len ohlcv 0:", endTime, symbol, flush=True)
                continue
            return ohlcv
        except Exception as ex:
            time.sleep(1)
            print("retry_fetch_ohlcv cnt:",  num_retries, symbol, flush=True)
            traceback.print_exc()
                
def set_meta_infos(myproxies, ticksize_limit = 3.0, money_limit = 10000000.0):
    # um_futures_client = UMFutures(proxies=myproxies)
    # exchange_info=um_futures_client.exchange_info()
    # spot_futures_client = SpotClient(proxies=myproxies)
    # spot_exchange_info=spot_futures_client.exchange_info()
    # spot_symbols = [item['symbol'] for item in spot_exchange_info["symbols"] if (item["status"] == "TRADING" and item['symbol'][-4:] == "USDT")]
    # symbols=[item for item in exchange_info["symbols"] if (item['symbol'] in spot_symbols and item['symbol'][-4:] == "USDT")]
        
    # ticker_24 = um_futures_client.ticker_24hr_price_change()
    # select_symbols = []
    # for symbol_info in symbols:
    #     symbol=symbol_info["symbol"]
    #     onboardDate = int(tools.tmu2i(symbol_info["onboardDate"]))
    #     if onboardDate > 20240601080000:
    #         print(symbol,onboardDate, "skip", flush=True)
    #         continue
    #     tick_size = weightedAvgPrice = money = closeTime = None
    #     if symbol[-4:] == "USDT":
    #         day_klines = retry_fetch_ohlcv(symbol, "1d", tools.tmi2u(20241220000000), limit=150)
    #         for filter_item in symbol_info["filters"]:
    #             if filter_item["filterType"] == "PRICE_FILTER":
    #                 tick_size = float(filter_item["tickSize"])
    #         for item in ticker_24:
    #             if item["symbol"] == symbol:
    #                 weightedAvgPrice = float(item["weightedAvgPrice"])
    #                 moneys = [float(x[7]) for x in day_klines]
    #                 money=sum(moneys)/len(moneys)
    #                 closeTime = int(item["closeTime"])
    #                 break
    #         if not( tick_size is None or weightedAvgPrice is None) and closeTime > int(time.time()*1000 - 10*60*1000):
    #             print(symbol, "weightedAvgPrice:", weightedAvgPrice, "money:", money, 
    #                   "tick_size:", tick_size, tick_size/weightedAvgPrice*10000.0)
    #             select_symbols.append([symbol, onboardDate, weightedAvgPrice, money, tick_size, tick_size/weightedAvgPrice*10000.0, closeTime])
    #             # if tick_size/weightedAvgPrice*10000.0 < ticksize_limit and money > money_limit:
                    
    # df=pd.DataFrame(select_symbols, columns=["symbol", "onboardDate", "weightedAvgPrice", "money", "tick_size", "bp", "closeTime"])
    # df=df.sort_values(by="money", ascending=False)
    # df=df.reset_index().drop(columns=["index"])
    # curtm=tools.tmu2i(int(time.time()*1000))
    # df.to_csv(g_cfg["meta_path"], index=False) 
    # # 
    # df=pd.read_csv(g_cfg["meta_path"])
    # df=df[df["onboardDate"] < 20240601080000]
    # df_less=df[df["bp"]<3.0][:50]
    # # df_less=df[df["bp"]<2.5][:50]
    # df_less=df_less.reset_index().drop(columns=["index"])
    # df_less.to_csv(g_cfg["sids_less_path"], index=False) 
    
    df=pd.read_csv(g_cfg["meta_path"])
    df=df[df["onboardDate"] < 20240601080000]
    df_less=df[df["bp"]<2.0][:150]
    df_less=df_less.reset_index().drop(columns=["index"])
    df_less.to_csv(g_cfg["sids_hedge_path"], index=False) 
    return
    
if __name__ == "__main__":
    config_logging(logging, logging.WARNING)
    
    cfgpath="./config/download.yaml"
    with open(cfgpath) as f:
        g_cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    os.environ["http_proxy"]=g_cfg["myproxies"]["http"]
    os.environ["https_proxy"]=g_cfg["myproxies"]["https"]
    
    set_meta_infos(g_cfg["myproxies"])
    g_start_tm=tools.tmi2u(g_cfg["start_tm"]) 
    df=pd.read_csv(g_cfg["meta_path"])
    lastday = '2023-01-01'
    while True:
        now = datetime.datetime.now()
        day = now.strftime("%Y-%m-%d")
        hour = now.strftime("%H")
        # process_alldays('2024-11-29')
        # if day > lastday and hour > '17':
        if day > lastday and hour > '0':
            print(f"start download {day}-{hour}", flush=True)
            process_alldays(day)
            lastday=day
        else:
            print(f"sleep download {day}-{hour}", flush=True)
            time.sleep(3600)

    

        


        


