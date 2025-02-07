#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 12:09:31 2024

@author: nb
"""

import sys
import os
sys.path.append(os.path.abspath(__file__+"../../../../"))
import json
from datetime import datetime, timezone
import threading
import time
import pandas as pd
from binance.um_futures import UMFutures
from multiprocessing import Manager, Process, Queue
from binance.websocket.um_futures.websocket_client import UMFuturesWebsocketClient
from binance.websocket.spot.websocket_stream import SpotWebsocketStreamClient
from binance.spot import Spot as Client
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
import cryptoqt.data.tools as tools
import copy
import random
import numpy as np
import yaml
import cryptoqt.data.data_manager as dm
import cryptoqt.data.swmrh5 as swmrh5
import cryptoqt.data.um_client_klines as um_client
import traceback
import argparse
g_log_ratio=0.001

kline_fields=['opentm', 'open', 'high', 'low', 'close','volume','closetm', 'money', 'tnum', 'tbv', 'tbm', 'ignore']  

# {
#   "e":"continuous_kline",	// 事件类型
#   "E":1607443058651,		// 事件时间
#   "ps":"BTCUSDT",			// 标的交易对
#   "ct":"PERPETUAL",			// 合约类型 
#   "k":{
#     "t":1607443020000,		// 这根K线的起始时间
#     "T":1607443079999,		// 这根K线的结束时间
#     "i":"1m",				// K线间隔
#     "f":116467658886,		// 这根K线期间第一笔更新ID
#     "L":116468012423,		// 这根K线期间末一笔更新ID
#     "o":"18787.00",			// 这根K线期间第一笔成交价
#     "c":"18804.04",			// 这根K线期间末一笔成交价
#     "h":"18804.04",			// 这根K线期间最高成交价
#     "l":"18786.54",			// 这根K线期间最低成交价
#     "v":"197.664",			// 这根K线期间成交量
#     "n":543,				// 这根K线期间成交笔数
#     "x":false,				// 这根K线是否完结(是否已经开始下一根K线)
#     "q":"3715253.19494",	// 这根K线期间成交额
#     "V":"184.769",			// 主动买入的成交量
#     "Q":"3472925.84746",	// 主动买入的成交额
#     "B":"0"					// 忽略此参数
#   }
# }
def json2df_aggtrades(msg):
    dd={}
    dd["opentm"]=np.int64(msg["t"])
    dd["open"]=np.float32(msg["o"])
    dd["high"]=np.float32(msg["h"])
    dd["low"]=np.float32(msg["l"])
    dd["close"]=np.float32(msg["c"])
    dd["volume"]=np.float32(msg["v"])
    dd["closetm"]=np.int64(msg["T"])
    dd["money"]=np.float32(msg["q"])
    dd["tnum"]=np.float32(msg["n"])
    dd["tbv"]=np.float32(msg["V"])
    dd["tbm"]=np.float32(msg["Q"])
    dd["ignore"]=np.float32(msg["B"])

    return dd

def json2df_klinerest(msg):
    dd={}
    dd["opentm"]=np.int64(msg[0])
    dd["open"]=np.float32(msg[1])
    dd["high"]=np.float32(msg[2])
    dd["low"]=np.float32(msg[3])
    dd["close"]=np.float32(msg[4])
    dd["volume"]=np.float32(msg[5])
    dd["closetm"]=np.int64(msg[6])
    dd["money"]=np.float32(msg[7])
    dd["tnum"]=np.float32(msg[8])
    dd["tbv"]=np.float32(msg[9])
    dd["tbm"]=np.float32(msg[10])
    dd["ignore"]=np.float32(msg[11])

    return dd

def selfcheck(symbol, h5f):
    # dr["min1info_tm"].tolist().index(20241128073900)
    for idx in range(h5f["opentm"].shape[0]):
        tm=h5f["opentm"][idx]
        vwap=h5f["money"][idx]/h5f["volume"][idx]
        ehFlag = vwap > h5f["high"][idx]*1.001
        llFlag = vwap < h5f["low"][idx]*0.999
        if ehFlag or llFlag:
            print(f"{symbol} ehFlag {tools.tmu2i(tm)}-{vwap} high:{h5f['high'][idx]} low:{h5f['low'][idx]}", flush=True)
            datas = fetch_and_add_data(umcs, symbol, tm, tm+60000)
            
            
    return

class WdMinKline():
    def __init__(self, symbols, path, stream_name, columns, proxy, data2df, is_combined = True, openh5fs = True):
        self.symbols=symbols
        self.path = path
        self.stream_name = stream_name
        self.columns = columns
        self.queue=Queue()
        self.proxy=proxy
        self.data2df=data2df
        self.h5fs={}
        self.is_combined = is_combined
        day = str(tools.tmu2i(int(time.time()*1000)))[:8]
        if openh5fs:
            for symbol in self.symbols:
                os.system(f'mkdir -p {self.path}/{symbol}')
                h5f=swmrh5.H5Swmr(f'{self.path}/{symbol}/{symbol}.h5', self.columns, mode="w")
                self.h5fs[symbol]=h5f
                # selfcheck(symbol, h5f.h5f)
                # lasted_items=h5f.lasted_items(1000000000)
                # if lasted_items.shape[0]>0:
                #     expectedDataLen = int((lasted_items.iloc[-1]["opentm"]-lasted_items.iloc[0]["opentm"])/60/1000+1)
                #     if expectedDataLen!=lasted_items.shape[0]:
                #         a=0
                #         # os.system(f'rm {self.path}/{symbol}/{symbol}.h5')
                #         # print(f"{symbol}-del", flush=True)
                #         for idx in range(lasted_items.shape[0]-1):
                #             tm1=lasted_items.iloc[idx]["opentm"]
                #             tm2=lasted_items.iloc[idx+1]["opentm"]
                #             if round(tm2-tm1) != 60000:
                #                 print(f"{symbol}-{tools.tmu2i(tm1)}-{tools.tmu2i(tm2)}", flush=True)
                #         raise
                
        # raise
        return
           
        
    def create_websocket_client(self):
        symbols=self.symbols
        def on_message_wrapper(_, message):
            try:
                self.queue.put(message)
                if random.random() < g_log_ratio:
                    msg = json.loads(message)
                    if "result" in msg and msg["result"] is None and "id" in msg:
                        return
                    if 'stream' not in msg or 'data' not in msg:
                        return
                    stream = msg['stream']
                    data = msg['data']
                    symbol=data['s']
                    tm=int(data['k']["T"])
                    current_time_ms = int(time.time() * 1000)
                    print(f"Network Cost {symbol}-{self.queue.qsize()}: {current_time_ms - tm} ms", flush=True)

            except Exception as e:
                print(f"Error in message_handler: {str(e)}", flush=True)
            return
    
        while True:
            client = None
            try:
                
                if g_cfg["ktype"] == 'PERPETUAL':
                    client = UMFuturesWebsocketClient(
                        on_message=on_message_wrapper,
                        #on_close=on_close_wrapper,
                        # on_error=on_error_wrapper,
                        is_combined=True,
                        proxies=self.proxy
                    )
                else:
                    client = SpotWebsocketStreamClient(
                            on_message=on_message_wrapper,
                            is_combined=True,
                            proxies=self.proxy
                        )
                    
                self.subscribe_and_write(client)

                print(f"WebSocket error with. wait Reconnecting...", 
                      threading.get_ident(), symbols, time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())), flush=True)
            except Exception as e:
                print(f"Error of own create_websocket_client: {str(e)}",
                      threading.get_ident(), symbols, time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())), flush=True)
            if not client is None:
                client.stop()
                print(f"WebSocket close. ", symbols, time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())), flush=True)
                del client
            time.sleep(10)
                
    def subscribe_and_write(self, client):
        symbols=self.symbols
        lasted_opentm = {}
        streams=[]
        for idx,symbol in enumerate(symbols):
            stream=symbol.lower()+self.stream_name
            streams.append(stream)
            lasted_items=self.h5fs[symbol].lasted_items(1000)
            if lasted_items.shape[0]>0:
                lasted_opentm[symbol]=lasted_items["opentm"].values[-1]
            else:
                opentm_zero = 20220101075900
                lasted_opentm[symbol]=tools.tmi2u(opentm_zero)
                
        client.streams=streams
        client.subscribe(streams)
        print(f"start client:", self.stream_name,
              threading.get_ident(), symbols, time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())), flush=True)

        noneCnt = 0
        while client.socket_manager.ws.connected:
            item=None
            try:
                item=self.queue.get(block=False)
            except Exception as e:
                curtm=np.int64(time.time() * 1000)
            if item is None:
                noneCnt += 1
                time.sleep(0.01)  # wait message
                if random.random() < g_log_ratio*0.1:
                    current_time_ms = int(time.time() * 1000)
                    print(f"queue item none: {tools.tmu2i(current_time_ms)} qsize:{self.queue.qsize()} connect:{client.socket_manager.ws.connected} noneCnt:{noneCnt}", 
                          flush=True)
                if noneCnt > 100*120:  #60s no data. start reconnect
                    print(f"noneCnt exceed {symbol}: {tools.tmu2i(current_time_ms)} connect:{client.socket_manager.ws.connected} noneCnt:{noneCnt}", 
                          flush=True)
                    break
                continue
            else:
                noneCnt=0
            
            msg = json.loads(item)
            if "result" in msg and msg["result"] is None and "id" in msg:
                continue
            if 'stream' not in msg or 'data' not in msg:
                continue
            stream = msg['stream']
            data = msg['data']
            symbol=data['s']     
            if not data['k']['x']:
                continue
            df=self.data2df(data['k'])

            opentm=df["opentm"]
            
            if lasted_opentm[symbol]+60000 != df['opentm']: # + 60s
                formTm = lasted_opentm[symbol]+60000
                endTm = df['opentm']
                if formTm>endTm:
                    print(f"warning formTm {symbol} formTm:{tools.tmu2i(formTm)}>endTm{tools.tmu2i(endTm)}", flush=True) 
                    continue
                datas = fetch_and_add_data(umcs, symbol, formTm, endTm)
                for data in datas:
                    day=str(tools.tmu2i(data["opentm"]))[:8]
                    self.h5fs[symbol].append_dict(data)
                    
            day=str(tools.tmu2i(opentm))[:8]
            self.h5fs[symbol].append_dict(df)
            
            lasted_opentm[symbol] = df['opentm']
            
            current_time_ms = int(time.time() * 1000)
            delay = current_time_ms - opentm - 60000
            # if random.random() < g_log_ratio:
            if delay > 1000 and random.random() < 1:
                print(f"delay warning: Cost {symbol}: {tools.tmu2i(current_time_ms)} delay:{delay} ms. qsize:{self.queue.qsize()}", flush=True)
            elif random.random() < 1:
                print(f"append Cost {symbol}: {tools.tmu2i(current_time_ms)} delay:{delay} ms. qsize:{self.queue.qsize()}", flush=True)
            # if delay > 60000:
            #     break
       
def fetch_and_add_data(umcs, symbol, formTm, endTm, limit=1000):
    num_retries=0
    all_fix_data = []
    old_formTm = formTm
    startTime = formTm
    while True:
        try:
            rest_client=umcs.get_um_client()
            if g_cfg["ktype"] == 'PERPETUAL':
                fetch_data = rest_client.client.continuous_klines(symbol, "PERPETUAL", "1m", startTime=startTime, limit=1000)
            else:
                fetch_data = rest_client.client.klines(symbol, "1m",  startTime=startTime, limit=1000)
            
            print(f"fetch_data one data{len(all_fix_data)}: old_formTm:{tools.tmu2i(old_formTm)} tendTd:{tools.tmu2i(endTm)} \
                  startTime:{tools.tmu2i(startTime)} {len(fetch_data)}-{tools.tmu2i(fetch_data[0][0])}-{tools.tmu2i(fetch_data[-1][0])} ",   
                  symbol, rest_client.proxies, flush=True)
            umcs.recycle(rest_client)
            if len(fetch_data)> 0:
                for item in fetch_data:
                    if item[0] >= startTime and item[0] < endTm:
                        all_fix_data.append(item)
                        startTime=item[0]+60000
            if startTime >= endTm:
                break
            # 20241029000000
        except Exception as ex:
            print("retry featch cnt:",  num_retries, symbol, rest_client.proxies, flush=True)
            traceback.print_exc()
            time.sleep(1)
            num_retries+=1
    datas=[]
    idx=0
    expectedOpenTm = old_formTm
    while idx < len(all_fix_data):
        item = all_fix_data[idx]
        if expectedOpenTm == item[0]:
            dd=json2df_klinerest(item)
            datas.append(dd)
            idx+=1
        else:
            itemEmpty={}
            itemEmpty["opentm"]=expectedOpenTm
            itemEmpty["open"]=np.nan
            itemEmpty["high"]=np.nan
            itemEmpty["low"]=np.nan
            itemEmpty["close"]=np.nan
            itemEmpty["volume"]=np.nan
            itemEmpty["closetm"]=expectedOpenTm+59*1000
            itemEmpty["money"]=np.nan
            itemEmpty["tnum"]=np.nan
            itemEmpty["tbv"]=np.nan
            itemEmpty["tbm"]=np.nan
            itemEmpty["ignore"]=np.nan
            datas.append(itemEmpty)
            print(f"{symbol}:add empty item:{tools.tmu2i(expectedOpenTm)}-{tools.tmu2i(item[0])}", flush=True)
        expectedOpenTm+=60*1000
    expectedDataLen = int((datas[-1]['opentm'] - datas[0]['opentm'])/60/1000+1)
    targetDataLen = int(int(endTm-old_formTm)/60/1000)
    assert expectedDataLen == len(datas) and (targetDataLen == len(datas))
    print("fetch rest data succ", symbol, formTm, "tm_dur:", tools.tmu2i(old_formTm), tools.tmu2i(datas[-1]['opentm']),  f"tendTd:{endTm}",
          len(datas), flush=True)
    return datas
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test for argparse', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--cfg', help='cfg', default="./config/wd_minkline.yaml", type=str)
    # parser.add_argument('--cfg', help='cfg', default="./config/wd_spotminkline.yaml", type=str)
    parser.add_argument('--off', help='off', default=0, type=int)
    parser.add_argument('--delta', help='delta', default=10000, type=int)
    args = parser.parse_args()
    cfgpath=args.cfg

    with open(cfgpath) as f:
        g_cfg = yaml.load(f, Loader=yaml.FullLoader)
    dm.init()
    dr=dm.dr
    if g_cfg["ktype"] == 'PERPETUAL':
        umcs=um_client.UmClient(g_cfg["um_client"], UMFutures)
    else:
        umcs=um_client.UmClient(g_cfg["um_client"], Client)

    proxyIdx = args.off % len(g_cfg["proxy"])
    print("proxy:",  g_cfg["proxy"][proxyIdx], args.off, args.delta)
    symbols=dr["sids"]
    tidx=dr["sids"].tolist().index('TLMUSDT')
    wd1=WdMinKline(
        symbols[args.off::args.delta],
        # symbols[:1],
                   g_cfg["save_path"], g_cfg["stream_name"], g_cfg["columns"], 
                    g_cfg["proxy"][proxyIdx], json2df_aggtrades, is_combined=True)
    wd1.create_websocket_client()
    
    
    
    
    
