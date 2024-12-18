#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 14:41:29 2023

@author: dli
"""
import sys
sys.path.append('../../../')
import os
import numpy as np
import torch
import cryptoqt.data.updatedata as ud
from functools import partial
import time
# from binance.lib.utils import config_logging
from binance.websocket.um_futures.websocket_client import UMFuturesWebsocketClient
from binance.websocket.spot.websocket_stream import SpotWebsocketStreamClient
import os
import socks
import socket
import cryptoqt.data.tools as tools
import threading
from queue import Queue
import json
import logging
import cryptoqt.data.datammap as dmmap
import random
from binance.um_futures import UMFutures
# config_logging(logging, logging.DEBUG, log_file='./logs/test.log')

logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.DEBUG,
                    filename='./test.log',
                    filemode='w')
logging.info("test")

# myproxies = {
#         'http': 'http://127.0.0.1:18890',
#         'https': 'http://127.0.0.1:18890'
# }

myproxies = {
        'http': 'http://127.0.0.1:33880',
        'https': 'http://127.0.0.1:33880'
}

kline_fields=['open', 'high', 'low', 'close','volume', 'money', 'tnum', 'tbv', 'tbm'] 
fieldmap=dict(open='o', high='h', low='l', close='c', volume='v', money='q', tnum='n', tbv='V', tbm='Q')
client = UMFutures(proxies=myproxies, timeout=2)
            
if __name__ == "__main__":
    ud.readuniverse(ud.g_data)
    # sids=ud.g_data["sids"][0:1]
    #RNDRUSDT
    idx=ud.g_data["sids"].tolist().index('SOLUSDT')
    sids=ud.g_data["sids"][idx:idx+1]
    sidmap={}
    endtms={}
    for idx,sid in enumerate(sids):
        sidmap[sid]=idx

    dr=ud.g_data
    # dmmap.open_memmap_a(dr)
    client = UMFutures(
                       proxies=myproxies)
    
    client.agg_trades("BTCUSDT", startTime=1723105327085)
    res=Queue(maxsize=1000000)
    timeout=3000
    def on_message(_, message):
        res.put(message)
    def on_error(ws, error):
        logging.error("connect error")
        ws.create_ws_connection()
        ws.fp.subscribe(ws.fp.streams)
        logging.error("reconnected")
        print("error", error)
        
    clients=[]
    streams=[]
    for ii in range(0,sids.shape[0], 50):
        maxii=min(ii+50, sids.shape[0])
        streams=[]
        for idx,sid in enumerate(sids[ii:maxii]):
            stream=sid.lower()+'@aggTrade'
            streams.append(stream)
        my_client = UMFuturesWebsocketClient(on_message=on_message, 
                                            #  on_open=on_open,
                                            # on_close=on_close,
                                            on_error=on_error,
                                             proxies=myproxies)
        clients.append(my_client)
        my_client.streams=streams
        my_client.socket_manager.fp=my_client
        my_client.subscribe(streams)

    cnt=0
    datas=[]
    while True:
        item=res.get()
        # print(item)
        item=json.loads(item)
        if not ('result' in item.keys() and item['result'] is None):
            datas.append(item)
            print(tools.tmu2ms(item['T']), tools.tmu2ms(item['E']), 
                  "delta:", item['E']-item['T'], item)

    for client in clients:
        client.stop()
    print("stop")
    a=10
        
        
        
        
        










        
        
