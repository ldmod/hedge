#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 12:40:22 2024

@author: ld
"""
import sys
import os
sys.path.append(os.path.abspath(__file__+"../../../../../"))
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
np.seterr(invalid='ignore')
import warnings
warnings.simplefilter("ignore", category=RuntimeWarning)
import h5py
import time
import datetime
from collections import deque
import cryptoqt.data.updatedata as ud
from functools import partial
import matplotlib.pyplot as plt
import cryptoqt.data.constants as conts
import cryptoqt.data.tools as tools
import cryptoqt.data.datammap as dmap
import cryptoqt.data.sec_klines.sec_klines as sk
import cryptoqt.alpha.secfeaextractor as secfea
import cryptoqt.trade.algo.gorders_at as gorders
import cryptoqt.trade.algo.view_order_info as voi
import yaml
import random
import threading
from queue import Queue
import logging
import cryptoqt.data.data_manager as dm
import argparse
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option("display.max_colwidth", 100)
pd.set_option('display.width', 10000)
delaysec=2
longterm=30
from binance.um_futures import UMFutures
# from binance.lib.utils import config_logging
import json
from binance.websocket.um_futures.websocket_client import UMFuturesWebsocketClient
from binance.websocket.spot.websocket_stream import SpotWebsocketStreamClient
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
import cryptoqt.trade.algo.um_client_trade as um_client_trade

def create_websocket_client(cfg, queue):
    global cur_listen_key
    def on_message(_, message):
        queue.put(message)
    def on_error(ws, error):
        print("error", error, flush=True)
        
    while True:
        try:
            my_client = UMFuturesWebsocketClient(on_message=on_message, proxies=cfg["myproxies"])
            client = UMFutures(key=cfg["key"],
                               secret=cfg["secret"],
                               proxies=cfg["myproxies"])
            response = client.new_listen_key()
            cur_listen_key = response["listenKey"]
            logging.info("Listen key : {}".format(response["listenKey"]))
            my_client.user_data(listen_key=response["listenKey"], id=1,)
            
            while my_client.socket_manager.ws.connected:
                time.sleep(1.0)
            print(f"WebSocket error with. wait Reconnecting...", 
                  threading.get_ident(), time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())), flush=True)
        except Exception as e:
            print(f"Error of own create_websocket_client: {str(e)}",
                  threading.get_ident(), time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())), flush=True)
            # raise
            

def get_account_value(clients):
    retry_cnt = 50
    while retry_cnt > 0:
        try:
            client=clients.get_um_client()
            account_info = client.client.balance()
            usdt_balance = bnb_balance = usdt_crossUnPnl = 0.0
            for entry in account_info:
                if entry['asset'] == 'USDT':
                    usdt_balance = float(entry['balance'])
                    usdt_crossUnPnl = float(entry['crossUnPnl'])
                elif entry['asset'] == 'BNB':
                    bnb_balance = float(entry['balance'])
    
            depth = client.client.depth(symbol='BNBUSDT', limit=5)
            price = round(float(depth['bids'][0][0]), 2)
            clients.recycle(client)
            account_value = round(bnb_balance * price + usdt_balance + usdt_crossUnPnl, 2)
            return account_value
        except Exception as e:
            logger.error(f"get_account_value error: {str(e)}")
            retry_cnt-=1
        time.sleep(5)
    return 0
        
def update_positions(clients, curtm, position_value_map: dict):
    data=[]
    retry_cnt = 50
    while retry_cnt > 0:
        try:
            client=clients.get_um_client()
            data_dict = client.client.account()
            clients.recycle(client)
            if not data_dict:
                logger.error("data is null")
                return data
            if "positions" not in data_dict:
                logger.error("positions not exist")
                return data
    
            # update: position_value_map
            
            for position in data_dict["positions"]:
                symbol = position["symbol"]
                notional = float(position["notional"])
                positionAmt = float(position["positionAmt"])
                entryPrice = float(position["entryPrice"])
                data.append(dict(curtm=curtm, symbol=symbol, money=notional, vol=positionAmt, price=entryPrice))
                if symbol in position_value_map:
                    position_value_map[symbol] = notional
    
            return data
        except Exception as e:
            logger.error(f"update_positions: {str(e)}")
        retry_cnt-=1
        time.sleep(5)
    return data
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='trade argparse', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--cfg', help='cfg', default="./config/account.yaml", type=str)
    args = parser.parse_args()
    with open(args.cfg, 'r') as file:
        cfg = yaml.safe_load(file)
        
    dm.init()
    dr=dm.dr
    myproxies=cfg["myproxies"]
    res=Queue(maxsize=1000000)
    timeout=3000

    clients=um_client_trade.UmClient(cfg["um_client"], UMFutures)
    streams=[]

    client = UMFutures(key=cfg["key"],
                       secret=cfg["secret"],
                       proxies=myproxies)
    cur_listen_key = ""
    
    response = client.api_trading_status(recvWindow=6000)
    
    # create_websocket_client(cfg, res)
    websocket_trhead = threading.Thread(target=create_websocket_client, args=(cfg, res))
    websocket_trhead.start()
    
    cnt=0
    datas=[]
    last_sec = 0
    last_day=0
    write_delta_tm = 20
    last_position_tm = 0
    available_token_map = {}
    for sid in dr["sids"]:
        available_token_map[sid]=0
    while True:
        item = None
        try:
            item=res.get(block=False)
        except Exception as e:
            time.sleep(1.0)  # wait message
        cur_sec=int(time.time())
        
        if random.random() < 0.005:
            lkey=client.renew_listen_key(cur_listen_key)
            print(f"renew listenkey",  tools.tmu2i(int(time.time())*1000), flush=True)
            
        if cur_sec > last_position_tm + 60 and cur_sec % 60 >=5:
            curtm = tools.tmu2i((cur_sec-cur_sec%60)*1000)
            cur_day=int(curtm/1000000)
            value=get_account_value(clients)
            logger.info(f"账户净值$: {value}")
            data=update_positions(clients, curtm, available_token_map)
            df=pd.DataFrame(data, columns=["curtm", "symbol", "money", "vol", "price"])
            long_value = df["money"][df["money"]>0].sum()
            short_value = df["money"][df["money"]<0].sum()
            logger.info(f"Long仓位$: {long_value}, Short仓位$: {short_value}")
            store = tools.open_csv(cfg["pos_path"]+"_"+str(cur_day)+".h5", mode="a")
            store.append('data', df, data_columns=df.columns, min_itemsize={'symbol':64})
            store.close()
            
            df = pd.DataFrame([dict(curtm=curtm, value=float(value), long_value=float(long_value), short_value=float(short_value))])
            store = tools.open_csv(cfg["accvalue_path"]+"_"+str(cur_day)+".h5", mode="a")
            store.append('data', df, data_columns=df.columns)
            store.close()
            
            last_position_tm = cur_sec-cur_sec%60
        
        if item is None:
            continue
        
        item=json.loads(item)
        if not ('result' in item.keys() and item['result'] is None):
            if item["e"] == "ORDER_TRADE_UPDATE":
                order_id=item["o"]["c"]
                t_tm=tools.tmu2i(item["T"])
                cur_day=int(t_tm/1000000)
                if cur_sec > last_sec + write_delta_tm \
                    or  cur_day>last_day:
                    
                    cmd=f'h5clear -s {cfg["save_path"]}'
                    os.system(cmd)
                    df=pd.DataFrame(datas)
                    datas=[]
                    store = tools.open_csv(cfg["save_path"]+"_"+str(cur_day)+".h5", mode="a")

                    store.append('data', df, data_columns=df.columns, min_itemsize={'order_id':128, 'symbol':64})
                    store.close()
                    last_sec=cur_sec-cur_sec%write_delta_tm
                    last_day=cur_day
                    print(t_tm, " save succ", flush=True)
                    
                if item["o"]["X"] == "CANCELED" or item["o"]["X"] == "FILLED":
                    info=dict(order_id=order_id, 
                              # s_tm=s_tm, 
                              t_tm=t_tm,
                              buyer=(item["o"]["S"]=="BUY"), symbol=item["o"]["s"],       
                              price=float(item["o"]["p"]), qty=float(item["o"]["z"]))
                    datas.append(info)

    for client in clients:
        client.stop()
    print("stop")
    a=10
    

    
    
    
    
