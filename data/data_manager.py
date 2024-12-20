#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 18:52:10 2024

@author: prod
"""

import sys
import os
sys.path.append(os.path.abspath(__file__+"../../../../"))
import pandas as pd
import numpy as np
np.seterr(invalid='ignore')
import cryptoqt.data.tools as tools
from cryptoqt.data.updatedata import g_data as dr
import cryptoqt.data.updatedata as ud
import cryptoqt.data.datammap as dmap
import yaml
from binance.um_futures import UMFutures
from binance.spot import Spot as Client
import cryptoqt.data.um_client_klines as umc
import cryptoqt.data.sec_klines.sec_klines as sk
import time

def read_txt(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = f.readlines()
        data=[item.strip('\n') for item in data]
    return data

def gen_symbols_flag(dr, symbols):
    flag = np.zeros(dr["sids"].shape).astype(bool)
    for symbol in symbols:
        flag[dr['sidmap'][symbol]]=True
    return flag
    
def load_ban_symbols(dr, cfg):
    dr["ban_symbols_et"]=read_txt(cfg["ban_symbols_et"])
    dr["ban_symbols_et_flag"] = gen_symbols_flag(dr, dr["ban_symbols_et"])
    
    dr["ban_symbols_at"]=read_txt(cfg["ban_symbols_at"])
    dr["ban_symbols_at_flag"] = gen_symbols_flag(dr, dr["ban_symbols_at"])
    
#根据unix时间戳获取分钟索引
def get_minidx(tm):
    return ud.gettmidx(tm)

#根据分钟索引获取unix时间戳
def get_tm_by_minidx(tmidx):
    return ud.gettm(tmidx)

#获取分钟索引最大值（最新值）
def lasted_minidx():
    return dmap.gettmsize()

#获取秒级索引最大值（最新值）
def wait_minidx(target_minidx, sleep_sec=1):
    while True:
        lasteddownload=dmap.gettmsize()
        if lasteddownload>=target_minidx:
            break
        time.sleep(sleep_sec)
    return target_minidx

#根据unix时间戳获取秒级索引
def get_secidx(tm):
    return sk.gtmidx_u(tm)

# 根据秒级索引获取unix时间戳
def get_tm_by_secidx(secidx):
    return sk.gtm_u(secidx)

#获取秒级索引最大值（最新值）
def lasted_secidx():
    return secdata.read_len()

#wait秒级索引最大值（最新值）
def wait_secidx(target_secidx, sleep_sec=0.2):
    while True:
        llen=secdata.read_len()
        if llen>=target_secidx:
            break
        time.sleep(sleep_sec)
    return target_secidx

def init():
    global secdata, secdr
    cfgpath=os.path.abspath(__file__+"../..")+"/config/dm.yaml"
    with open(cfgpath) as f:
        g_cfg = yaml.load(f, Loader=yaml.FullLoader)
        
    for key in g_cfg.keys():
        dr[key]=g_cfg[key]
        
    ####### read base data ###############
    ud.readuniverse(dr)
    load_ban_symbols(dr, g_cfg)
    lasteddownload=dmap.read_memmap(dr)
    
    # secdata=sk.SecondData(g_cfg["sec_data_path"], ud.g_data["sids"])
    # secdr=secdata.secdr
    ####### read base data ###############
    
if __name__ == "__main__":
    init()
    tm=20240910104900
    sidx=dr["sidmap"]["ANKRUSDT"]
    minidx=get_minidx(tools.tmi2u(tm))
    print("vwap", dr["min1info_vwap"][minidx, sidx], dr["min1info_money"][minidx, sidx], dr["min1info_volume"][minidx, sidx],
          "open:", dr["min1info_open"][minidx, sidx], "close:", dr["min1info_close"][minidx, sidx])
    
    tm=20240910105001
    sidx=dr["sidmap"]["BNBUSDT"]
    secidx=get_secidx(tools.tmi2u(tm))
    print("vwap", secdr["s1info_vwap"][secidx, sidx], secdr["s1info_money"][secidx, sidx], secdr["s1info_volume"][secidx, sidx])
    
    
    
    
    
    
    
    
    