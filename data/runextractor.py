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
import sys
import itertools
import h5py
from multiprocessing import Pool
import time
import pdb
import cryptoqt.data.feaextractor as feaextractor
import cryptoqt.data.sfeaextractor as sfeaextractor
import matplotlib.pyplot as plt
from multiprocessing.dummy import Pool as ThreadPool
import ccxt
import calendar
import cryptoqt.data.tools as tools
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import math
import cryptoqt.data.constants as conts
import logging
from binance.um_futures import UMFutures
from binance.lib.utils import config_logging
from binance.spot import Spot as Client
import cryptoqt.data.updatedata as ud

        
def geth5field(f, key, stockscnt):
    if key in f.keys():
        return f[key], f[key].shape[0]
    else:
        data=f.create_dataset(key, shape=(0, stockscnt), maxshape = (None,stockscnt), chunks = True)
        return data, 0

def calcfield(dr, key, ds,de):
    feafunc=feaextractor.g_feafunc
    func=feafunc[key]
    data=func(dr,ds,de)
    return data

def addextrafields(dr): 
    stockscnt=dr["sids"].shape[0]
    h5path=ud.g_h5path
    # fday=h5py.File(h5path+"dayinfo_extra.h5", 'a', libver='latest')
    # de=dr["dayinfo_close"].shape[0]
    # for key in ["dayinfo_pre_close", "dayinfo_return", "dayinfo_twap", "dayinfo_vwap",  "dayinfo_tbvr",  "dayinfo_tsv",  "dayinfo_tsm", 
    #             "dayinfo_5twap", "dayinfo_5vwap", "dayinfo_5high", "dayinfo_5low", "dayinfo_5volume", "dayinfo_5money", "dayinfo_5return", "dayinfo_5pvcorr",
    #             "dayinfo_5tnum", "dayinfo_5tbv", "dayinfo_5tbm", "dayinfo_5tbvr", "dayinfo_5tsv", "dayinfo_5tsm", 
    #             "dayinfo_10twap", "dayinfo_10vwap", "dayinfo_10high", "dayinfo_10low", "dayinfo_10volume", "dayinfo_10money", "dayinfo_10return", "dayinfo_10pvcorr",
    #             "dayinfo_10tnum", "dayinfo_10tbv", "dayinfo_10tbm", "dayinfo_10tbvr", "dayinfo_10tsv", "dayinfo_10tsm", 
    #             "dayinfo_20twap", "dayinfo_20vwap", "dayinfo_20high", "dayinfo_20low", "dayinfo_20volume", "dayinfo_20money", "dayinfo_20return", "dayinfo_20pvcorr",
    #             "dayinfo_20tnum", "dayinfo_20tbv", "dayinfo_20tbm", "dayinfo_20tbvr", "dayinfo_20tsv", "dayinfo_20tsm", 
    #             ]:
    #     fieldv, ds=geth5field(fday, key, stockscnt)
    #     if ds < de:
    #         value=calcfield(dr, key, ds,de)
    #         fieldv.resize((de, stockscnt))
    #         fieldv[ds:de]=value 
    #     dr[key]=fieldv
    #     fday.flush()
    # fday.close()
    
    f=h5py.File(h5path+"min1info_extra.h5", 'a', libver='latest')
    keys = [x for x in feaextractor.g_feafunc if x[:8]=="min1info"]
    min1e=dr["min1info_close"].shape[0]
    for key in keys:
        fieldv, min1s=geth5field(f, key, stockscnt)
        if min1s < min1e:
            value=calcfield(dr, key, min1s, min1e)
            fieldv.resize((min1e, stockscnt))
            fieldv[min1s:min1e]=value 
        dr[key]=fieldv
        f.flush()
    f.close()
    
    f=h5py.File(h5path+"smin1info_extra.h5", 'a', libver='latest')
    keys = [x for x in feaextractor.g_feafunc if x[:9]=="smin1info"]
    min1e=dr["smin1info_close"].shape[0]
    for key in keys:
        fieldv, min1s=geth5field(f, key, stockscnt)
        if min1s < min1e:
            value=calcfield(dr, key, min1s, min1e)
            fieldv.resize((min1e, stockscnt))
            fieldv[min1s:min1e]=value 
        dr[key]=fieldv
        f.flush()
    f.close()
    return

def addmin5fields(dr): 
    stockscnt=dr["sids"].shape[0]
    h5path=ud.g_h5path
    
    f=h5py.File(h5path+"min5info.h5", 'a')
    min5e=int(dr["min1info_close"].shape[0]/conts.min5mincnt)
    keys = [x for x in feaextractor.g_feafunc if x[:4]=="min5"]
    for key in keys:
        fieldv, min5s=geth5field(f, key, stockscnt)
        if min5s < min5e:
            value=calcfield(dr, key, min5s,min5e)
            fieldv.resize((min5e, stockscnt))
            fieldv[min5s:min5e]=value 
        dr[key]=fieldv
        f.flush()
    f.close()
    
    f=h5py.File(h5path+"smin5info.h5", 'a')
    min5e=int(dr["smin1info_close"].shape[0]/conts.min5mincnt)
    keys = [x for x in feaextractor.g_feafunc if x[:5]=="smin5"]
    for key in keys:
        fieldv, min5s=geth5field(f, key, stockscnt)
        if min5s < min5e:
            value=calcfield(dr, key, min5s,min5e)
            fieldv.resize((min5e, stockscnt))
            fieldv[min5s:min5e]=value 
        dr[key]=fieldv
        f.flush()
    f.close()
    return

kline_fields=['opentm', 'open', 'high', 'low', 'close','volume','closetm', 'money', 'tnum', 'tbv', 'tbm', 'ignore']  
if __name__ == "__main__":
    
    ud.readh5new(["min1info", "smin1info"], mem=True)
    addextrafields(ud.g_data)
    ud.readh5new(["min1info", "smin1info", "min1info_extra", "smin1info_extra"], mem=True)
    addmin5fields(ud.g_data)

        


        

