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
sys.path.append('/home/crypto/')
import itertools
import h5py
import time
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
import traceback
import cryptoqt.data.updatedata as ud
import threading
import yaml
import cryptoqt.data.datammap as dmap
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option("display.max_colwidth", 100)
pd.set_option('display.width', 10000)
kline_fields=['twap', 'vwap', 'high', 'low', 'volume', 'money', 'tsv', 'tsm'] 
g_start_tm=tools.tmi2u(20240101080000) 
df = pd.read_csv('/home/crypto/binancedata/binance-public-data/python/data/futures/um/daily/trades/CRVUSDT/CRVUSDT-trades-2024-07-15.zip', compression='zip')

df2 = pd.read_csv('/home/crypto/binancedata/binance-public-data/python/data/futures/um/daily/trades/BTCUSDT/BTCUSDT-trades-2024-07-15.zip', compression='zip')

dfcrv=pd.read_csv('/home/crypto/binancedata/binance-public-data/python/data/futures/um/daily/aggTrades/CRVUSDT/CRVUSDT-aggTrades-2024-07-15.zip',
                  compression='zip')
dfbtc=pd.read_csv('/home/crypto/binancedata/binance-public-data/python/data/futures/um/daily/aggTrades/BTCUSDT/BTCUSDT-aggTrades-2024-07-15.zip',
                  compression='zip')
def convertdf(df):
    tms=df["time"]
    tms=[tools.tmu2i(x) for x in tms]
    df["time"]=np.array(tms).astype(int)
    return df
a=1