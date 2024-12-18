import sys
import os
sys.path.append(os.path.abspath(__file__+"../../../../../"))
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
from pandarallel import pandarallel
import cryptoqt.data.sec_klines.sec_trades as sec_trades
import random
import numba
from numba import njit
from multiprocessing import Manager, Process, Queue
import cryptoqt.data.sec_klines.um_client as um_client
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option("display.max_colwidth", 100)
pd.set_option('display.width', 10000)
kline_fields=['s1info_twap', 's1info_vwap', 's1info_volume', 's1info_money', 's1info_tn',
              's1info_svwap', 's1info_svolume', 's1info_smoney','s1info_tsn', 's1info_shigh', 's1info_slow','s1info_spstd',
              's1info_bvwap',  
              # 's1info_bvolume', 's1info_bmoney', 's1info_tbn', 
              's1info_bhigh', 's1info_blow','s1info_bpstd',
              'valid'
              ] 

g_start_tm=tools.tmi2u(20240101080000) 
g_cfg=None
class SecondData(object):
    def __init__(self, path, symbols, keys=None, mode="r"):
        self.__datas={}
        if keys is None:
            self.keys=kline_fields
        else:
            self.keys=keys
        self.path=path
        self.symbols=symbols
        self.mode=mode
        self.caches=Queue(maxsize=10000)
        self.loaddata()
        return
    def loaddata(self):
        for key in self.keys:
            self.opendata(key)
        
    @property   
    def secdr(self):
        return self.__datas
    
    def opendata(self, key):
        size=int(365*1.2*1440*60)
        if os.path.exists(self.path+'/'+key+'.npy'):
            self.__datas[key]=np.memmap(self.path+'/'+key+'.npy', dtype=np.float32, mode=self.mode, shape=(size,len(self.symbols)))
            # self.__datas[key].resize((size,len(self.symbols)))
        else:
            self.__datas[key]=np.memmap(self.path+'/'+key+'.npy', dtype=np.float32, mode='w+', shape=(size,len(self.symbols)))
        # self.__datas[key].fill(np.nan)
    def getdata(self, key):
        return self.__datas[key]
    def append_data(self, tmpdd, s, e):
        length=self.read_len()
        assert s==length, "appenderror:"+str((s,e))+"   "+str(length)
        for key in self.keys:
            self.__datas[key][s:e,:]=tmpdd[key]
            self.__datas[key].flush()
        self.write_len(e)
        
    def append_data_inner(self, s):
        length=self.read_len()
        assert s==length, "update error:"+str((s,))+"   "+str(length)
        def update_secdata(dd, tidx):
            nanflag=(~np.isfinite(dd["s1info_vwap"][tidx, :]) | (dd["s1info_vwap"][tidx, :] <=0))
            dd["s1info_twap"][tidx, nanflag]=dd["s1info_vwap"][tidx-1, nanflag]
            dd["s1info_vwap"][tidx, nanflag]=dd["s1info_vwap"][tidx-1, nanflag]
            
            nanflag=(~np.isfinite(dd["s1info_bvwap"][tidx, :]) | (dd["s1info_bvwap"][tidx, :] <=0))
            dd["s1info_bvwap"][tidx, nanflag]=dd["s1info_bvwap"][tidx-1, nanflag]
            dd["s1info_bhigh"][tidx, nanflag]=dd["s1info_bvwap"][tidx-1, nanflag]
            dd["s1info_blow"][tidx, nanflag]=dd["s1info_bvwap"][tidx-1, nanflag]
            
            nanflag=(~np.isfinite(dd["s1info_svwap"][tidx, :]) | (dd["s1info_svwap"][tidx, :] <=0))
            dd["s1info_svwap"][tidx, nanflag]=dd["s1info_svwap"][tidx-1, nanflag]  
            dd["s1info_shigh"][tidx, nanflag]=dd["s1info_vwap"][tidx-1, nanflag]
            dd["s1info_slow"][tidx, nanflag]=dd["s1info_vwap"][tidx-1, nanflag]       
            return
        update_secdata(self.__datas, s)
        self.write_len(s+1)

    def write_data(self, secidx, sidx, keys, dd):
        for key in keys:
            self.__datas[key][secidx, sidx]=dd[key]
        self.__datas['valid'][secidx, sidx]=1
        
    def set_valid(self, s, e, value=0):
        # isfiniteflag=np.isfinite(self.__datas['s1info_vwap'][:s,:])
        # self.__datas['valid'][:s, :][isfiniteflag]=1
        self.__datas['valid'][s:e, :]=value
        self.__datas['valid'].flush()
        
    def set_valid_zero(self, idx, s, e, value=0):
        self.__datas['valid'][s:e, idx]=value
        self.__datas['valid'].flush()
        
    def flush(self):
        for key in self.keys:
            self.__datas[key].flush()

    def write_data_f(self, secidx, sidx, keys, dd):
        self.caches.put((secidx, sidx, keys, dd))
        if self.caches.qsize()>100:
            while not self.caches.empty():
                secidx, sidx, keys, dd=self.caches.get()
                for key in keys:
                    self.__datas[key][secidx, sidx]=dd[key]
                    # self.__datas[key].flush()
                self.__datas['valid'][secidx, sidx]=1
            for key in self.keys:
                self.__datas[key].flush()
        
    def write_len(self, length):
        f=tools.writeh5(self.path+"/rw_lock.h5", 0.002)
        np.save(self.path+"_shapeinfos", np.array([length], dtype=object))
        f.close()
        return f
    def read_len(self):
        if os.path.exists(self.path+"/rw_lock.h5"):
            f=tools.readh5(self.path+"/rw_lock.h5", 0.002)
            shapeinfos=np.load(self.path+"_shapeinfos.npy", allow_pickle=True)
            length=shapeinfos[0]
            f.close()
        else:
            length=0
        return length
 
def gtmidx_i(tm):
    return int((tools.tmi2u(tm)-g_start_tm)/1000)
def gtmidx_u(tm):
    return int((tm-g_start_tm)/1000)
def gtm_u(idx):
    return g_start_tm+idx*1000
def gtm_i(idx):
    return tools.tmu2i(g_start_tm+idx*1000)
def gtm_ms(idx):
    return tools.tmu2ms(g_start_tm+idx*1000)

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

def calcphis(dfbnew, dfsnew):
    pandarallel.initialize()
    def calcprice(x, columns=[]):
        xbak=x
        x=x.copy().sort_values(by='price')
        vtmp,mtmp=0,0
        split=len(columns)
        vlimit=1/split*x["volume"].sum()
        vwaps, vols, moneys=[], [0 for x in range(split)], [0 for x in range(split)]
        xi=0
        for idx in range(split):
            while xi < x.shape[0] and vols[idx]+x.iloc[xi]["volume"] < vlimit:
                vols[idx]+=x["volume"].iloc[xi]
                moneys[idx]+=x["money"].iloc[xi]
                xi+=1
            if xi>=x.shape[0]:
                break
            vdelta=vlimit-vols[idx]
            mdelta=(vlimit-vols[idx])/x["volume"].iloc[xi]*x["money"].iloc[xi]
            vols[idx]+=vdelta
            moneys[idx]+=mdelta
            x["volume"].iloc[xi]-=vdelta
            x["money"].iloc[xi]-=mdelta
        vwaps=[moneys[idx]/vols[idx] for idx in range(split)]
        res=pd.DataFrame([vwaps], columns=columns)
        return res
    
    pbdata=dfbnew.parallel_apply(partial(calcprice, columns=["s1info_bp20", "s1info_bp40", "s1info_bp60", "s1info_bp80", "s1info_bp100"]))
   
    pbdata=dfbnew.apply(partial(calcprice, columns=["s1info_bp20", "s1info_bp40", "s1info_bp60", "s1info_bp80", "s1info_bp100"]))
    psdata=dfsnew.apply(partial(calcprice, columns=["s1info_sp20", "s1info_sp40", "s1info_sp60", "s1info_sp80", "s1info_sp100"]))
   
def processdf(df):
    dfnew=pd.DataFrame()
    dfnew["secidx"]=((df["transact_time"]-g_start_tm)/1000).astype(int)
    dfnew["money"]=df["price"]*df["quantity"]
    dfnew["volume"]=df["quantity"]
    dfnew["price"]=df["price"]
    dfnew["is_buyer_maker"]=(df["is_buyer_maker"])
    dfnew["is_buyer_taker"]=(~df["is_buyer_maker"])

    dfsnew=dfnew[dfnew["is_buyer_maker"]].groupby("secidx")
    dfbnew=dfnew[~dfnew["is_buyer_maker"]].groupby("secidx")
    dfnew=dfnew.groupby("secidx")
    dfagg=pd.DataFrame()
    # dfagg["secidx"]=dfnew["secidx"].agg('mean').astype(int)
    dfagg["s1info_twap"]=dfnew["price"].agg("mean")
    dfagg["s1info_money"]=dfnew["money"].agg("sum")
    dfagg["s1info_volume"]=dfnew["volume"].agg("sum")
    dfagg["s1info_vwap"]=dfagg["s1info_money"]/dfagg["s1info_volume"]
    dfagg["s1info_tn"]=dfnew["price"].agg('count')
    
    dfagg["s1info_bvwap"]=dfbnew["money"].agg("sum")/dfbnew["volume"].agg("sum")
    dfagg["s1info_bhigh"]=dfbnew["price"].agg("max")
    dfagg["s1info_blow"]=dfbnew["price"].agg("min")
    dfagg["s1info_bpstd"]=dfbnew["price"].agg("std")
    # dfagg["s1info_bp25"]=dfbnew["price"].quantile(0.25, interpolation='higher')
    
    dfagg["s1info_svwap"]=dfsnew["money"].agg("sum")/dfsnew["volume"].agg("sum")
    dfagg["s1info_svolume"]=dfsnew["volume"].agg("sum")
    dfagg["s1info_smoney"]=dfsnew["money"].agg("sum")
    dfagg["s1info_shigh"]=dfsnew["price"].agg("max")
    dfagg["s1info_slow"]=dfsnew["price"].agg("min")
    dfagg["s1info_spstd"]=dfsnew["price"].agg("std")
    # dfagg["s1info_sp25"]=dfsnew["price"].quantile(0.75, interpolation='lower')
    
    dfagg["s1info_tsn"]=dfsnew["price"].agg("count")
    
    dfagg=dfagg.fillna(0)
    return dfagg

def aggdf(dd):
    pool = ThreadPool(32)
    def getp(df, pkey, vkey, volumelimit):
        df.sort_values(pkey, ascending=False)
        df.cumsum()
        
    def processdfdd(symbol):
        if symbol in dd:
            df=dd[symbol]
            dd[symbol]=processdf(df)  
    pool.map(processdfdd, ud.g_data["sids"])
    # processdf('BTCUSDT')
    return dd
 
def appenddata(dd, tmstart, tmend, pseconds=86400):
    pool = ThreadPool(32)
    secdr=g_secdata.secdr
    for tm in range(tmstart, tmend, pseconds*1000):
        secidx=int((tm-g_start_tm)/1000) 
        ss,ee=secidx, min(secidx+pseconds, int((tmend-g_start_tm)/1000))
        tmpdd={}
        for key in kline_fields:
            tmpdd[key]=np.zeros((ee-ss, len(ud.g_data["sids"])))
        
        def setvalue(idx):
            symbol=ud.g_data["sids"][idx]
            
            if symbol in dd:
                df=dd[symbol]
                df=df[(df.index>=ss) & (df.index <ee)]
                if df.shape[0]>0:
                    # if (df.index-ss).min()<0 or (df.index-ss).max()>=ee-ss:
                    #     print("error:",symbol,ss,ee, df.index[:10], flush=True)
                    tmpdd["s1info_money"][df.index-ss, idx]=df["s1info_money"]
                    tmpdd["s1info_volume"][df.index-ss, idx]=df["s1info_volume"]
                    tmpdd["s1info_twap"][df.index-ss, idx]=df["s1info_twap"]
                    tmpdd["s1info_vwap"][df.index-ss, idx]=df["s1info_vwap"]
                    tmpdd["s1info_tn"][df.index-ss, idx]=df["s1info_tn"]
                    
                    tmpdd["s1info_bvwap"][df.index-ss, idx]=df["s1info_bvwap"]
                    tmpdd["s1info_bhigh"][df.index-ss, idx]=df["s1info_bhigh"]
                    tmpdd["s1info_blow"][df.index-ss, idx]=df["s1info_blow"]
                    tmpdd["s1info_bpstd"][df.index-ss, idx]=df["s1info_bpstd"]
                    
                    tmpdd["s1info_svwap"][df.index-ss, idx]=df["s1info_svwap"]
                    tmpdd["s1info_svolume"][df.index-ss, idx]=df["s1info_svolume"]
                    tmpdd["s1info_smoney"][df.index-ss, idx]=df["s1info_smoney"]
                    tmpdd["s1info_tsn"][df.index-ss, idx]=df["s1info_tsn"]
                    tmpdd["s1info_shigh"][df.index-ss, idx]=df["s1info_shigh"]
                    tmpdd["s1info_slow"][df.index-ss, idx]=df["s1info_slow"]
                    tmpdd["s1info_spstd"][df.index-ss, idx]=df["s1info_spstd"]
                # else:
                #     print("df empty:", symbol,ss,ee, tools.tmu2i(g_start_tm+ss*1000), flush=True)
                    
                moneyzeroidx=np.where(tmpdd["s1info_vwap"][:,idx]==0)[0]
                for sidx in moneyzeroidx:
                    if sidx==0:
                        tmpdd["s1info_twap"][0, idx]=secdr["s1info_vwap"][ss-1][idx]
                        tmpdd["s1info_vwap"][0, idx]=secdr["s1info_vwap"][ss-1][idx]
                    else:
                        tmpdd["s1info_twap"][sidx, idx]=tmpdd["s1info_vwap"][sidx-1, idx]
                        tmpdd["s1info_vwap"][sidx, idx]=tmpdd["s1info_vwap"][sidx-1, idx]
                        
                moneyzeroidx=np.where(tmpdd["s1info_bvwap"][:,idx]==0)[0]
                for sidx in moneyzeroidx:
                    if sidx==0:
                        tmpdd["s1info_bvwap"][0, idx]=secdr["s1info_bvwap"][ss-1][idx]
                        tmpdd["s1info_bhigh"][0, idx]=secdr["s1info_bvwap"][ss-1][idx]
                        tmpdd["s1info_blow"][0, idx]=secdr["s1info_bvwap"][ss-1][idx]
                    else:
                        tmpdd["s1info_bvwap"][sidx, idx]=tmpdd["s1info_bvwap"][sidx-1, idx]
                        tmpdd["s1info_bhigh"][sidx, idx]=tmpdd["s1info_bvwap"][sidx-1, idx]
                        tmpdd["s1info_blow"][sidx, idx]=tmpdd["s1info_bvwap"][sidx-1, idx]
                        
                moneyzeroidx=np.where(tmpdd["s1info_svwap"][:,idx]==0)[0]
                for sidx in moneyzeroidx:
                    if sidx==0:
                        tmpdd["s1info_svwap"][0, idx]=secdr["s1info_svwap"][ss-1][idx]
                        tmpdd["s1info_shigh"][0, idx]=secdr["s1info_svwap"][ss-1][idx]
                        tmpdd["s1info_slow"][0, idx]=secdr["s1info_svwap"][ss-1][idx]
                    else:
                        tmpdd["s1info_svwap"][sidx, idx]=tmpdd["s1info_svwap"][sidx-1, idx]
                        tmpdd["s1info_shigh"][sidx, idx]=tmpdd["s1info_vwap"][sidx-1, idx]
                        tmpdd["s1info_slow"][sidx, idx]=tmpdd["s1info_vwap"][sidx-1, idx]
                tmpdd["valid"][sidx, idx]=1

            else:
                tmpdd["s1info_money"][:, idx]=np.nan
                tmpdd["s1info_volume"][:, idx]=np.nan              
                tmpdd["s1info_twap"][:, idx]=np.nan
                tmpdd["s1info_vwap"][:, idx]=np.nan
                tmpdd["s1info_tn"][:, idx]=np.nan
                
                tmpdd["s1info_bvwap"][:, idx]=np.nan
                tmpdd["s1info_bhigh"][:, idx]=np.nan
                tmpdd["s1info_blow"][:, idx]=np.nan
                tmpdd["s1info_bpstd"][:, idx]=np.nan
                
                tmpdd["s1info_svwap"][:, idx]=np.nan
                tmpdd["s1info_shigh"][:, idx]=np.nan
                tmpdd["s1info_slow"][:, idx]=np.nan
                tmpdd["s1info_svolume"][:, idx]=np.nan
                tmpdd["s1info_smoney"][:, idx]=np.nan
                tmpdd["s1info_tsn"][:, idx]=np.nan
                tmpdd["s1info_spstd"][:, idx]=np.nan
                
            return
        
        for idx in range(len(ud.g_data["sids"])):
            setvalue(idx)
 
        g_secdata.append_data(tmpdd, ss, ee)
        print("save seconds:",ss,ee, tools.tmu2i(g_start_tm+ss*1000), flush=True)
        
def process_months(months=["2024-01", "2024-02", "2024-03", "2024-04", "2024-05", "2024-06"]):
    pool = ThreadPool(32)
    # processpools=Pool(16)
    for idx in range(len(months)-1):
        month=months[idx]
        nextmonth=months[idx+1]
        tmstart=int(tools.string_toTimestamp(month+"-01T08:00:00")*1000)
        secdr=g_secdata.secdr
        tmstart=max(g_secdata.read_len()*1000+g_start_tm, tmstart)
        tmend=int(tools.string_toTimestamp(nextmonth+"-01T08:00:00")*1000)
        if tmstart >= tmend:
            continue
        
        dd=readzip(g_cfg["month_zippath"], ud.g_data["sids"], month)
        dd=aggdf(dd)
        appenddata(dd, tmstart, tmend)
        print("append month succ:",month, flush=True)
    return

def process_day(day):
    tmstart=int(tools.string_toTimestamp(day+"T08:00:00")*1000)
    tmend=tmstart+24*3600*1000
    secdr=g_secdata.secdr
    tmstart=max(g_secdata.read_len()*1000+g_start_tm, tmstart)
    
    if tmstart >= tmend:
        print("day data already exist:",day, flush=True)
        return
    
    dd=readzip(g_cfg["dayiy_zippath"], ud.g_data["sids"], day)
    dd=aggdf(dd)
    appenddata(dd, tmstart, tmend)
    print("append day succ:",day, flush=True)
    return

def downloadday(date, validsymbols, symbol):
    cmd=f'python /home/crypto/binancedata/binance-public-data/python/download-aggTrade.py -t um -s {symbol} -d {date} -skip-monthly 1 -c 1'
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
            time.sleep(1)
            print("download error:",symbol, date, maxcnt, flush=True)
    else:
        return 1

    return 0

def waitdownloaday(sids, date, validsymbols):
    pool = ThreadPool(32)
    while True:
        succnts=pool.map(partial(downloadday,date, validsymbols), sids)
        if sum(succnts) == len(sids):
            print("download all symbols succ:", date, flush=True)
            break
        print("download all symbols fail:", date, flush=True)
        time.sleep(300)
    return
       
def process_alldays(endday):
    endtm=int(tools.string_toTimestamp(endday+"T08:00:00")*1000)
    tmstart=g_secdata.read_len()*1000+g_start_tm
    for tm in range(tmstart, endtm, 24*3600*1000):
        day=tools.tmu2s_utc(tm, "%Y-%m-%d")
        ud.readuniverse(ud.g_data)
        lasteddownload=dmap.read_memmap(ud.g_data)
        #download delay half day
        while lasteddownload < ud.gettmidx(tm)+conts.daymincnt+conts.h1mincnt:
            print("wait new day:", day, lasteddownload, ud.g_data["min1info_tm"][lasteddownload], flush=True)
            time.sleep(3600)
            lasteddownload=dmap.read_memmap(ud.g_data)
            
        dr=ud.g_data
        valid=dr["min1info_money"][ud.gettmidx(tm):ud.gettmidx(tm)+conts.daymincnt].sum(axis=0)>0
        validsymbols=ud.g_data["sids"][valid]
        
        waitdownloaday(ud.g_data["sids"], day, validsymbols)
        process_day(day)

def merge_secdata(dfnew):
    # dfnew["secidx"]=((dfnew["transact_time"]-g_start_tm)/1000).astype(int)
    dfnew["money"]=dfnew["price"]*dfnew["quantity"]
    dfnew["volume"]=dfnew["quantity"]

    dfsnew=dfnew[dfnew["is_buyer_maker"]]
    dfbnew=dfnew[~dfnew["is_buyer_maker"]]

    dfagg={}
    if dfnew.shape[0]>0:
        dfagg["s1info_money"]=dfnew["money"].sum()
        dfagg["s1info_volume"]=dfnew["volume"].sum()
        dfagg["s1info_twap"]=dfnew["price"].mean()
        dfagg["s1info_vwap"]=dfagg["s1info_money"]/dfagg["s1info_volume"]
        dfagg["s1info_tn"]=dfnew.shape[0]
    else:
        dfagg["s1info_money"]=0
        dfagg["s1info_volume"]=0
        dfagg["s1info_tn"]=0
        dfagg["s1info_twap"]=0
        dfagg["s1info_vwap"]=0
    
    if dfbnew.shape[0]>0:
        dfagg["s1info_bhigh"]=dfbnew["price"].max()
        dfagg["s1info_blow"]=dfbnew["price"].min()
        dfagg["s1info_bpstd"]=dfbnew["price"].std()
        dfagg["s1info_bvwap"]=(dfbnew["money"].sum())/(dfbnew["volume"].sum())
    else:
        dfagg["s1info_bhigh"]=0
        dfagg["s1info_blow"]=0
        dfagg["s1info_bpstd"]=0
        dfagg["s1info_bvwap"]=0
        
    if dfsnew.shape[0]>0:   
        dfagg["s1info_svolume"]=dfsnew["volume"].sum()
        dfagg["s1info_smoney"]=dfsnew["money"].sum()
        dfagg["s1info_svwap"]=dfagg["s1info_smoney"]/dfagg["s1info_svolume"]
        dfagg["s1info_shigh"]=dfsnew["price"].max()
        dfagg["s1info_slow"]=dfsnew["price"].min()
        dfagg["s1info_spstd"]=dfsnew["price"].std()
        dfagg["s1info_tsn"]=dfsnew.shape[0]
    else:
        dfagg["s1info_svolume"]=0
        dfagg["s1info_smoney"]=0
        dfagg["s1info_shigh"]=0
        dfagg["s1info_slow"]=0
        dfagg["s1info_spstd"]=0
        dfagg["s1info_tsn"]=0
        dfagg["s1info_svwap"]=0
    
    return dfagg

g_log_ratio=0.001
# @numba.jit(nopython=True)
def merge_and_write(symbol, df, secdata, transact_time):
    sidx=ud.g_data['sidmap'][symbol]
    secidx=int((transact_time-g_start_tm)/1000)
    dfagg=merge_secdata(df)
    secdata.write_data(secidx, sidx, dfagg.keys(), dfagg)
    return    
   
def set_valid_zero(symbol, s, e, secdata):
    sidx=ud.g_data['sidmap'][symbol]
    ss=int((s-g_start_tm)/1000)
    ee=int((e-g_start_tm)/1000)+1
    secdata.set_valid_zero(sidx, ss, ee)
    print("set_valid_zero:", symbol, sidx, tools.tmu2i(s), tools.tmu2i(e), ss, ee, flush=True)

def fetch_and_add_data(umcs, symbol, start_time, end_time, limit=1000):
    num_retries=0
    max_retries=10
    while num_retries < max_retries:
        try:
            all_fix_data = []
            old_start_time = start_time
            while True:
                rest_client=umcs.get_um_client()
                fetch_data = rest_client.client.agg_trades(symbol=symbol, startTime=start_time, endTime=end_time, limit=1000)
                if len(fetch_data)> 0:
                    all_fix_data+=fetch_data
                    start_time=fetch_data[-1]['T']
                    for i in range(len(all_fix_data)):
                        if all_fix_data[len(all_fix_data)-i-1]['T'] != start_time:
                            break
                    if i !=(len(all_fix_data)-1):
                        all_fix_data=all_fix_data[:len(all_fix_data)-i]
                    else:
                        start_time=fetch_data[-1]['T']+1
                if len(fetch_data) < 1000 or (start_time-old_start_time)>1000:
                    break
            break
        except Exception as ex:
            num_retries+=1
            time.sleep(5)
            print("retry featch cnt:",  num_retries, symbol, flush=True)
            traceback.print_exc()
            if num_retries > max_retries:
                rest_client.set_flag(0)
                print("retry_fetch exceed:", num_retries, symbol, flush=True)
                return None        
    df = pd.DataFrame([{
        'event_time': item['T'],
        'agg_order_id': item['a'],
        'price': item['p'],
        'quantity': item['q'],
        'first_order_id': item['f'],
        'last_order_id': item['l'],
        'transact_time': item['T'],
        'is_buyer_maker': item['m']
    } for item in all_fix_data], 
        columns=['event_time',
    'agg_order_id',
    'price',
    'quantity',
    'first_order_id',
    'last_order_id',
    'transact_time',
    'is_buyer_maker'])
    

    df['price']=df['price'].astype(np.float32)
    df['quantity']=df['quantity'].astype(np.float32)
    df['transact_time']=df['transact_time'].astype(int)
    df['is_buyer_maker']=df['is_buyer_maker'].astype(bool)
    print("fetch rest data succ", symbol, tools.tmu2i(start_time), df.shape[0], flush=True)
    return df

# def process_rest_data(secdata, symbol, df, starttm):
#     if df.shape[0]>0:
#         empytdf=pd.DataFrame(columns=df.columns)
#         endtm=int(df["transact_time"].max()/1000)*1000
#         df["transact_time"]=(df["transact_time"]/1000).astype(int)
#         dfagg=df.groupby("transact_time")
#         dd={}
#         for secidx, tmpdf in dfagg:
#             dd[secidx]=tmpdf
#         for tm in range(starttm, endtm, 1000):
#             if int(tm/1000) in dd:
#                 tmpdf=dd[int(tm/1000)]
#             else:
#                 tmpdf=empytdf
#             merge_and_write(symbol, tmpdf, secdata, tm)
#         print("process rest data succ", symbol, df.shape[0], tools.tmu2i(starttm), tools.tmu2i(endtm), flush=True)
        
def process_rest_data(secdata, symbol, df, starttm, endtm):
    endtm=int(df["transact_time"].max()/1000)*1000 if df.shape[0]>0 else endtm
    endtm=max(starttm+1000, endtm)
    for tm in range(starttm, endtm, 1000):
        tmpdf=df[(df['transact_time']>=tm) & (df['transact_time']<tm+1000)].copy()
        merge_and_write(symbol, tmpdf, secdata, tm)
    print("process rest data succ", symbol, df.shape[0], tools.tmu2i(starttm), tools.tmu2i(endtm), flush=True)
            
def update_llen(secdata, cfg):
    umcs=um_client.UmClient(cfg)
    secdr=secdata.secdr
    sids=sec_trades.getsids()
    sidslen=len(sids)
    
    while True:
        cur_tm=int(time.time()*1000)
        cur_tm_idx=gtmidx_u(cur_tm)
        llen=secdata.read_len()
        secdata_tm=llen*1000+g_start_tm
    
        # secdata.set_valid(llen, cur_tm_idx-60*20)
        if llen < cur_tm_idx :
            need_fix_data=False
            recnt=0
            for reidx in range(recnt):
                if secdr["valid"][llen-recnt+reidx].sum() < len(sids):
                    secdata.write_len(llen-recnt+reidx)
                    need_fix_data=True
                    print("fix data:", tools.tmu2i(secdata_tm), "error second:",  recnt-reidx, flush=True)
                    break
            if need_fix_data:
                continue
                    
                    
            if secdr["valid"][llen].sum() == len(sids):
                st=int(time.time()*1000)
                secdata.append_data_inner(llen)
                et=int(time.time()*1000)
                print("data last tm:", tools.tmu2i(secdata_tm), "delay:",  (et-secdata_tm-1000), 
                      "up time:", tools.tmu2i(secdata_tm+1000),
                      "cur_tm:", datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'), flush=True)
            elif secdata_tm < cur_tm-10000:  #wait 10 s, need rest api get data
                for idx in range(secdr["valid"][llen].shape[0]):
                    sid=ud.g_data['sids'][idx]
                    if secdr["valid"][llen][idx] == 0 and ud.g_data['sids'][idx] in sids:
                        tmdelta=2000
                        endidx=np.where(secdr["valid"][llen:llen+tmdelta,idx]==1)[0]
                        endidx = llen + (endidx[0] if endidx.shape[0]>0 else tmdelta) + 2
                        endtm=endidx*1000+g_start_tm
                        df=fetch_and_add_data(umcs, sid, secdata_tm, endtm)
                        if not df is None:
                            process_rest_data(secdata, sid, df, int(secdata_tm/1000)*1000, int(endtm/1000-1)*1000)
                        print("process symbol idx:", idx, tools.tmu2i(secdata_tm), tools.tmu2i(endtm), flush=True)
            else:
                time.sleep(0.01)
        else:
            time.sleep(0.01)

    return
    
    
if __name__ == "__main__":
    config_logging(logging, logging.WARNING)
    
    cfgpath="./config/sec_klines.yaml"
    with open(cfgpath) as f:
        g_cfg = yaml.load(f, Loader=yaml.FullLoader)
    os.environ["http_proxy"]=g_cfg["http_proxy"]
    os.environ["https_proxy"]=g_cfg["https_proxy"]
    ud.readuniverse(ud.g_data)
    g_secdata=SecondData(g_cfg["secdata_path"], ud.g_data["sids"], kline_fields, mode="r+")
    process_months(["2024-01", "2024-02", "2024-03", "2024-04", "2024-05", "2024-06"])

    # process_alldays('2024-08-08')
    
    update_llen(g_secdata, g_cfg["um_client"])
    

        


        


