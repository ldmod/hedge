import sys
import os
sys.path.append(os.path.abspath(__file__+"../../../../"))
import pandas as pd
import numpy as np
import pickle
from multiprocessing import Pool
import time
import multiprocessing
from functools import partial
import concurrent.futures as futures
import datetime
import itertools
import h5py
from multiprocessing import Pool
import time
import cryptoqt.data.feaextractor as feaextractor
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
import cryptoqt.data.runextractor as runext
import cryptoqt.data.datammap as dmmap
import threading
import cryptoqt.data.um_client_klines as umc
import yaml
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option("display.max_colwidth", 100)
pd.set_option('display.width', 10000)

def readuniverse(g_data):
    global g_spotsymbols
    universepath=g_path+"universe/"

    g_data["sids"]=np.load(universepath+"sids.npy")
    sidmap={}
    for sidx in range(g_data["sids"].shape[0]):
        sidmap[g_data["sids"][sidx]]=sidx
    g_data['sidmap']=sidmap
    g_data["min1info_tm"]=np.load(universepath+"min1info_tm.npy")
    g_data["g_spotsymbols"]=np.load(universepath+"g_spotsymbols.npy")
    g_spotsymbols=g_data["g_spotsymbols"]

def setuniverse():
    universepath=g_path+"universe/"
    
    sids_less=pd.read_csv(f"{universepath}/sids_hedge.csv")
    
    tsymbols=[x for x in sids_less['symbol'] if x[-4:]=='USDT']

    np.save(universepath+"sids", np.array(tsymbols))
    
    tms=[]
    for tm in  range(g_start_tm, g_start_tm+g_mspermin*60*24*3000, g_mspermin):
        tms.append(tools.tmu2i(tm))
    np.save(universepath+"min1info_tm", np.array(tms))

    myproxies = {
            'http': 'http://127.0.0.1:33881',
            'https': 'http://127.0.0.1:33881'
    }
    spot_client = Client( proxies=myproxies, timeout=10)
    g_spotsymbols=spot_client.exchange_info()['symbols']
    g_spotsymbols=[x['symbol'] for x in g_spotsymbols]
    g_spotsymbols= [x for x in tsymbols if x in g_spotsymbols]
    np.save(universepath+"g_spotsymbols", np.array(g_spotsymbols))
    
def getdlastidx():
    h5path=g_h5path
    path=h5path+"smin1info.h5"
    tidx = 0
    if os.path.exists(path):
        f=h5py.File(path, "r")
        if "tm" in f.keys():
            tidx=f["tm"].shape[0]
        f.close()
    return  tidx

def gettmidx(tm, frequency='1m'):  
    if frequency=='1d':
        cnt=g_mspermin*1440
    elif frequency=='1m':
        cnt=g_mspermin
    idx=math.floor((tm-g_start_tm)/cnt)
    return idx

def gettm(idx, frequency='1m'):  
    if frequency=='1d':
        cnt=g_mspermin*1440
    elif frequency=='1m':
        cnt=g_mspermin
    idx=g_start_tm+cnt*idx
    return idx

def retry_fetch_ohlcv(savename, max_retries, symbol, timeframe, startTime, endTime, cidx):
    num_retries = 0
    ohlcvretry=0
    while num_retries< max_retries and ohlcvretry < 3:
        try:
            num_retries += 1
            if savename[0]=='s':
                if symbol in g_spotsymbols:
                    client=spot_clients.get_um_client()
                    ohlcv=client.client.klines(symbol, timeframe, limit=1500, startTime=startTime, endTime=endTime)
                    spot_clients.recycle(client)
                else:
                    ohlcv=[]
            else:
                client=um_futures_clients.get_um_client()
                ohlcv = client.client.continuous_klines(symbol, "PERPETUAL", timeframe, limit=1500, startTime=startTime, endTime=endTime)
                um_futures_clients.recycle(client)
            if len(ohlcv) <=0 and ohlcvretry < 2:
                print("error len ohlcv 0:", savename, startTime, symbol, ohlcvretry, flush=True)
                ohlcvretry+=1
                continue
            return ohlcv
        except Exception as ex:
            time.sleep(5)
            print("retry_fetch_ohlcv cnt:", cidx, num_retries, symbol, flush=True)
            traceback.print_exc()
            # cidx=(cidx+1)%len(um_futures_clients)
            if num_retries > max_retries:
                print("retry_fetch_ohlcv exceed:", num_retries, symbol, flush=True)
                raise  # Exception('Failed to fetch', timeframe, symbol, 'OHLCV in', max_retries, 'attempts')

g_clientmap_lock = threading.Lock()
g_clientmap={}
def updatevalue(frequency, start_tm, end_tm, h5f, savename, off, sid):
    # limit maxvalue is 1000!
    cnt=gettmidx(end_tm, frequency)-gettmidx(start_tm, frequency)
    tidx=threading.currentThread().ident
    with g_clientmap_lock:
        if tidx not in g_clientmap:
            g_clientmap[tidx]=len(g_clientmap)
    
    # cidx=g_clientmap[tidx]
    cidx=g_data["sids"].tolist().index(sid)%g_clientnum
    # cidx=0
    
    # print("tidx:", tidx, cidx, flush=True)

    data=retry_fetch_ohlcv(savename, 100, sid, frequency, start_tm, end_tm, cidx)
    symbolidx=g_data["sids"].tolist().index(sid)
    succnt=0
    if data is None:
        print("error tm update", frequency, sid, tools.tmu2i(start_tm), tools.tmu2i(end_tm), flush=True)
        return -1
    for item in data:
        tm=item[0]
        tmidx=gettmidx(tm, frequency)
        
        if tm >=start_tm and tm < end_tm:
            succnt+=1
            for idx,key in enumerate(kline_fields):
                h5f[savename+"_"+key][tmidx-off, symbolidx]=item[idx]
        else:
            # print("error tm update", sid, tools.tmu2i(start_tm), tools.tmu2i(end_tm), tools.tmu2i(tm))
            continue
    print("end update", savename, frequency, sid, tools.tmu2i(start_tm), tools.tmu2i(end_tm), succnt, flush=True)
    return succnt
    
def bs_dumpdayh5(ss, end_tm,  frequency, savename):
    delta=g_mspermin if frequency == '1m' else g_mspermin*1440
    end_tm_idx=gettmidx(end_tm, frequency)
    h5path=g_h5path
    fpath=h5path+savename+".h5"
    
    tpl=tools.TimeProfilingLoop(savename)

    stocklist=ss.tolist()
    net=tpl.add("net")
    if not os.path.exists(fpath):
        f=tools.writeh5(h5path+savename+".h5")
        f.create_dataset("tm", shape=(0,), maxshape = (None,), chunks = True, dtype=np.int64)
        for key in kline_fields:
            f.create_dataset(savename+"_"+key, shape=(0, len(ss)), maxshape = (None, len(ss)), chunks = True)
        f.swmr_mode=True
        start_tm=g_start_tm
        f.close()
    else:
        f=tools.writeh5(h5path+savename+".h5")
        if f["tm"].shape[0] <=0:
            start_tm=g_start_tm
        else:
            start_tm=f["tm"][-1]+delta
        f.close()
    start_tm_idx=gettmidx(start_tm, frequency)
    fixend_tm=gettm(end_tm_idx, frequency)
    if end_tm_idx <= start_tm_idx or end_tm_idx <0:
        print("alread unpdate data", tools.tmu2i(start_tm), tools.tmu2i(fixend_tm), frequency)
        return

    while True:
        dd={}
        for idx,key in enumerate(kline_fields):
            dd[savename+"_"+key]=np.zeros((end_tm_idx-start_tm_idx, ss.shape[0]))
            dd[savename+"_"+key].fill(np.nan)
                
        fetchfunc=partial(updatevalue, frequency, start_tm, fixend_tm, dd, savename, start_tm_idx)
        executor = ThreadPoolExecutor(max_workers=g_clientnum)
        rets=executor.map(fetchfunc, ss.tolist())
        for idx, ret in enumerate(rets):
            print("end update info:", ss[idx], ret, frequency, tools.tmu2i(start_tm), tools.tmu2i(end_tm), flush=True)
            if ret < 0:
                print("download error:", ss[idx], ret, frequency, tools.tmu2i(start_tm), tools.tmu2i(end_tm), flush=True)
                exit(0)
        # for sidx, sid in enumerate(ss):
        #     fetchfunc(sid)
        if (np.isfinite(dd[savename+"_close"][:,0]).sum()==dd[savename+"_close"].shape[0] and
            np.isfinite(dd[savename+"_close"][:,1]).sum()==dd[savename+"_close"].shape[0])  \
                or (fixend_tm >= tools.tmi2u(20230324200000) and fixend_tm <= tools.tmi2u(20230325080000)):
            break
        print("data error:", savename, tools.tmu2i(start_tm), tools.tmu2i(end_tm))
        time.sleep(60)
    
    f=tools.writeh5(h5path+savename+".h5")
    for key in f.keys():
        if key != "tm":
            f[key].resize((end_tm_idx,)+ f[key].shape[1:])
            f[key][start_tm_idx:,:]=dd[key][:,:]
    key='tm'
    f[key].resize((end_tm_idx,)+ f[key].shape[1:])
    for idx in range(start_tm_idx, end_tm_idx):
        f["tm"][idx]=idx*delta+g_start_tm
    
    f.flush()
    f.close()  
    # time.sleep(10)
    net.end()
    
    tpl.end()
    print("save succ", savename, frequency, tools.tmu2i(start_tm), tools.tmu2i(end_tm), tpl.to_string(), flush=True)
    for client in um_futures_clients.um_futures_clients:
        print(client.use, client.proxies, flush=True)
    return
    
class GetKline:
    def __init__(self, proxies, q):
        self.proxies=proxies
        self.um_futures_client=UMFutures(proxies=proxies)
        self.spot_client = Client(base_url="https://api.binance.com", proxies=proxies)
        self.q=q
    def run(self):
        while True:
            item=self.q.get()

def readh5new(infoname=["dayinfo", "min1info"], mem=False):
    h5path=g_h5path
    readuniverse(g_data)
    fs=[]
    for info in infoname:
        f=h5py.File(h5path+info+".h5", 'r', swmr=True, libver='latest')
        for key in f.keys():
            if key != "tm":
                if mem:
                    g_data[key]=f[key][:]
                else:
                    g_data[key]=f[key]
            else:
                tms=[]
        if mem:
            f.close()
        else:
            fs.append(f)
    return fs

g_spotsymbols=None

g_start_tm=tools.tmi2u(20220101080000)
g_mspermin=60000
g_path='/data/nb/proddata/crypto/'
g_h5path=g_path+"/h5/"

g_data={}
kline_fields=['opentm', 'open', 'high', 'low', 'close','volume','closetm', 'money', 'tnum', 'tbv', 'tbm', 'ignore']  

def updateforover():

    readuniverse(g_data)
    mapdd={}
    dmmap.load_memmap(mapdd)
    msperday=1000*60*60*24

    while True:
        tm=time.time()
        tmu=int(tm*10)*100
        downloadidx=getdlastidx()
        start_tm=tools.tmi2u(g_data["min1info_tm"][downloadidx])
        if tmu > start_tm+int(msperday/2):
            end_tm=start_tm+int(g_mspermin*720)
        else:
            end_tm=tmu-int(g_mspermin*0) #delay 3 min
            tmsecond=tools.tmu2i(tmu)
            # if int(tmsecond / 100+1)%10 in [0, 5]:
            if int(tmsecond / 100+1)%100 in [0, 15, 30, 45]:
                um_futures_clients.test_client_ping()
                um_futures_clients.test_client_ping()
                
            # if not ((int(tmsecond / 100)%10 in [0,5]) \
            if not (int(tmsecond / 100)%100 in [0,15, 30, 45] \
            # if not (int(tmsecond / 100)%100 in [0, 30] \
                    and tmsecond %100 > 1 \
                        and end_tm-start_tm > g_mspermin):

                time.sleep(1)
                continue

                
        print("start update to:", tools.tmu2i(start_tm), tools.tmu2i(end_tm), tools.tmu2i(tmu), flush=True)
        tpl=tools.TimeProfilingLoop("updatedata")
        bs_dumpdayh5(g_data["sids"], end_tm, "1m", "min1info")
        bs_dumpdayh5(g_data["sids"], end_tm, "1m", "smin1info")

        tpl.end()
        print("update to:", tools.tmu2i(end_tm), tpl.to_string(), flush=True)
        
        tpl=tools.TimeProfilingLoop("runext")    
        fs=[]
        fs+=readh5new(["min1info", "smin1info"])
        runext.addextrafields(g_data)
        fs+=readh5new(["min1info_extra", "smin1info_extra"])
        runext.addmin5fields(g_data)
        tpl.end()
        for  f in fs:
            f.close()
        print("runext to:", tools.tmu2i(end_tm), tpl.to_string(), flush=True)
        dmmap.update_memmap(mapdd)
        
def reseth5(end_tm, infoname=["min1info", "smin1info"]):
    h5path=g_h5path
    readuniverse(g_data)
    h5path=g_h5path
    for info in infoname:
        f=tools.writeh5(h5path+info+".h5")
        llen=g_data["min1info_tm"][:].tolist().index(end_tm)
        f["tm"].resize(llen, axis=0)
        f.close()
    return  

    
if __name__ == "__main__":
    timeout=2
    cfgpath="./config/updatedata.yaml"
    with open(cfgpath) as f:
        g_cfg = yaml.load(f, Loader=yaml.FullLoader)
    um_futures_clients=umc.UmClient(g_cfg["um_client"], UMFutures)  
    spot_clients=umc.UmClient(g_cfg["um_client"], Client)
    # g_clientnum=len(um_futures_clients.um_futures_clients)
    g_clientnum=20
    
    config_logging(logging, logging.WARNING)
    # setuniverse()
    # updatedatah5()
    # reseth5(20240801160000)
    updateforover()
    # redo_updatedatah5()
    
    readh5new(["min1info", "smin1info"])

    dr=g_data
    min1i=1000030
    idx=5
    tools.tmi2s(dr["min1info_tm"][min1i]), dr["sids"][idx],dr["min1info_close"][min1i, idx]

        


        

