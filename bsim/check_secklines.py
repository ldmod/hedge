import sys
import os
sys.path.append(os.path.abspath(__file__+"../../../../../"))
import numpy as np
import torch
import cryptoqt.data.updatedata as ud
from functools import partial
import cryptoqt.data.datammap as dmap
from binance.um_futures import UMFutures
import cryptoqt.data.tools as tools
from binance.spot import Spot as Client
import time
import cryptoqt.data.sec_klines.sec_klines as sk
import pandas as pd
import mplfinance as mpf

kline_fields=['opentm', 'open', 'high', 'low', 'close','volume','closetm', 'money', 'tnum', 'tbv', 'tbm', 'ignore']  

def diffvwap(secdr, xx=7330400, idx=0, delta=10):
    svwap=(secdr["s1info_smoney"][xx:xx+delta,idx].sum()/secdr["s1info_svolume"][xx:xx+delta,idx].sum())
    vwap=(secdr["s1info_money"][xx:xx+delta,idx].sum()/secdr["s1info_volume"][xx:xx+delta,idx].sum())
    bvwap=(secdr["s1info_money"][xx:xx+delta,idx].sum()-secdr["s1info_smoney"][xx:xx+delta,idx].sum())/(
        secdr["s1info_volume"][xx:xx+delta,idx].sum()-secdr["s1info_svolume"][xx:xx+delta,idx].sum())
    diff=(svwap/bvwap-1)*10000
    diffsv=(svwap/vwap-1)*10000
    print("svwap:", svwap, "bvwap:", bvwap, "vwap:", vwap, "diff:", diff, diffsv)
  

def checkdata(dr, secdr, min1i, sidx):
    global secstartidx
    tm=dr["min1info_tm"][min1i-1]

    volume=dr["min1info_volume"][min1i-1, sidx]
    volume2=secdr["s1info_volume"][(min1i-1-secstartidx)*60:(min1i-secstartidx)*60].sum(axis=0)[sidx]
    
    money=dr["min1info_money"][min1i-1, sidx]
    money2=secdr["s1info_money"][(min1i-1-secstartidx)*60:(min1i-secstartidx)*60].sum(axis=0)[sidx]
    
    # tn=dr["min1info_tnum"][min1i-1, sidx]
    # tn2=secdr["s1info_tn"][(min1i-1-secstartidx)*60:(min1i-secstartidx)*60].sum(axis=0)[sidx]
    if abs(money2-money)>10:
        print(tm, dr["sids"][sidx], sidx, (volume2-volume)/volume, (money2-money)/money,flush=True)
        return 1
    return 0

    
if __name__ == "__main__":
    did="1"
    if len(sys.argv)>1:
        did=sys.argv[1]
    os.environ['CUDA_VISIBLE_DEVICES'] = did
    os.environ["HDF5_USE_FILE_LOCKING"]="False"

    ud.readuniverse(ud.g_data)
    lasteddownload=dmap.read_memmap(ud.g_data)
    dr=ud.g_data
    secstartidx=dr["min1info_tm"].tolist().index(20240101080000)
    secdata=sk.SecondData('/home/crypto/proddata/crypto/secdata/', ud.g_data["sids"])
    secdr=secdata.secdr
    llen=secdata.read_len()
    
    cnt=10000
    
    lasteddownload=ud.gettmidx(sk.gtm_u(llen))
    a=np.random.randint(lasteddownload-60, lasteddownload, size=[cnt])
    b=np.random.randint(0, dr["sids"].shape[0], size=[cnt])
    loss_cnt=0
    for i in range(a.shape[0]):
        loss_cnt+=checkdata(dr, secdr, a[i], b[i])
    print("loss_ratio:", loss_cnt/cnt)
    
    
