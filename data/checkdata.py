import sys
import os
sys.path.append(os.path.abspath(__file__+"../../../../"))
import numpy as np
import torch
import cryptoqt.data.updatedata as ud
from functools import partial
import cryptoqt.data.datammap as dmap
from binance.um_futures import UMFutures
import cryptoqt.data.tools as tools
from binance.spot import Spot as Client
import time
myproxies = {
        'http': 'http://127.0.0.1:7990',
        'https': 'http://127.0.0.1:7990'
}

um_futures_client = UMFutures(proxies=myproxies)
spot_client = Client(base_url="https://api.binance.com", proxies=myproxies)
kline_fields=['opentm', 'open', 'high', 'low', 'close','volume','closetm', 'money', 'tnum', 'tbv', 'tbm', 'ignore']  
def checkdata(dr, min1i, sidx):
    tm=dr["min1info_tm"][min1i-1]
    close=dr["min1info_close"][min1i-1, sidx]
    open=dr["min1info_open"][min1i-1, sidx]
    volume=dr["min1info_volume"][min1i-1, sidx]
    item=um_futures_client.continuous_klines(dr["sids"][sidx], "PERPETUAL","1m", startTime=tools.tmi2u(tm), limit=1)
    if (tools.tmu2i(item[0][0]) != tm):
        aa=0
        return 0
    cclose=float(item[0][kline_fields.index("close")])
    cvolume=float(item[0][kline_fields.index("volume")])
    print(tm, dr["sids"][sidx], sidx, (close-cclose)/close, (volume-cvolume)/volume,flush=True)
    if abs(cclose/close-1.0)>0.001 or abs(cvolume/volume-1.0) > 0.001:
        print(tm, dr["sids"][sidx], sidx, abs(cclose/close-1.0), abs(cvolume/volume-1.0),flush=True)
        return 1
    return 0

def selfcheck(dr, lastedMin1Idx):
    # dr["min1info_tm"].tolist().index(20241128073900)
    for min1i in range(lastedMin1Idx):
        ehFlag = dr["min1info_vwap"][min1i] > dr["min1info_high"][min1i]*1.001
        llFlag = dr["min1info_vwap"][min1i] < dr["min1info_low"][min1i]*0.999
        if ehFlag.sum()>0:
            print(f"ehFlag {dr['min1info_tm'][min1i]}-{np.where(ehFlag)}")
        if llFlag.sum()>0:
            print(f"llFlag {dr['min1info_tm'][min1i]}-{np.where(llFlag)}")
            
        ehFlag = dr["smin1info_vwap"][min1i] > dr["smin1info_high"][min1i]*1.001
        llFlag = dr["smin1info_vwap"][min1i] < dr["smin1info_low"][min1i]*0.999
        if ehFlag.sum()>0:
            print(f"sehFlag {dr['min1info_tm'][min1i]}-{np.where(ehFlag)}")
        if llFlag.sum()>0:
            print(f"sllFlag {dr['min1info_tm'][min1i]}-{np.where(llFlag)}")
    return
    

def checkdata2(dr, min1i, sidx):
    tm=dr["min1info_tm"][min1i-1]
    close=dr["smin1info_close"][min1i-1, sidx]
    open=dr["smin1info_open"][min1i-1, sidx]
    volume=dr["smin1info_volume"][min1i-1, sidx]
    item=spot_client.klines(dr["sids"][sidx],"1m", startTime=tools.tmi2u(tm), limit=1)
    cclose=float(item[0][kline_fields.index("close")])
    cvolume=float(item[0][kline_fields.index("volume")])
    print(tm, dr["sids"][sidx], sidx, (close-cclose)/close, (volume-cvolume)/volume,flush=True)
    time.sleep(0.5)
    return
    
if __name__ == "__main__":
    did="1"
    if len(sys.argv)>1:
        did=sys.argv[1]
    os.environ['CUDA_VISIBLE_DEVICES'] = did 
    os.environ["HDF5_USE_FILE_LOCKING"]="False"

    ud.readuniverse(ud.g_data)
    lasteddownload=dmap.read_memmap(ud.g_data)
    dr=ud.g_data
    # selfcheck(dr, lasteddownload)
    cnt=2000
    a=np.random.randint(lasteddownload-1440, lasteddownload, size=[cnt])
    b=np.random.randint(0, dr["sids"].shape[0], size=[cnt])
    loss_cnt=0
    for i in range(a.shape[0]):
        loss_cnt+=checkdata(dr, a[i], b[i])
    print("loss_ratio:", loss_cnt/cnt)
    
    
