import os
import sys
sys.path.append('/home/crypto/')
import numpy as np
import torch
import cryptoqt.data.updatedata as ud
from functools import partial
import cryptoqt.data.datammap as dmap
from binance.um_futures import UMFutures
import cryptoqt.data.tools as tools
from binance.spot import Spot as Client
import time
import cryptoqt.data.sec_klines.secklines as sk
import pandas as pd
import mplfinance as mpf

def plotsec(dr, secdata, symbol, stm, etm=0, mode='s'):
    if etm==0:
        etm=tools.tmu2i((tools.tmi2u(stm)+5*60*1000))
    ss=sk.gtmidx_i(stm)
    ee=sk.gtmidx_i(etm)
    sidx=dr["sids"].tolist().index(symbol)
    secdr=secdata.secdr
    llen=secdata.read_len()
    assert ss>0 and ss <llen and ee>ss and ee< llen
    vwap=secdr["s1info_vwap"][ss:ee, sidx]
    money=secdr["s1info_money"][ss:ee, sidx]
    if np.isfinite(vwap).sum()==0:
        print("not valid symbol:", symbol)
        return
    
    df=pd.DataFrame()
    tms=pd.date_range(tools.tmi2s(stm),tools.tmi2s(etm),freq="1s")[:-1]
    df.index=tms
    if mode =='s':
        df["open"]=secdr["s1info_svwap"][ss:ee, sidx]
        df["close"]=secdr["s1info_svwap"][ss:ee, sidx]
        df["high"]=secdr["s1info_shigh"][ss:ee, sidx]
        df["low"]=secdr["s1info_slow"][ss:ee, sidx]
        df["volume"]=secdr["s1info_svolume"][ss:ee, sidx]
    elif mode =='b':
        df["open"]=secdr["s1info_bvwap"][ss:ee, sidx]
        df["close"]=secdr["s1info_bvwap"][ss:ee, sidx]
        df["high"]=secdr["s1info_bhigh"][ss:ee, sidx]
        df["low"]=secdr["s1info_blow"][ss:ee, sidx]
        df["volume"]=secdr["s1info_volume"][ss:ee, sidx]-secdr["s1info_svolume"][ss:ee, sidx]
    else:
        df["open"]=secdr["s1info_vwap"][ss:ee, sidx]
        df["close"]=secdr["s1info_vwap"][ss:ee, sidx]
        df["high"]=secdr["s1info_bhigh"][ss:ee, sidx]
        df["low"]=secdr["s1info_slow"][ss:ee, sidx]
        df["volume"]=secdr["s1info_volume"][ss:ee, sidx]
    mc = mpf.make_marketcolors(
    up="red",  # 上涨K线的颜色
    down="green",  # 下跌K线的颜色
    edge="black",  # 蜡烛图箱体的颜色
    volume="blue",  # 成交量柱子的颜色
    wick="black",  # 蜡烛图影线的颜色
    )
    s = mpf.make_mpf_style(
    gridaxis='both',
    gridstyle='-.',
    y_on_right=True,
    marketcolors=mc,
    edgecolor='b',
    # figcolor='r',
    # facecolor='y', 
    gridcolor='c')

    mpf.plot(df, type='candle', volume=True, title=symbol, style=s, update_width_config=dict(candle_linewidth=1))
    return
    
if __name__ == "__main__":

    ud.readuniverse(ud.g_data)
    lasteddownload=dmap.read_memmap(ud.g_data)
    dr=ud.g_data
    secdata=sk.SecondData('/home/crypto/proddata/crypto/secdata/', ud.g_data["sids"])
    secdr=secdata.secdr
    llen=secdata.read_len()
    
    # plotsec(dr, secdata, 'BTCUSDT', 20240701180000)
    # plotsec(dr, secdata, 'CRVUSDT', 20240706180000)
    
    tms=np.random.randint(tools.tmi2u(20240802000000), tools.tmi2u(20240816000000), size=(1000))
    for sidx,sid in enumerate(dr["sids"]):
        plotsec(dr, secdata, sid, tools.tmu2i(tms[sidx]))
    
    
    
    
    
    
    
    
    
    
    
