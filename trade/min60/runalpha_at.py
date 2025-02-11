#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 12:40:22 2024

@author: ld
"""
import os
import sys
sys.path.append(os.path.abspath(__file__+"../../../../../"))
import pandas as pd
import numpy as np
import warnings
warnings.simplefilter("ignore", category=RuntimeWarning)
import h5py
import time
import datetime
from collections import deque
import cryptoqt.data.data_manager as dm
from functools import partial
import matplotlib.pyplot as plt
import cryptoqt.alpha.dr5 as dr5
import cryptoqt.data.constants as conts
import cryptoqt.alpha.day_alpha as day_alpha
import cryptoqt.alpha.h4_alpha as h4_alpha
import cryptoqt.alpha.min15_alpha as min15_alpha
import cryptoqt.alpha.h1_alpha as h1_alpha
import cryptoqt.alpha.min30_alpha as min30_alpha
import cryptoqt.alpha.min5_alpha as min5_alpha
import cryptoqt.data.tools as tools
import cryptoqt.data.datammap as dmap
import argparse
from scipy.stats import rankdata
import scipy.stats
daymincnt=conts.daymincnt
h4mincnt=conts.h4mincnt
h1mincnt=conts.h1mincnt
min15mincnt=conts.min15mincnt
infoname=["min1info"]
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option("display.max_colwidth", 100)
pd.set_option('display.width', 10000)

def calcw_random(dr, min1i, alpha=None, cnt=10, money=10000, tratio=0.05, lastbookw=None, money_limit=10000000):
    alpha=(np.random.randint(0, 2, [dm.dr["ban_symbols_at_flag"].shape[0]])*2-1).astype(float)
    alpha[dm.dr["ban_symbols_at_flag"]]=np.nan
    money_flag=(dm.dr["min1info_money"][min1i-5:min1i].mean(axis=0)>money_limit/1440)
    
    alpha[~money_flag]=np.nan
    alpha[~np.isfinite(alpha)]=0
    
    long=alpha>0
    short=alpha<0
    w=np.zeros(alpha.shape)
    avg_money=money/(np.abs(alpha)>0).sum()
    w[long]=alpha[long]*avg_money
    w[short]=-1.0*alpha[short]*avg_money
    return w, long, short

def calcw(dr, min1i, alpha, cnt=10, money=10000, tratio=0.05, lastbookw=None, money_limit=10000000):
    alpha=alpha.copy()
    alpha[~np.isfinite(alpha)]=0
    argsort=alpha.argsort()
    alpha[argsort[cnt:-cnt]]=0
    long=alpha>0
    short=alpha<0
    w=np.zeros(alpha.shape)
    w[long]=alpha[long]/alpha[long].sum()*money
    w[short]=-1.0*alpha[short]/alpha[short].sum()*money
    return w, long, short
    
def calcwtopk(dr, min1i, alpha, cnt=10, money=10000, tratio=0.05, lastbookw=None,
              money_limit=10000000):
    alpha=alpha.copy()
    # alpha[dm.dr["ban_symbols_at_flag"]]=np.nan
    # money_flag=(dm.dr["min1info_money"][min1i-5:min1i].mean(axis=0)>money_limit/1440)
    # alpha[~money_flag]=np.nan
    # alpha=alpha-np.median(alpha[np.isfinite(alpha)])
    alpha[~np.isfinite(alpha)]=0
    argsort=alpha.argsort()
    alpha[argsort[cnt:-cnt]]=0
    
    long=alpha>0
    short=alpha<0
    w=np.zeros(alpha.shape)
    w[long]=money/long.sum()
    w[short]=-1.0*money/short.sum()
    return w, long, short

def calcwtopkw(dr, min1i, alpha, cnt=10, money=10000, 
               tratio=0.05, lastbookw=None, topk_ratio=0.5, money_limit=10000000):
    w1, long1, short1 = calcw(dr, min1i, alpha, cnt=cnt, money=money, 
                              tratio=tratio, lastbookw=lastbookw, money_limit=money_limit)
    w2, long2, short2 = calcwtopk(dr, min1i, alpha, cnt=cnt, money=money,
                                  tratio=tratio, lastbookw=lastbookw, money_limit=money_limit)
    w=w1*(1-topk_ratio)+w2*topk_ratio
    long=w>0
    short=w<0

    return w, long, short

def calcwtopkliqnew(dr, min1i, alpha, cnt=10, money=10000, tratio=0.05, lastbookw=None):
    alpha=alpha-np.nanmean(alpha)
    alpha[~np.isfinite(alpha)]=0
    argsort=alpha.argsort()


    w=np.zeros(alpha.shape)
    longmoney=money
    cnt=alpha.shape[0]-1
    delta=60
    scale=3
    while longmoney>0 and cnt >0:
        sidx=argsort[cnt]
        hismoney=min(dr["min1info_money"][min1i-delta:min1i, sidx].mean(axis=0),
                     np.median(dr["min1info_money"][min1i-delta:min1i, sidx], axis=0))
        w[sidx] = min(money/scale, hismoney*tratio)
        w[sidx]=min(w[sidx], longmoney)
        longmoney-=w[sidx]
        cnt-=1
    shortmoney=money
    cnt=0
    while shortmoney>0 and cnt < alpha.shape[0]:
        sidx=argsort[cnt]
        hismoney=min(dr["min1info_money"][min1i-delta:min1i, sidx].mean(axis=0),
                     np.median(dr["min1info_money"][min1i-delta:min1i, sidx], axis=0))
        w[sidx] = min(money/scale, hismoney*tratio)
        w[sidx]=min(w[sidx], shortmoney)
        shortmoney-=w[sidx]
        w[sidx]*=-1.0
        cnt+=1
    long=w>0
    short=w<0
    
    return w, long, short

def round_with(a, min_delta=1000):
    v=round(a/min_delta)*min_delta
    return v

def calcwtopkliqV3(dr, min1i, alpha, lastw=None, cnt=10, money=10000, 
                   tratio=0.05, lastbookw=None, ratio_limit=100, scale=3, money_limit=10000000,
                   top_limit=30, min_delta=1000):
    alpha=alpha-np.nanmean(alpha)
    alpha[~np.isfinite(alpha)]=0
    money_flag=(dm.dr["min1info_money"][min1i-1440:min1i].mean(axis=0)>money_limit/1440)
    alpha[~money_flag]=0
    argsort=alpha.argsort()
    rank=argsort.argsort()
    w=np.zeros(alpha.shape)
    longmoney=money
    shortmoney=money
    if not lastbookw is None:
        sids_num=lastbookw.shape[0]
        for idx in range(sids_num):
            bookw=lastbookw[idx]
            if bookw > 1:
                ratio=np.clip( 1.0-(sids_num-top_limit-rank[idx])/ratio_limit, 0.0, 1.0) 
                w[idx]=round_with(bookw*ratio, min_delta)
                longmoney-=w[idx]
            elif bookw < -1:
                ratio=np.clip( 1.0-(rank[idx]-top_limit)/ratio_limit, 0.0, 1.0) 
                w[idx]=round_with(bookw*ratio, min_delta)
                shortmoney+=w[idx]

    cnt=alpha.shape[0]-1
    delta=30
    while longmoney>0 and cnt >0:
        sidx=argsort[cnt]
        hismoney=min(dr["min1info_money"][min1i-delta:min1i, sidx].mean(axis=0),
                     np.median(dr["min1info_money"][min1i-delta:min1i, sidx], axis=0))
        new_bookw = min(money/scale, hismoney*tratio)
        new_bookw=min(new_bookw, longmoney)
        diff=new_bookw-w[sidx]
        diff = round_with(diff, min_delta)
        if diff > 0:
            w[sidx]+=diff
            longmoney-=diff
        cnt-=1

    cnt=0
    while shortmoney>0 and cnt < alpha.shape[0]:
        sidx=argsort[cnt]
        hismoney=min(dr["min1info_money"][min1i-delta:min1i, sidx].mean(axis=0),
                     np.median(dr["min1info_money"][min1i-delta:min1i, sidx], axis=0))
        new_bookw = min(money/scale, hismoney*tratio)
        new_bookw=min(new_bookw, shortmoney)
        diff=-new_bookw-w[sidx]
        diff = round_with(diff, min_delta)
        if diff < 0:
            w[sidx]+=diff
            shortmoney+=diff
        cnt+=1
    long=w>0
    short=w<0
    
    return w, long, short

def alphaavg(alphas):
    return alphas[-1]+alphas[-2]*0.5
def alphaavg2(alphas):
    return alphas[-1]

def run_alpha(delta, alphafunc, tsf=None, start=20231030000000, end=20241221203000, 
              money=30000, tratio=0.1, alphaavg=alphaavg, delaymin=5, endflag=False,
              ban_hours=[],
              aftermin=1,
              singalDelayMin = 0,
              calcw=calcwtopkliqnew, ban_symbols=[], lb_ratio=0.0,
              path="/home/crypto/cryptoqt/smlp/model_states/infmodel/tsmlp15/res"):

    dr=dm.dr
    dr["uid"]=dr["sids"]
    years, months, lrets, srets, mrets = [], [], [], [], []
    tms, days, tvrs=[], [], []
    ics, sa_lrets, sa_srets = [], [], []
    realmeanrets=[]
    ban_flag=np.zeros(dr["uid"].shape)
    for symbol in ban_symbols:
        ban_flag[dr['sidmap'][symbol]]=1
    end=dr["min1info_tm"].tolist().index(end)
    start=dr["min1info_tm"].tolist().index(start)
    dfs={}
    lastbookw=None
    mis, miics=[], []
    # dtrades=loadtradeinfo(dr)
    alphas=deque(maxlen=100)
    for min1i in range(start+singalDelayMin, end, delta):
        tm=dr["min1info_tm"][min1i]
        if tm == 20241029000000:
            a=0
        if int((tm % 1000000)/10000) in ban_hours:
            continue
        fname=path+"/"+str(tm)+"_pred.csv"
        lasteddownload=dmap.gettmsize()
        if not (os.path.exists(fname) and lasteddownload>=min1i+delta+delaymin) :
            if endflag:
                break
        while not (os.path.exists(fname) and lasteddownload>=min1i+delta+delaymin):
            time.sleep(2)
            lasteddownload=dmap.gettmsize()
            continue
        
        alpha, alphamin1=alphafunc(dr, min1i)
        alpha[ban_flag.astype(bool)]=np.nan
        alphas.append(alpha)
        if len(alphas)>6:
            alpha=alphaavg(alphas)

        bookw, long, short = calcw(dr, min1i, alpha.copy(), money=money, tratio=tratio, lastbookw=lastbookw)
        if not lastbookw is None:
            bookw=bookw*(1-lb_ratio)+lastbookw*lb_ratio
        endp=np.nanmean(dr["min1info_vwap"][min1i+delta:min1i+delta+delaymin], axis=0)
        startp=np.nanmean(dr["min1info_vwap"][min1i:min1i+delaymin], axis=0)
        if not alphamin1 is None:
            startp_aftermin=np.nanmean(dr["min1info_vwap"][min1i+aftermin:min1i+aftermin+delaymin], axis=0)
            longAfterFlag = (long & (alphamin1 < -0.1)) 
            shortAfterFlag = (short & (alphamin1 > 0.1))
            startp[longAfterFlag | shortAfterFlag] = startp_aftermin[longAfterFlag | shortAfterFlag]
        realret=(endp/startp-1.0)
        #排除掉异常值(NaN等)
        valid=np.isfinite(alpha)&np.isfinite(realret)
        #异常值处理为0
        realret[~np.isfinite(realret)]=0
        # pricevalid=

        #求收益均值(nanmean是mean的增强型,忽略NaN)
        realretmean=np.nanmean(realret[valid]) if valid.sum()>0 else 0
        #stopLoss
        stopLossLimit = 0.0125
        maxprice=np.max(dr["min1info_high"][min1i+delaymin:min1i+delta], axis=0)
        minPrice=np.min(dr["min1info_low"][min1i+delaymin:min1i+delta], axis=0)
        stopLongLossFlag = long & ((minPrice/startp-1.0) < -stopLossLimit)
        stopShortLossFlag = short & ((maxprice/startp-1.0) > stopLossLimit)
        realret[stopLongLossFlag]=-stopLossLimit
        realret[stopShortLossFlag]=stopLossLimit
        
        #收益X权重
        ret=bookw*realret
        
        lret=ret[long].sum()
        sret=ret[short].sum()

        df=pd.DataFrame()
        df["sid"]=dr["uid"]
        df["bookw"]=bookw
        df["ret"]=ret
        df["long"]=long
        df["short"]=short
        df["bprice"]=np.nanmean(dr["min1info_vwap"][min1i:min1i+delaymin], axis=0)
        df["eprice"]=np.nanmean(dr["min1info_vwap"][min1i+delta:min1i+delta+delaymin], axis=0)
        df["money"]=dr["min1info_money"][min1i:min1i+delta].mean(axis=0)
        df["pred"]=alpha.copy()
        # print("\n\ninfo tm:", tm, "ret:", lret+sret)
        # print("long ret:", lret, "\n", df[long])
        # print("short ret:", sret, "\n", df[short])
        dfs[tm]=df
        lastbookw=bookw
        
        if dr["min1info_tm"][min1i-delta] in dfs:
            lastdf=dfs[dr["min1info_tm"][min1i-delta]]
            lastw=dfs[dr["min1info_tm"][min1i-delta]]["bookw"]
        else:
            lastw=np.zeros(bookw.shape)
        tvr=np.abs(bookw-lastw).sum()
        months.append(int(tm/100000000))
        days.append(int(tm/1000000))
        tms.append(tm)
        lrets.append(lret)
        srets.append(sret)
        tvrs.append(tvr)
        
        mret=(lret+sret)/money/2*10000-tvr/money*1.5
        mrets.append(mret)
        
        sa_lrets.append(lret-realretmean*money-tvr/money/2)
        sa_srets.append(sret+realretmean*money-tvr/money/2)

        if ~np.isfinite(sa_lrets[-1]):
            a=0
        #alpha和realret的相关性
        ic=np.corrcoef(alpha[valid], realret[valid])[0,1]
        ics.append(ic)
        realmeanrets.append(realretmean)

    #打印结果构建
    df=pd.DataFrame()
    df["ic"]=np.array(ics)
    scale=money/10000.0
    df["ret"]=(np.array(lrets)+np.array(srets))/scale/2
    df["lret"]=np.array(lrets)/scale/2
    df["sret"]=np.array(srets)/scale/2
    df["tvr"]=np.array(tvrs)/scale/2
    df["mret"] = np.array(mrets)
    # df["sa_lret"]=np.array(sa_lrets)/scale
    # df["sa_sret"]=np.array(sa_srets)/scale
    # df["realmeanret"]=np.array(realmeanrets)
    df["month"]=np.array(months)
    df["day"]=np.array(days)
    df["tm"]=np.array(tms)
    # df["tmidx"]=np.array([int(tm%1000000/1000) for tm in tms])
    stats=df.groupby("day").mean()
    stats["ic"]=df.groupby("day").mean()["ic"]
    
    xsdate = [datetime.datetime.strptime(str(d), '%Y%m%d').date() for d in days]
    # plt.plot(xsdate, np.cumsum(sa_lrets),color='r', linewidth=1)
    # plt.plot(xsdate, np.cumsum(sa_srets),color='b', linewidth=1)
    plt.plot(xsdate, np.cumsum(mrets),color='y')
    plt.tick_params(axis='both',which='both',labelsize=10)
    plt.gcf().autofmt_xdate()  # 自动旋转日期标记
    plt.show()
    # plt.plot(xsdate, ics)
    # plt.plot(xsdate, tools.moving_average(np.array(ics),10))
    # plt.tick_params(axis='both',which='both',labelsize=10)
    # plt.gcf().autofmt_xdate()  # 自动旋转日期标记
    # plt.show()
    print(stats)
    
    return df

def readcsv_v2avg(dr, min1i, path="/home/crypto/smlpv2/cryptoqt/smlp/model_states/infmodel/tsmlpv215/res",
                  pathmin1="/home/crypto/smlpv2/cryptoqt/smlp/model_states/infmodel/tsmlpv215/res",
                  fields = ["pred0", "pred1", "pred2"]):
    di=int(min1i/conts.daymincnt)
    h4i=int(min1i/conts.h4mincnt)
    h1i=int(min1i/conts.h1mincnt)
    min15i=int(min1i/conts.min15mincnt) 
    fname=path+"/"+str(dr["min1info_tm"][min1i])+"_pred.csv"
    df=pd.read_csv(fname)
    alpha=np.zeros(df["pred0"].shape)
    for field in fields:
        alpha+=tools.npnorm(df[field].to_numpy())
    alphamin1rank=None
    fnamemin1=pathmin1+"/"+str(dr["min1info_tm"][min1i])+"_pred.csv"
    dfmin1=pd.read_csv(fnamemin1)
    alphamin1=np.zeros(df["pred0"].shape)
    for field in fields:
        alphamin1+=tools.npnorm(dfmin1[field].to_numpy())
    nanflag=~np.isfinite(alphamin1)
    alphamin1[nanflag]=0
    alphamin1rank = rankdata(alphamin1)
    alphamin1rank = alphamin1rank/alphamin1.shape[0]*2-1.0
    alphamin1rank[nanflag]=0
    scipy.stats.pearsonr(alphamin1, alphamin1rank)
    return alpha, alphamin1rank
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test for argparse', formatter_class=argparse.RawTextHelpFormatter)
    # parser.add_argument('--start_date', help='start_date', default=20231001000000, type=int)
    # parser.add_argument('--start_date', help='start_date', default=20240101000000, type=int)
    parser.add_argument('--start_date', help='start_date', default=20240831000000, type=int)
    # parser.add_argument('--end_date', help='end_date', default=20241030000000, type=int)
    parser.add_argument('--end_date',  help='end_date', default=20250815120000, type=int)
    args = parser.parse_args()
    dm.init()
    dr=dm.dr
    
    
    ban_symbols=dr["ban_symbols_at"]
    
    path="/home/nb/v1/cryptoqt/smlp/model_states/infmodel/nsmlpv260/res"
    singalDelayMin = 0
    
    # path="/home/nb/v1/cryptoqt/smlp/model_states/infmodel/nsmlpv260_2/res"
    # singalDelayMin = 30
    pathmin1 = "/home/nb/v1/cryptoqt/smlp/model_states/infmodel/tsmlpv25/res"
    #p1 20-10-1-1-0.2
    #p2 50-20-1-0-0.1
    aa=run_alpha(int(conts.h1mincnt/1), partial(readcsv_v2avg,
            path=path, 
            pathmin1=pathmin1,
            fields=["pred2"]),
            path=path,
            tsf=None,endflag=True, 
            delaymin=1,
            aftermin = 0,
            alphaavg=alphaavg2,
            singalDelayMin = singalDelayMin,
            # calcw=calcw,
            # calcw=partial(calcwtopk, cnt=10), 
            calcw=partial(calcwtopkliqV3, ratio_limit=50, scale=2, money_limit=10000000, 
                          top_limit=20, min_delta=1000), 
            ban_symbols=ban_symbols,
            # ban_hours=dr["ban_hours_less"],
              start=args.start_date, end=args.end_date, 
              money=30000, tratio=0.1, lb_ratio=0.0)
    # print("\nsummary:", args.start_date, "~", args.end_date, "sum ret:", aa["ret"].sum())
    
    stats=aa.groupby("month").mean()
    stats["ic"]=aa.groupby("month").mean()["ic"]
    print("\n----------month  performance ----------------")
    print(stats)
    
    stats=aa.groupby("day").mean()
    tools.plotg(np.cumsum(stats["mret"].values))
    
    print("\n----------last 20  signal  performance ----------------")
    print(aa[-20:].set_index("tm"))
    print(f"meanret:{aa['mret'].mean()} std:{aa['mret'].std()} sharp:", aa["mret"].mean()/aa["mret"].std())
    


    # xx=0
    
    
    
    
