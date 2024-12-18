#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 12:40:22 2024

@author: ld
"""
import os
import pandas as pd
import numpy as np
import h5py
import time
import sys
import datetime
from collections import deque
sys.path.append('/home/crypto/sec_klines_env/')
import cryptoqt.data.updatedata as ud
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
daymincnt=conts.daymincnt
h4mincnt=conts.h4mincnt
h1mincnt=conts.h1mincnt
min15mincnt=conts.min15mincnt
infoname=["min1info"]
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option("display.max_colwidth", 100)
pd.set_option('display.width', 10000)
delaymin=5
def calcw(alpha):
    alpha[~np.isfinite(alpha)]=0
    long=alpha>0
    short=alpha<0
    w=np.zeros(alpha.shape)
    w[long]=alpha[long]/alpha[long].sum()
    w[short]=-1.0*alpha[short]/alpha[short].sum()
    return w, long, short
    
def calcwtopk(dr, min1i, alpha, cnt=10, money=10000, tratio=0.05):
    alpha=alpha-np.nanmean(alpha)
    alpha[~np.isfinite(alpha)]=0
    argsort=alpha.argsort()
    alpha[argsort[cnt:-cnt]]=0
    long=alpha>0
    short=alpha<0
    w=np.zeros(alpha.shape)
    w[long]=alpha[long]/alpha[long].sum()*money
    w[short]=-1.0*alpha[short]/alpha[short].sum()*money
    return w, long, short

def calcwtopkliq(dr, min1i, alpha, cnt=10, money=10000, tratio=0.05):
    alpha=alpha-np.nanmean(alpha)
    alpha[~np.isfinite(alpha)]=0
    argsort=alpha.argsort()


    w=np.zeros(alpha.shape)
    longmoney=money
    cnt=alpha.shape[0]-1
    delta=60
    scale=10
    while longmoney>0 and cnt >0:
        sidx=argsort[cnt]
        w[sidx] = min(money/scale, dr["min1info_money"][min1i-delta:min1i, sidx].mean(axis=0)*tratio)
        w[sidx]=min(w[sidx], longmoney)
        longmoney-=w[sidx]
        cnt-=1
    shortmoney=money
    cnt=0
    delta=15
    while shortmoney>0 and cnt < alpha.shape[0]:
        sidx=argsort[cnt]
        w[sidx] = min(money/scale, dr["min1info_money"][min1i-delta:min1i, sidx].mean(axis=0)*tratio)
        w[sidx]=min(w[sidx], shortmoney)
        shortmoney-=w[sidx]
        w[sidx]*=-1.0
        cnt+=1
    long=w>0
    short=w<0
    
    return w, long, short

def calcwtopkliqnew(dr, min1i, alpha, cnt=10, money=10000, tratio=0.05):
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

def calcwtopkliqV3(dr, min1i, alpha, lastw=None, cnt=10, money=10000, 
                   tratio=0.05, tratioPos=0.2):
    alpha=alpha-np.nanmean(alpha)
    alpha[~np.isfinite(alpha)]=0
    argsort=alpha.argsort()

    w=np.zeros(alpha.shape)
    longmoney=money
    cnt=alpha.shape[0]-1
    delta=60
    scale=2
    if lastw is None:
        lastw=np.zeros(alpha.shape)
    while longmoney>0 and cnt >0:
        sidx=argsort[cnt]
        hismoney=min(dr["min1info_money"][min1i-delta:min1i, sidx].mean(axis=0),
                     np.median(dr["min1info_money"][min1i-delta:min1i, sidx], axis=0))
        w[sidx] = min(hismoney*tratioPos, min(money/scale, lastw[sidx]+hismoney*tratio))
        w[sidx]=min(w[sidx], longmoney)
        longmoney-=w[sidx]
        cnt-=1
    shortmoney=money
    cnt=0
    while shortmoney>0 and cnt < alpha.shape[0]:
        sidx=argsort[cnt]
        hismoney=min(dr["min1info_money"][min1i-delta:min1i, sidx].mean(axis=0),
                     np.median(dr["min1info_money"][min1i-delta:min1i, sidx], axis=0))
        w[sidx] = min(hismoney*tratioPos, min(money/scale, lastw[sidx]+hismoney*tratio))
        w[sidx]=min(w[sidx], shortmoney)
        shortmoney-=w[sidx]
        w[sidx]*=-1.0
        cnt+=1
    long=w>0
    short=w<0
    
    return w, long, short

def alphaavg(alphas):
    return alphas[-1]+alphas[-2]*0.5
def alphaavg2(alphas):
    return alphas[-1]

def run_alpha(delta, alphafunc, tsf=None, start=20231030000000, end=20241221203000, 
              money=30000, tratio=0.1, alphaavg=alphaavg, delaymin=delaymin, endflag=False,
              calcw=calcwtopkliqnew,
              path="/home/crypto/cryptoqt/smlp/model_states/infmodel/tsmlp15/res"):
    ud.readuniverse(ud.g_data)
    lasteddownload=dmap.read_memmap(ud.g_data)
    dr=ud.g_data
    dr["uid"]=dr["sids"]
    years, months, lrets, srets = [], [], [], []
    tms, days, tvrs=[], [], []
    ics, sa_lrets, sa_srets = [], [], []
    realmeanrets=[]
    end=dr["min1info_tm"].tolist().index(end)
    start=dr["min1info_tm"].tolist().index(start)
    dfs={}
    lastbookw=None
    mis, miics=[], []
    # dtrades=loadtradeinfo(dr)
    alphas=deque(maxlen=100)
    for min1i in range(start, end, delta):
        tm=dr["min1info_tm"][min1i]

        fname=path+"/"+str(tm)+"_pred.csv"
        lasteddownload=dmap.gettmsize()
        if not (os.path.exists(fname) and lasteddownload>min1i+delta+delaymin) :
            if endflag:
                break
        while not (os.path.exists(fname) and lasteddownload>min1i+delta+delaymin):
            time.sleep(2)
            lasteddownload=dmap.gettmsize()
            continue
        
        alpha=alphafunc(dr, min1i)
        alphas.append(alpha)
        if len(alphas)>6:
            alpha=alphaavg(alphas)

        bookw, long, short = calcw(dr, min1i, alpha.copy(), money=money, tratio=tratio)

        # endp=(dr["min1info_vwap"][min1i+delta:min1i+delta+delaymin]*dr["min1info_volume"][min1i+delta:min1i+delta+delaymin]
        #       ).sum(axis=0)/dr["min1info_volume"][min1i+delta:min1i+delta+delaymin].sum(axis=0)
        # startp=(dr["min1info_vwap"][min1i:min1i+delaymin]*dr["min1info_volume"][min1i:min1i+delaymin]
        #       ).sum(axis=0)/dr["min1info_volume"][min1i:min1i+delaymin].sum(axis=0)
        endp=dr["min1info_vwap"][min1i+delta:min1i+delta+delaymin].mean(axis=0)
        startp=dr["min1info_vwap"][min1i:min1i+delaymin].mean(axis=0)
        realret=(endp/startp-1.0)
        #排除掉异常值(NaN等)
        valid=np.isfinite(alpha)&np.isfinite(realret)
        #异常值处理为0
        realret[~np.isfinite(realret)]=0
        # pricevalid=

        #求收益均值(nanmean是mean的增强型,忽略NaN)
        realretmean=np.nanmean(realret[valid]) if valid.sum()>0 else 0
        #收益X权重
        ret=bookw*realret
        lret=ret[long].sum()
        sret=ret[short].sum()

        if (not lastbookw is None) and (not tsf is None):
            trademoney=bookw-lastbookw
            tradesignal=tsf(dr, min1i)
            tsfarg=np.zeros(tradesignal.shape)
            tmpv=tradesignal[np.isfinite(tradesignal)]
            tmpv=tmpv.argsort().argsort()/tmpv.shape[0]
            tsfarg[np.isfinite(tradesignal)]=tmpv
            
            tvalid=np.isfinite(tradesignal)&(np.abs(trademoney)>0.0001)
            targetmin=5
            retmi=dr["min1info_vwap"][min1i+targetmin]/dr["min1info_vwap"][min1i]-1.0
            tradetmidx=np.zeros(tradesignal.shape)

            lf=trademoney[tvalid]>0
            sf=trademoney[tvalid]<0
            tmpv=np.zeros(trademoney[tvalid].shape)
            tmpv[lf]=((4-tsfarg[tvalid]*5).astype(int)[lf])
            tmpv[sf]=(tsfarg[tvalid]*5).astype(int)[sf]
            tradetmidx[tvalid]=tmpv
            tradetmidx[~tvalid]=0
            # tradetmidx[:]=1
            tradetmidx=tradetmidx.astype(int)
            mi=(dr["min1info_vwap"][min1i:min1i+targetmin][(tradetmidx, np.arange(tradetmidx.shape[0]))]/startp-1.0)*trademoney
            mi[~np.isfinite(mi)]=0
            tsv=np.isfinite(tradesignal)&np.isfinite(retmi)
            miic=np.corrcoef(tradesignal[tsv], retmi[tsv])[0,1]
        else:
            mi=np.zeros(bookw.shape)
            miic=0

        df=pd.DataFrame()
        df["sid"]=dr["uid"]
        df["bookw"]=bookw
        df["ret"]=ret
        df["mi"]=mi
        df["long"]=long
        df["short"]=short
        df["bprice"]=dr["min1info_vwap"][min1i:min1i+delaymin].mean(axis=0)
        df["eprice"]=dr["min1info_vwap"][min1i+delta:min1i+delta+delaymin].mean(axis=0)
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
        mis.append(mi.sum())

        sa_lrets.append(lret-realretmean*money)
        sa_srets.append(sret+realretmean*money)

        if ~np.isfinite(sa_lrets[-1]):
            a=0
        #alpha和realret的相关性
        ic=np.corrcoef(alpha[valid], realret[valid])[0,1]
        ics.append(ic)
        miics.append(miic)
        realmeanrets.append(realretmean)

    #打印结果构建
    df=pd.DataFrame()
    df["ic"]=np.array(ics)
    scale=money/10000.0
    df["ret"]=(np.array(lrets)+np.array(srets))/scale
    df["mi"]=np.array(mis)/scale
    df["miic"]=np.array(miics)
    df["lret"]=np.array(lrets)/scale
    df["sret"]=np.array(srets)/scale
    df["tvr"]=np.array(tvrs)/scale
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
    # plt.plot(xsdate, np.cumsum(sa_lrets),color='r')
    # plt.plot(xsdate, np.cumsum(sa_srets),color='b')
    # plt.tick_params(axis='both',which='both',labelsize=10)
    # plt.gcf().autofmt_xdate()  # 自动旋转日期标记
    # plt.show()
    # plt.plot(xsdate, ics)
    # plt.plot(xsdate, tools.moving_average(np.array(ics),10))
    # plt.tick_params(axis='both',which='both',labelsize=10)
    # plt.gcf().autofmt_xdate()  # 自动旋转日期标记
    # plt.show()
    print(stats)
    
    return df
  
def calcdiff(dr, dtrades, start=20240702053000, end=20240702053800):
    delta=1
    start=dr["min1info_tm"].tolist().index(start)
    end=dr["min1info_tm"].tolist().index(end)
    lsum, ssum=[], []
    ldp, sdp=[], []
    lm, sm=[],[]
    rets=[]
    stats=pd.DataFrame(columns=['tm','ret', 'retv', 'lam', 'sam', 'difflm', 'diffsm'])
    
    for min1i in range(start, end, delta):
        tm=dr["min1info_tm"][min1i]
        if tm not in dtrades:
            print("miss:", tm)
            stats.loc[len(stats.index)] =[int(tm), 0, 0, 0, 0, 0, 0]
            continue
        a=dtrades[tm]
        la=a[a["long"]]
        sa=a[a["short"]]
        difflm=la["dp"]*la["money"]*-1
        diffsm=sa["dp"]*sa["money"]
        retsum=la["ret"].sum()+sa["ret"].sum()
        retvsum=la["retv"].sum()+sa["retv"].sum()
        lsum.append(difflm.sum())
        ssum.append(diffsm.sum())
        lm.append(la["money"].sum())
        sm.append(sa["money"].sum())
        ldp.append((la["dp"]*la["money"]).sum()/la["money"].sum())
        sdp.append((sa["dp"]*sa["money"]).sum()/sa["money"].sum())
        print(tm, retsum, retvsum, "money:", la["money"].sum(),sa["money"].sum()
              , difflm.sum(), diffsm.sum()
              )
        stats.loc[len(stats.index)] =[int(tm), retsum, retvsum, la["money"].sum(),sa["money"].sum(), difflm.sum(), diffsm.sum()]
    print("sum:", sum(lsum), sum(ssum), sum(lsum)/sum(lm), sum(ssum)/sum(sm), sum(lm), sum(sm))
    stats["tm"]=stats["tm"].astype(int)
    return stats

def loadtradeinfo(dr, delta=15, delaymin=5, path='/home/crypto/cryptoqt/导出历史成交记录.xlsx'):
    # info=pd.read_csv(path, encoding='gbk')
    info = pd.read_excel(path)
    dtrades={}
    tmlist=dr["min1info_tm"].tolist()
    sidlist=dr["sids"].tolist()
    for index, item in info.iterrows():
        tm=item['时间(UTC)']
        tm=datetime.datetime.strptime(tm, "%Y-%m-%d %H:%M:%S")+datetime.timedelta(hours=8)
        tm=int(tm.strftime('%Y%m%d%H%M%S'))
        tm=int(tm/10000)*10000+int((tm%10000)/100)*100
        sid=item['合约']
        sidx=sidlist.index(sid)
        dt=item['方向']
        price=item['价格']
        volume=item['数量']
        money=item['成交额']
        fee=item['手续费']
        ret=item['已实现盈亏']
        if tm not in dtrades:
            trade = np.zeros((len(sidlist), 5))
            dtrades[tm]=trade
        else:
            trade=dtrades[tm]
        trade[sidx,0]=1 if dt=='买入' else -1
        trade[sidx,1]+=money
        trade[sidx,2]+=volume
        trade[sidx,3]+=ret
    
    for key in dtrades:
        item=dtrades[key]
        long=item[:,0]>0
        short=item[:,0]<0
        money=item[:,1]
        volume=item[:,2]
        ret=item[:,3]
        vprice=money/volume
        df=pd.DataFrame()
        df["sid"]=np.array(sidlist)
        df["long"]=long
        df["short"]=short
        df["vprice"]=vprice
        df["volume"]=volume
        df["money"]=money
        
        df["money"][short]=-1.0*df["money"][short]
        
        min1i=tmlist.index(key)
        vwapmin=min1i-min1i%15
        # df["vwap"]=dr["min1info_vwap"][vwapmin:vwapmin+delaymin].mean(axis=0)
        df["vwap"]=(dr["min1info_money"][vwapmin:vwapmin+delaymin]).sum(axis=0)/dr["min1info_volume"][vwapmin:vwapmin+delaymin].sum(axis=0)
        # futervwap=dr["min1info_money"][vwapmin+delta:vwapmin+delta+delaymin].sum(axis=0)/dr["min1info_volume"][vwapmin+delta:vwapmin+delta+delaymin].sum(axis=0)
        # df["ret"]=(futervwap/vprice-1.0)*money
        # df["retv"]=(futervwap/df["vwap"]-1.0)*money
        df["diff_money"]=df["volume"]*df["vwap"]*np.sign(df["money"])-df["money"]
        # df["retv"]=(futervwap/df["vwap"]-1.0)
        # df["dp"]=(df["vprice"]/df["vwap"]-1.0)
        dtrades[key]=df
    return dtrades
            

def calcdiff2(dr, dtrades, start=20240810100000, end=20240810164500):
    delta=15
    start=dr["min1info_tm"].tolist().index(start)
    end=dr["min1info_tm"].tolist().index(end)
    lsum, ssum=[], []
    ldp, sdp=[], []
    lm, sm=[],[]
    rets=[]
    stats=pd.DataFrame(columns=['sid', 'l_diff_money', 'l_money', 's_diff_money', 's_money', 'vwap'])
    merge_dfs=[]
    for min1i in range(start, end, delta):
        merge_df=pd.DataFrame(columns=['tm','l_diff_money', 'l_money', 's_diff_money', 's_money', 'vwap'])
        for idx in range(6):
            tm=dr["min1info_tm"][min1i+idx]
            if tm not in dtrades:
                print("miss:", tm)
                continue
            a=dtrades[tm]
            long=a["money"]>0
            short=a["money"]<0
            l_money=a["money"].fillna(0)
            l_money[short]=0
            l_diff_money=a["diff_money"].fillna(0)
            l_diff_money[short]=0
            s_money=a["money"].fillna(0)
            s_money[long]=0
            s_diff_money=a["diff_money"].fillna(0)
            s_diff_money[long]=0
            if idx==0:
                merge_df["sid"]=a["sid"]
                merge_df["tm"][:]=tm
                merge_df["l_money"]=l_money
                merge_df["l_diff_money"]=l_diff_money
                merge_df["s_money"]=s_money
                merge_df["s_diff_money"]=s_diff_money
                merge_df["vwap"]=a["vwap"]
                merge_df["volume"]=a["volume"]
            else:
                merge_df["l_money"]+=l_money
                merge_df["l_diff_money"]+=l_diff_money
                merge_df["s_money"]+=s_money
                merge_df["s_diff_money"]+=s_diff_money
                merge_df["volume"]+=a["volume"]
                
        if  stats.shape[0]==0:
            stats["sid"]=merge_df["sid"]
            stats["l_money"]=merge_df["l_money"]
            stats["l_diff_money"]=merge_df["l_diff_money"]
            stats["s_money"]=merge_df["s_money"]
            stats["s_diff_money"]=merge_df["s_diff_money"]
            stats["vwap"]=merge_df["vwap"]
            stats["volume"]=merge_df["volume"]
        else:
            stats["l_money"]+=merge_df["l_money"]
            stats["l_diff_money"]+=merge_df["l_diff_money"]
            stats["s_money"]+=merge_df["s_money"]
            stats["s_diff_money"]+=merge_df["s_diff_money"]
            stats["volume"]+=merge_df["volume"]
        merge_df["l_dp"]=(merge_df["l_diff_money"]/merge_df["l_money"])*10000.0
        merge_df["s_dp"]=(merge_df["s_diff_money"]/merge_df["s_money"])*10000.0
        merge_dfs.append(merge_dfs)
    # stats["l_dp"]=(stats["l_diff_money"]/stats["l_money"])*10000.0
    # stats["s_dp"]=(stats["s_diff_money"]/stats["s_money"])*10000.0
    return stats
        
    
    
if __name__ == "__main__":
    ud.readuniverse(ud.g_data)
    lasteddownload=dmap.read_memmap(ud.g_data)
    dr=ud.g_data
    dr["uid"]=dr["sids"]
    
    dtrades=loadtradeinfo(dr, delaymin=5)
    stats=calcdiff2(dr, dtrades, start=20240824130000, end=20240824141500)
    a=stats[(stats["l_money"]>0)|(stats["s_money"]<0)]
    print("cost:", a["l_diff_money"].sum()+a["s_diff_money"].sum(), 
          "all money:", (a["l_money"].sum()-a["s_money"].sum()))
    bp=(a["l_diff_money"].sum()+a["s_diff_money"].sum())/(a["l_money"].sum()-a["s_money"].sum())
    print(bp)
    
    tsf=partial(min5_alpha.readcsv_avg,
            path="/home/crypto/smlpv2/cryptoqt/smlp/model_states/infmodel/smlpv25/res/")
    # aa=run_alpha(int(conts.h1mincnt/12), partial(min5_alpha.readcsv_avg,
    #         path="/home/crypto/smlpv2/cryptoqt/smlp/model_states/infmodel/smlpv25/res/"),
    #               tsf=None,
    #           start=20240101000000, end=20240601000000, money=50000, tratio=0.1,
    #           path="/home/crypto/smlpv2/cryptoqt/smlp/model_states/infmodel/smlpv25/res/",
    #           delaymin=1)
    
    aa=run_alpha(int(conts.h1mincnt/4), partial(min15_alpha.readcsv_v2avg,
            path="/home/crypto/smlpv2/cryptoqt/smlp/model_states/infmodel/tsmlpv215/res"),
            path="/home/crypto/smlpv2/cryptoqt/smlp/model_states/infmodel/tsmlpv215/res",
            tsf=None,endflag=False, alphaavg=alphaavg2,calcw=calcwtopkliqnew,
              start=20240814000000, end=20250723110000, money=100000, tratio=1.0)
    
    print("\n----------last 20  signal  performance ----------------")
    print(aa[-20:].set_index("tm"))
    # print(aa)
    # aa=run_alpha(int(conts.h1mincnt/2), partial(min30_alpha.readcsvavg,
    #         path="/home/crypto/smlpv2/cryptoqt/smlp/model_states/infmodel/tsmlpv230/res"),
    #         path="/home/crypto/smlpv2/cryptoqt/smlp/model_states/infmodel/tsmlpv230/res",
    #         tsf=None,endflag=True, alphaavg=alphaavg,calcw=calcwtopkliqnew,
    #           start=20240701000000, end=20250722000000, money=30000, tratio=0.1)    

    
    # dtrades=loadtradeinfo(dr, delaymin=5)
    # stats=calcdiff(dr, dtrades, start=20240711000000, end=20240711122000)
    # run_alpha(conts.daymincnt, day_alpha.ndayr5)
    # run_alpha(conts.h4mincnt, min30_alpha.readcsv)
    # run_alpha(conts.h1mincnt*2, min30_alpha.readcsv)
    # run_alpha(conts.min15mincnt, min15_alpha.readcsv)

    # xx=0
    
    
    
    
