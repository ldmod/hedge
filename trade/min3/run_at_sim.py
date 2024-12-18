#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 12:40:22 2024

@author: ld
"""
import sys
import os
sys.path.append(os.path.abspath(__file__+"../../../../../"))
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
np.seterr(invalid='ignore')
import warnings
warnings.simplefilter("ignore", category=RuntimeWarning)
import h5py
import time
import datetime
from collections import deque
import cryptoqt.data.updatedata as ud
from functools import partial
import matplotlib.pyplot as plt
import cryptoqt.data.constants as conts
import cryptoqt.data.tools as tools
import cryptoqt.data.datammap as dmap
import cryptoqt.data.sec_klines.sec_klines as sk
import cryptoqt.alpha.secfeaextractor as secfea
import cryptoqt.trade.min3.gorders_at as gorders
import cryptoqt.trade.min3.view_order_info as voi
import yaml
import random
import cryptoqt.data.data_manager as dm
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option("display.max_colwidth", 100)
pd.set_option('display.width', 10000)
delaysec=2
longterm=30


def run_trade(delta, cfg, start=20240515190000, end=20240515190500, min_delta=5, 
              trademoney=30000, tratio=0.1, delaysec=delaysec, tradelen=300, 
              path="/home/crypto/smlp_prm/cryptoqt/prm/model_states/infmodel/prmv3/res/res.h5",
                book_path="/home/crypto/signal/predv25_1w/", scale=1.0,
                # book_path="/home/crypto/signal/predv25_at/", scale=0.5,
                # book_path="/home/crypto/signal/predv215_at/", scale=1.0,
              ):
    ud.readuniverse(ud.g_data)
    lasteddownload=dmap.read_memmap(ud.g_data)
    dr=ud.g_data
    dr["uid"]=dr["sids"]
    secdata=sk.SecondData('/home/crypto/proddata/crypto/secdata/', ud.g_data["sids"])
    secdr=secdata.secdr
    start_date, end_date=start, end
    years, months, lrets, srets = [], [], [], []
    tms, days, diffsvwaps=[], [], []
    end=sk.gtmidx_i(end)
    start=sk.gtmidx_i(start)
    dfs={}
    lastbookw=None
    mis, miics=[], []
    
    tradedelta=min_delta*60
    avgmoney=trademoney/(tradelen/5)
    allmb_money=np.zeros(dr["uid"].shape[0])
    allmb_volume=np.zeros(dr["uid"].shape[0])
    allms_money=np.zeros(dr["uid"].shape[0])
    allms_volume=np.zeros(dr["uid"].shape[0])
    mbrs=np.zeros(dr["uid"].shape[0])
    mbpds=np.zeros(dr["uid"].shape[0])
    msrs=np.zeros(dr["uid"].shape[0])
    mspds=np.zeros(dr["uid"].shape[0])
    spds=np.zeros(dr["uid"].shape[0])
    bpds=np.zeros(dr["uid"].shape[0])
    diffmsvwaps=np.zeros(dr["uid"].shape[0])
    diffmbvwaps=np.zeros(dr["uid"].shape[0])
    ms_tradecnt=np.zeros(dr["uid"].shape[0])
    mb_tradecnt=np.zeros(dr["uid"].shape[0])
    tradecnt=0
    gorder=gorders.GenerateOrders(secdata, cfg["gorder"])
    gorder.update_tratio(sk.gtm_i(start))
    records=[]
    trade_infos = []
    trade_info = None
    for tidx in range(start, end, tradedelta):
        tstart=tidx
        tend=tstart+tradelen
        #maker buy money
        mb_money=np.zeros(dr["uid"].shape[0])
        mb_volume=np.zeros(dr["uid"].shape[0])
        #maker sell money
        ms_money=np.zeros(dr["uid"].shape[0])
        ms_volume=np.zeros(dr["uid"].shape[0])
        diffmsvwap=np.zeros(dr["uid"].shape[0])
        diffmbvwap=np.zeros(dr["uid"].shape[0])
        mbcnt=np.zeros(dr["uid"].shape[0])
        mscnt=np.zeros(dr["uid"].shape[0])
        swmoney=np.zeros((dr["uid"].shape[0]))
        bwmoney=np.zeros((dr["uid"].shape[0]))
        smpairs={}
        
        fname=book_path+"/"+str(sk.gtm_i(tstart))+"_book.csv"
        cur_book=pd.read_csv(fname)
        # cur_book["bookw"].values[:]=np.abs(cur_book["bookw"].values[:])*\
        #     np.random.choice([-1,1],size=cur_book["bookw"].values[:].shape)
        fname=book_path+"/"+str(sk.gtm_i(tstart-min_delta*60))+"_book.csv"
        last_book=pd.read_csv(fname)
        # last_book["bookw"].values[:]=np.abs(last_book["bookw"].values[:])*\
        #     np.random.choice([-1,1],size=last_book["bookw"].values[:].shape)
        
        
        if not trade_info is None:
            last_cur_money = trade_info["cur_money"].values.copy()
        else:
            last_cur_money = last_book["bookw"]*scale
        trade_info=pd.DataFrame()
        trade_info["symbol"]=cur_book["sid"]
        trade_info["cur_money"]=last_cur_money.copy()
        trade_info["b_money"]=trade_info["cur_money"].copy()
        # 
        trade_info["t_money"]=cur_book["bookw"]*scale
        trade_info["f_money"]=trade_info["t_money"]-trade_info["b_money"]
        trade_info["order_volume"]=last_book["bookw"]*0
        trade_info["alpha"]=(cur_book["alpha"]-cur_book["alpha"].mean())/cur_book["alpha"].std()
        trade_info["alpha"]=trade_info["alpha"].fillna(0)
        
        for idx in range(dr["uid"].shape[0]):
            item=trade_info.iloc[idx]
            smpairs[dr["uid"][idx]]=(item["t_money"], item["b_money"], item["alpha"])
                
        tstart+=25
        # gorder.update_tratio(sk.gtm_i(tstart-60*10))
        gorder.restart(sk.gtm_i(tstart), sk.gtm_i(tend), delta, smpairs)
        for s1i in range(tstart, tend, delta):
            tm=sk.gtm_i(s1i)
            
            leftcnt=int((tend-s1i)/delta)
            
            smpairs={}
            for sidx,sid in enumerate(ud.g_data["sids"]):
                if abs(trade_info.iloc[sidx]["f_money"]) > 1:
                    smpairs[sid]=trade_info.iloc[sidx]["cur_money"]
                    
            # gorder.get_tr(tm, 60)
            orders=gorder.update_and_gorders(tm, smpairs)
            completed_sids=gorder.get_completed_symbols()
            ######### simulator trade ###########
            vwap, vf_y, svwap, shigh, slow, vf_ys, bvwap, bhigh, blow, vf_yb, smoney, bmoney, ysl, ybl=gorders.getyvalues(
                secdr, s1i, delta)
            vwap, _, svwap, shigh, slow, _, bvwap, bhigh, blow, _, smoney, bmoney, _, _=gorders.getyvalues(
                secdr, s1i, delta*gorder.cancel_delay)
            svf=vf_ys
            bvf=vf_yb
            leftcnt=int((tend-s1i)/delta)
            # limit_ratio=gorder.limit_ratio
            money_limit_ratio=cfg["money_limit_ratio"]
            trade_succ_ratio=cfg["trade_succ_ratio"]
            for order in orders:
                symbol,sidx, order_price, order_money, order_info = order
                if order_money > 0:
                    # if svf[sidx] & (order_price >= slow[sidx]) & (order_price<=bhigh[sidx]) & (random.random() < trade_succ_ratio):
                    if svf[sidx] & (order_price >= slow[sidx]) & (random.random() < trade_succ_ratio):
                        trade_money=min(abs(order_money), smoney[sidx]*money_limit_ratio)
                        trade_info.loc[sidx, "order_volume"]+=abs(trade_money/order_price)
                        diffmbvwap[sidx]+=np.abs((order_price/svwap[sidx]-1.0)*10000.0)
                        mbcnt[sidx]+=1
                        trade_info.loc[sidx, "cur_money"]+=trade_money
                else:
                    # if bvf[sidx] & (order_price <= bhigh[sidx]) & (order_price>=slow[sidx]) & (random.random() < trade_succ_ratio):
                    if bvf[sidx] & (order_price <= bhigh[sidx]) & (random.random() < trade_succ_ratio):
                        trade_money=min(abs(order_money), bmoney[sidx]*money_limit_ratio)
                        trade_info.loc[sidx, "order_volume"]+=abs(trade_money/order_price)
                        diffmsvwap[sidx]+=np.abs((order_price/bvwap[sidx]-1.0)*10000.0)
                        mscnt[sidx]+=1
                        trade_info.loc[sidx, "cur_money"]-=trade_money
            ######### simulator trade ###########
            
        ######### simulator alpha ###########
        min1i=dm.get_minidx(tools.tmi2u(sk.gtm_i(tstart)))
        delaymin=5
        endp=np.nanmean(dr["min1info_vwap"][min1i+min_delta:min1i+min_delta+delaymin], axis=0)
        startp=np.nanmean(dr["min1info_vwap"][min1i:min1i+delaymin], axis=0)
        realret=(endp/startp-1.0)
        realret[~np.isfinite(realret)]=0
        ret=trade_info["cur_money"]*realret
        tret=trade_info["t_money"]*realret
        trade_info["ret"]=ret
        trade_info["tret"]=tret
        trade_info["tp"]=np.sign((trade_info["t_money"]-trade_info["cur_money"]
                                    ).values)*realret*10000.0
        trade_info[["ret", "tret", "tp"]]=trade_info[["ret", "tret", "tp"]].round(3)
        ######### simulator alpha ###########
            
        print("complete:", sk.gtm_i(tstart), ret.sum())
        # gorder.save_order_info(sk.gtm_i(tstart)+2)
        order_info=gorder.get_order_info(sk.gtm_i(tstart))
        if not order_info is None:
            order_info=voi.process_order_data(order_info)
            allvwap=secdr["s1info_money"][tstart:tend].sum(axis=0)/secdr["s1info_volume"][tstart:tend].sum(axis=0)
            svwap=secdr["s1info_smoney"][tstart:tend].sum(axis=0)/secdr["s1info_svolume"][tstart:tend].sum(axis=0)
            bvwap=(secdr["s1info_money"][tstart:tend].sum(axis=0)-secdr["s1info_smoney"][tstart:tend].sum(axis=0))/\
                (secdr["s1info_volume"][tstart:tend].sum(axis=0)-secdr["s1info_svolume"][tstart:tend].sum(axis=0))
            order_data=order_info.drop(columns=["order_id", "curtm"])
            # lsw=0.5
            # order_data["ls_merge"]=order_data["lsr"]*lsw+order_data["ls_mom"]*(1-lsw)
            dfagg=order_data.groupby('symbol')
        
        trade_info["vprice"]=(trade_info["cur_money"]-trade_info["b_money"])/trade_info["order_volume"]
        trade_info["vwap"]=allvwap
        trade_info["cost"]=trade_info["order_volume"]*allvwap*np.sign(trade_info["f_money"])-(trade_info["cur_money"]-trade_info["b_money"])
        trade_info["cr"]=((trade_info["cur_money"]-trade_info["b_money"])/trade_info["f_money"]).round(2)
        
        # trade_info["day"]=np.full(ud.g_data["sids"].shape, int(sk.gtm_i(tstart)/1000000), dtype=int)
        trade_info=trade_info.set_index("symbol")
        
        def cosine(col1, col2, x):
            flag=(~(x[col1].isna() | x[col2].isna()))
            x=x[flag]
            inner_sum=(x[col1]*x[col2]).sum()
            inner_m=np.sqrt((x[col1]*x[col1]).sum())*np.sqrt((x[col2]*x[col2]).sum())
            return inner_sum/inner_m
        # sl_corr=dfagg[['dv','ysl']].apply(partial(cosine, 'dv','ysl')).round(2)
        # sl_corr=dfagg[['pred_merge','ysl']].apply(partial(cosine, 'pred_merge','ysl')).round(2)
        sl_corr=dfagg[['pred','ysl']].apply(partial(cosine, 'pred','ysl')).round(2)
        # sl_corr=dfagg[['ls_merge','ysl']].apply(partial(cosine, 'ls_merge','ysl')).round(2)
        trade_info=pd.merge(trade_info, sl_corr.rename('sl_corr'), how='left',left_index=True, right_index=True)
        a_shift=dfagg[["a_shift", "tr_ratio", "cr_avg"]].agg("mean").round(3)
        trade_info=pd.merge(trade_info, a_shift, how='left',left_index=True, right_index=True)
        trade_info["ntp"]=trade_info["tp"]
        trade_info.drop(columns=["tp"], inplace = True)

        old_trade_info = trade_info.copy()
        trade_info=trade_info[trade_info["f_money"].abs()>100]
        print( trade_info)
        
        complete_money=(trade_info["cur_money"]-trade_info["b_money"]).abs().sum()
        target_money=trade_info["f_money"].abs().sum()
        print("trade_info:", complete_money/target_money, target_money.round(2), complete_money.round(2),
              "cost:", trade_info["cost"].sum().round(2), "bp:", trade_info["cost"].sum()/complete_money*10000,
              "alpha:", trade_info["ret"].sum().round(2), "bp:", trade_info["ret"].sum()/complete_money*10000,
              "tret:", trade_info["tret"].sum().round(2),
              "sl_corr:", trade_info["sl_corr"].mean(),
              )
        records.append(dict(curtm=sk.gtm_i(tstart), target_money=target_money, 
                            complete_money=complete_money, cost=trade_info["cost"].sum(),
                            alpharet=trade_info["ret"].sum(), tret=trade_info["tret"].sum(),
                            sl_corr=trade_info["sl_corr"].mean(),
                            ))
        trade_infos.append(trade_info.reset_index())
        trade_info = old_trade_info
        
    print("cfg:", cfg, flush=True)
    
    df=pd.DataFrame(records)
    print(df)
    print(df.sum())
    print("bp:", df["cost"].sum()/df["complete_money"].sum()*10000,
          "\nalphabp:", df["alpharet"].sum()/df["complete_money"].sum()*10000,
          "\nsl_corr:", df["sl_corr"].mean())
    
    aa=[item.reset_index() for item in trade_infos]
    trade_info=pd.concat(aa, ignore_index=True)
    trade_info["complete_money"]=(trade_info["cur_money"]-trade_info["b_money"]).abs()
    trade_info["f_money"]=(trade_info["f_money"]).abs()
    trade_info=trade_info.drop(columns=["index", "t_money", "cur_money", "b_money"])
    trade_info=trade_info[["complete_money"]+trade_info.columns[:-1].tolist()]
    
    tt=trade_info.groupby('symbol').agg("mean")
    print(tt[tt["f_money"]>100])
    # print("day detail:")
    # tt=trade_info.groupby(['day', 'symbol']).agg("mean")
    # print(tt[tt["f_money"]>100])
    
    return df
    
if __name__ == "__main__": 
    tools.setup_seed(0)
    dm.init()
    dr=dm.dr
    dr["uid"]=dr["sids"]
    with open('./config/at_sim.yaml', 'r') as file:
        cfg = yaml.safe_load(file)

    print("cfg:", cfg, flush=True)
    
    # starttm=20240825120000
    # endtm=20240825150000
    starttm=cfg["starttm"]
    endtm=cfg["endtm"]
    # endtm=starttm+1500
    aa=run_trade(conts.s5seccnt, cfg,
            start=starttm, end=endtm, trademoney=10000, tratio=0.2, min_delta=5,
            tradelen=5*60)
    

    
    
    
    
