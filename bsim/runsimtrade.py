#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 12:40:22 2024

@author: ld
"""
import sys
import os
sys.path.append(os.path.abspath(__file__+"../../../../"))
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
np.seterr(invalid='ignore')
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
# import cryptoqt.bsim.gorders_new as gorders
import cryptoqt.bsim.gorders as gorders
import cryptoqt.bsim.view_order_info as voi
import yaml
import random
import cryptoqt.data.data_manager as dm
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option("display.max_colwidth", 100)
pd.set_option('display.width', 10000)
delaysec=2
longterm=30


def run_trade(delta, cfg, start=20240515190000, end=20240515190500, 
              trademoney=30000, tratio=0.1, delaysec=delaysec,
              path="/home/crypto/smlp_prm/cryptoqt/prm/model_states/infmodel/prmv3/res/res.h5",
              # book_path="/home/crypto/signal/predv215_3w/", scale=0.5,
                book_path="/home/crypto/signal/predv215_3w/", scale=1.0,
                # book_path="/home/crypto/signal/predv215_5w/", scale=0.3,
              ):
    dr=dm.dr
    secdata=sk.SecondData('/home/crypto/proddata/crypto/secdata/', dr["sids"])
    secdr=secdata.secdr
    start_date, end_date=start, end
    years, months, lrets, srets = [], [], [], []
    tms, days, diffsvwaps=[], [], []
    end=sk.gtmidx_i(end)
    start=sk.gtmidx_i(start)
    dfs={}
    lastbookw=None
    mis, miics=[], []
    
    tradedelta=15*60
    tradelen=5*60
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
    # gorder.update_tratio(sk.gtm_i(start))
    records=[]

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
        fname=book_path+"/"+str(sk.gtm_i(tstart-900))+"_book.csv"
        last_book=pd.read_csv(fname)

        trade_info=pd.DataFrame()
        trade_info["symbol"]=cur_book["sid"]
        trade_info["t_money"]=cur_book["bookw"]*scale
        trade_info["b_money"]=last_book["bookw"]*scale
        trade_info["cur_money"]=last_book["bookw"]*scale
        trade_info["f_money"]=trade_info["t_money"]-trade_info["b_money"]
        trade_info["order_volume"]=last_book["bookw"]*0
        
        for idx in range(dr["uid"].shape[0]):
            item=trade_info.iloc[idx]
            smpairs[dr["uid"][idx]]=(item["t_money"], item["b_money"])
                
        tstart+=25
        gorder.update_tratio(sk.gtm_i(tstart-60*10))
        gorder.restart(sk.gtm_i(tstart), sk.gtm_i(tend), delta, smpairs)
        for s1i in range(tstart, tend, delta):
            tm=sk.gtm_i(s1i)
    
            leftcnt=int((tend-s1i)/delta)
            
            smpairs={}
            for sidx,sid in enumerate(dr["sids"]):
                if abs(trade_info.iloc[sidx]["f_money"]) > 1:
                    smpairs[sid]=trade_info.iloc[sidx]["cur_money"]

            orders=gorder.update_and_gorders(tm, smpairs)

            ######### simulator trade ###########
            # vwap, vf_y, svwap, shigh, slow, vf_ys, bvwap, bhigh, blow, vf_yb, smoney, bmoney, ysl, ybl=gorders.getyvalues(
            #     secdr, s1i, delta*gorder.cancel_delay, longterm)
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
                    if svf[sidx] & (order_price >= slow[sidx]) & (order_price<=bhigh[sidx]) & (random.random() < trade_succ_ratio):
                        trade_money=min(abs(order_money), smoney[sidx]*money_limit_ratio)
                        trade_info.loc[sidx, "order_volume"]+=abs(trade_money/order_price)
                        diffmbvwap[sidx]+=np.abs((order_price/svwap[sidx]-1.0)*10000.0)
                        mbcnt[sidx]+=1
                        trade_info.loc[sidx, "cur_money"]+=trade_money
                else:
                    if bvf[sidx] & (order_price <= bhigh[sidx]) & (order_price>=slow[sidx]) & (random.random() < trade_succ_ratio):
                        trade_money=min(abs(order_money), bmoney[sidx]*money_limit_ratio)
                        trade_info.loc[sidx, "order_volume"]+=abs(trade_money/order_price)
                        diffmsvwap[sidx]+=np.abs((order_price/bvwap[sidx]-1.0)*10000.0)
                        mscnt[sidx]+=1
                        trade_info.loc[sidx, "cur_money"]-=trade_money
            ######### simulator trade ###########
            
        print("complete:", sk.gtm_i(tstart))
        # gorder.save_order_info(sk.gtm_i(tstart)+2)
        order_info=gorder.get_order_info(sk.gtm_i(tstart))
        order_info=voi.process_df(order_info)
        allvwap=secdr["s1info_money"][tstart:tend].sum(axis=0)/secdr["s1info_volume"][tstart:tend].sum(axis=0)
        svwap=secdr["s1info_smoney"][tstart:tend].sum(axis=0)/secdr["s1info_svolume"][tstart:tend].sum(axis=0)
        bvwap=(secdr["s1info_money"][tstart:tend].sum(axis=0)-secdr["s1info_smoney"][tstart:tend].sum(axis=0))/\
            (secdr["s1info_volume"][tstart:tend].sum(axis=0)-secdr["s1info_svolume"][tstart:tend].sum(axis=0))
        
        trade_info["vprice"]=(trade_info["cur_money"]-trade_info["b_money"])/trade_info["order_volume"]
        trade_info["vwap"]=allvwap
        trade_info["cost"]=trade_info["order_volume"]*allvwap*np.sign(trade_info["f_money"])-(trade_info["cur_money"]-trade_info["b_money"])
        trade_info=trade_info.set_index("symbol")
        dftmp=order_info
        # dftmp['dsl']=(dftmp['pred_sl']-dftmp['ysl']).abs().round(2)
        # dftmp['dbl']=(dftmp['pred_bl']-dftmp['ybl']).abs().round(2)
        # dftmp['dsl']=(dftmp['pred_sl']-dftmp['ysl']).round(2)
        # dftmp['dbl']=(dftmp['pred_bl']-dftmp['ybl']).round(2)
        # dftmp=dftmp.drop(columns=["order_volume", "cur_money", "a_shift", 'cr_avg', 'ecr'])
        dfagg=dftmp.groupby('symbol')
        def cosine(col1, col2, x):
            # x[col1]=x[col1]-x[col1].rolling(window=2).mean()
            # x[col1]=x[col1].rolling(window=3).mean()
            # x[col2]=x[col2].iloc[::-1].rolling(window = 12).mean().iloc[::-1]
            flag=(~(x[col1].isna() | x[col2].isna()))
            x=x[flag]
            # x[col1]=x[col1]-x[col1].mean()
            inner_sum=(x[col1]*x[col2]).sum()
            inner_m=np.sqrt((x[col1]*x[col1]).sum())*np.sqrt((x[col2]*x[col2]).sum())
            return inner_sum/inner_m
        # sl_corr=dfagg[['pred_sl','ysl']].apply(partial(cosine, 'pred_sl','ysl'))
        # bl_corr=dfagg[['pred_bl','ybl']].apply(partial(cosine, 'pred_bl','ybl'))
        # dl=dfagg[['dsl','dbl', 'pmr', 'pred_sl', 'ysl', 'pred_bl','ybl']].agg("mean").round(2)
        # sl_corr=dfagg[['pred_sl','ysl']].corr().unstack().iloc[:,1]
        # bl_corr=dfagg[['pred_bl','ybl']].corr().unstack().iloc[:,1]
        # trade_info=pd.merge(trade_info, sl_corr.rename('sl_corr'), how='left',left_index=True, right_index=True)
        # trade_info=pd.merge(trade_info, bl_corr.rename('bl_corr'), how='left',left_index=True, right_index=True)
        # trade_info=pd.merge(trade_info, dl, how='left',left_index=True, right_index=True)
        print( trade_info[trade_info["f_money"].abs()>10])
        
        complete_money=(trade_info["cur_money"]-trade_info["b_money"]).abs().sum()
        target_money=trade_info["f_money"].abs().sum()
        print("trade_info:", complete_money/target_money, target_money.round(2), complete_money.round(2),
              trade_info["cost"].sum().round(2),
              # "corr:", trade_info["sl_corr"].mean().round(2), trade_info["bl_corr"].mean().round(2),
              # "dl:", trade_info["dsl"].mean().round(2), trade_info["dbl"].mean().round(2),
              # "dl:", trade_info["pred_sl"].mean().round(2), trade_info["ysl"].mean().round(2),
              )
        records.append(dict(curtm=sk.gtm_i(tstart), target_money=target_money, complete_money=complete_money, cost=trade_info["cost"].sum(),
                            # sl_corr=trade_info["sl_corr"].mean(), 
                            # bl_corr=trade_info["bl_corr"].mean(),
                            # dsl=trade_info["dsl"].mean(),
                            # dbl=trade_info["dbl"].mean(),
                            ))

        a=0
        
    print("cfg:", cfg, flush=True)
    
    df=pd.DataFrame(records)
    print(df)
    print(df.mean())
    return df
    
if __name__ == "__main__": 
    tools.setup_seed(0)
    dm.init()
    dr=dm.dr
    with open('./config/simtrade.yaml', 'r') as file:
        cfg = yaml.safe_load(file)

    print("cfg:", cfg, flush=True)
    
    # starttm=20240825120000
    # endtm=20240825150000
    starttm=cfg["starttm"]
    endtm=cfg["endtm"]
    # endtm=starttm+1500
    aa=run_trade(conts.s5seccnt, cfg,
            start=starttm, end=endtm, trademoney=10000, tratio=0.2)
    

    
    
    
    
