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
sys.path.append('/home/prod/')
import cryptoqt.data.updatedata as ud
from functools import partial
import matplotlib.pyplot as plt
import cryptoqt.data.constants as conts
import cryptoqt.data.tools as tools
import cryptoqt.data.datammap as dmap
import cryptoqt.data.sec_klines.secklines as sk
import cryptoqt.alpha.secfeaextractor as secfea
import cryptoqt.bsim.gorders as gorders

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option("display.max_colwidth", 100)
pd.set_option('display.width', 10000)
delaysec=1
longterm=30


def getyvalues(secdr, s1i, tmperiod, longterm, delaysec=1):
    money=secdr["s1info_money"][s1i+delaysec:s1i+tmperiod+delaysec].sum(axis=0)
    volume=secdr["s1info_volume"][s1i+delaysec:s1i+tmperiod+delaysec].sum(axis=0)
    vwap=money/volume
    vf_y=(np.isfinite(vwap))
    
    money=secdr["s1info_smoney"][s1i+delaysec:s1i+tmperiod+delaysec].sum(axis=0)
    volume=secdr["s1info_svolume"][s1i+delaysec:s1i+tmperiod+delaysec].sum(axis=0)
    svwap=money/volume
    shigh=secdr["s1info_slow"][s1i+delaysec:s1i+tmperiod+delaysec].max(axis=0)
    slow=secdr["s1info_slow"][s1i+delaysec:s1i+tmperiod+delaysec].min(axis=0)
    vf_ys=(np.isfinite(svwap) & np.isfinite(shigh) & np.isfinite(slow))
    lsvwap=secdr["s1info_smoney"][s1i+delaysec:s1i+longterm+delaysec].sum(axis=0)/\
        secdr["s1info_svolume"][s1i+delaysec:s1i+longterm+delaysec].sum(axis=0)
    ysl=(lsvwap/svwap-1.0)*10000.0
    
    money=secdr["s1info_money"][s1i+delaysec:s1i+tmperiod+delaysec].sum(axis=0)-\
        secdr["s1info_smoney"][s1i+delaysec:s1i+tmperiod+delaysec].sum(axis=0)
    volume=secdr["s1info_volume"][s1i+delaysec:s1i+tmperiod+delaysec].sum(axis=0)-\
        secdr["s1info_svolume"][s1i+delaysec:s1i+tmperiod+delaysec].sum(axis=0)
    bvwap=money/volume
    bhigh=secdr["s1info_bhigh"][s1i+delaysec:s1i+tmperiod+delaysec].max(axis=0)
    blow=secdr["s1info_bhigh"][s1i+delaysec:s1i+tmperiod+delaysec].min(axis=0)
    vf_yb=(np.isfinite(bvwap) & np.isfinite(bhigh) & np.isfinite(blow))
    lbvwap=(secdr["s1info_money"][s1i+delaysec:s1i+longterm+delaysec].sum(axis=0)-secdr["s1info_smoney"][s1i+delaysec:s1i+longterm+delaysec].sum(axis=0))/\
        (secdr["s1info_volume"][s1i+delaysec:s1i+longterm+delaysec].sum(axis=0)-secdr["s1info_svolume"][s1i+delaysec:s1i+longterm+delaysec].sum(axis=0))
    ybl=(lbvwap/bvwap-1.0)*10000.0
    
    return vwap, vf_y, svwap, shigh, slow, vf_ys, bvwap, bhigh, blow, vf_yb, ysl, ybl


def run_trade(delta, prmfunc, start=20240515190000, end=20240515190500, 
              money=30000, tratio=0.1, delaysec=delaysec,
              path="/home/prod/cryptoqt/prm/model_states/infmodel/prmv3/res/res.h5"):
    ud.readuniverse(ud.g_data)
    lasteddownload=dmap.read_memmap(ud.g_data)
    dr=ud.g_data
    dr["uid"]=dr["sids"]
    secdata=sk.SecondData('/home/crypto/proddata/crypto/secdata/', ud.g_data["sids"])
    secdr=secdata.secdr
    
    years, months, lrets, srets = [], [], [], []
    tms, days, diffsvwaps=[], [], []
    end=sk.gtmidx_i(end)
    start=sk.gtmidx_i(start)
    dfs={}
    lastbookw=None
    mis, miics=[], []
    
    tradedelta=15*60
    tradelen=5*60
    avgmoney=money/(tradelen/5)
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
    diffsls=np.zeros(dr["uid"].shape[0])
    diffbls=np.zeros(dr["uid"].shape[0])
    tradecnt=0
    gorder=gorders.GenerateOrders(path=path)
    
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
        swmoney=np.full((dr["uid"].shape[0]), money*1.0)
        bwmoney=np.full((dr["uid"].shape[0]), money*1.0)
        tratio=np.full((dr["uid"].shape[0]), 2.0)
        diffsl=np.zeros(dr["uid"].shape[0])
        diffbl=np.zeros(dr["uid"].shape[0])
        for s1i in range(tstart, tend, delta):
            tm=sk.gtm_i(s1i)
    
            llen=secdata.read_len()
            while not ( llen>s1i+delta+delaysec):
                time.sleep(2)
                llen=secdata.read_len()
                continue
            
            dd=prmfunc(secdr, tm)
            xsvwap, xslow=dd["svwap"], dd["slow"]
            xbvwap, xbhigh=dd["bvwap"], dd["bhigh"]
            xsl, xbl =dd["sl"], dd["bl"]
            xsvwap=(xsvwap+xslow)/2
            xbvwap=(xbvwap+xbhigh)/2
            valid=np.isfinite(xsvwap)
            vwap, vf_y, svwap, shigh, slow, vf_ys, bvwap, bhigh, blow, vf_yb, ysl, ybl=\
                getyvalues(secdr, s1i, delta, longterm)
            svf=valid&vf_ys
            bvf=valid&vf_yb
            leftcnt=int((tend-s1i)/delta)
            
            tmoney=(bwmoney/leftcnt*tratio)
            tmoney=np.full((dr["uid"].shape[0]), avgmoney*5)
            mb_tradeflag=(svf & (xsvwap>=slow) & (xsvwap <= bhigh))
            mb_money[mb_tradeflag]+=tmoney[mb_tradeflag]
            bwmoney[mb_tradeflag]-=tmoney[mb_tradeflag]
            mb_volume[mb_tradeflag]+=tmoney[mb_tradeflag]/xsvwap[mb_tradeflag]
            diffmbvwap[mb_tradeflag]+=np.abs((xsvwap[mb_tradeflag]/svwap[mb_tradeflag]-1.0)*10000.0)
            diffsl[mb_tradeflag]+=np.abs(xsl[mb_tradeflag]-ysl[mb_tradeflag])
            mbcnt[mb_tradeflag]+=1
            
            tmoney=(swmoney/leftcnt*tratio)
            tmoney=np.full((dr["uid"].shape[0]), avgmoney*5)
            ms_tradeflag=(bvf & (xbvwap<=bhigh) & (xbvwap >= slow))
            ms_money[ms_tradeflag]+=tmoney[ms_tradeflag]
            swmoney[ms_tradeflag]-=tmoney[ms_tradeflag]
            ms_volume[ms_tradeflag]+=tmoney[ms_tradeflag]/xbvwap[ms_tradeflag]
            diffmsvwap[ms_tradeflag]+=np.abs((xbvwap[ms_tradeflag]/bvwap[ms_tradeflag]-1.0)*10000.0)
            diffbl[mb_tradeflag]+=np.abs(xbl[mb_tradeflag]-ybl[mb_tradeflag])
            mscnt[ms_tradeflag]+=1
            
            
            bwmoney[bwmoney<0]=0
            swmoney[swmoney<0]=0
        allvwap=secdr["s1info_money"][tstart:tend].sum(axis=0)/secdr["s1info_volume"][tstart:tend].sum(axis=0)
        svwap=secdr["s1info_smoney"][tstart:tend].sum(axis=0)/secdr["s1info_svolume"][tstart:tend].sum(axis=0)
        bvwap=(secdr["s1info_money"][tstart:tend].sum(axis=0)-secdr["s1info_smoney"][tstart:tend].sum(axis=0))/\
            (secdr["s1info_volume"][tstart:tend].sum(axis=0)-secdr["s1info_svolume"][tstart:tend].sum(axis=0))
    
        mbr=mb_money/money
        mbvwap=mb_money/mb_volume
        mbpd=(mbvwap/allvwap-1.0)*10000.0
        spd=(svwap/allvwap-1.0)*10000.0
        msr=ms_money/money
        msvwap=ms_money/ms_volume
        mspd=(msvwap/allvwap-1.0)*10000.0
        bpd=(bvwap/allvwap-1.0)*10000.0
        
        mbrs[np.isfinite(mbr)]+=mbr[np.isfinite(mbr)]
        mbpds[np.isfinite(mbpd)]+=mbpd[np.isfinite(mbpd)]
        msrs[np.isfinite(msr)]+=msr[np.isfinite(msr)]
        mspds[np.isfinite(mspd)]+=mspd[np.isfinite(mspd)]
        
        spds[np.isfinite(spd)]+=spd[np.isfinite(spd)]
        bpds[np.isfinite(bpd)]+=bpd[np.isfinite(bpd)]
        
        diffmbvwap=diffmbvwap/mbcnt
        diffmbvwaps[np.isfinite(diffmbvwap)]+=diffmbvwap[np.isfinite(diffmbvwap)]
        diffmsvwap=diffmsvwap/mscnt
        diffmsvwaps[np.isfinite(diffmsvwap)]+=diffmsvwap[np.isfinite(diffmsvwap)]
        
        diffsl/=mbcnt
        diffbl/=mscnt
        diffsls[np.isfinite(diffsl)]+=diffsl[np.isfinite(diffsl)]
        diffbls[np.isfinite(diffbl)]+=diffbl[np.isfinite(diffbl)]
        tradecnt+=1
    
    df=pd.DataFrame()
    df["sid"]=dr["sids"]
    df["mbrs"]=mbrs/tradecnt
    df["mbpds"]=mbpds/tradecnt
    df["msrs"]=msrs/tradecnt
    df["mspds"]=mspds/tradecnt
    df["diffmbvwaps"]=diffmbvwaps/tradecnt
    df["diffmsvwaps"]=diffmsvwaps/tradecnt
    df["diffsls"]=diffsls/tradecnt
    df["diffbls"]=diffbls/tradecnt
    
    # df["spds"]=spds/tradecnt
    # df["bpds"]=bpds/tradecnt

    return df
    
if __name__ == "__main__":
    ud.readuniverse(ud.g_data)
    lasteddownload=dmap.read_memmap(ud.g_data)
    dr=ud.g_data
    dr["uid"]=dr["sids"]
    
    
    aa=run_trade(conts.s5seccnt, partial(secfea.readh5res,
            path="/home/prod/cryptoqt/prm/model_states/infmodel/prmv3/res/res.h5"),
            start=20240701010000, end=20240702120000, money=100000, tratio=0.2)
    

    
    
    
    
