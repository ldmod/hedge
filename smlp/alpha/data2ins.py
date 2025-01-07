#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 20:52:18 2024

@author: prod
"""
import cryptoqt.data.constants as conts
import numpy as np
from cryptoqt.smlp.alpha.yamlcfg import gv
import scipy.stats as stats
import cryptoqt.smlp.alpha.tools as tools

    
def getalpha(dr, min1i, alphaname):
    di=int(min1i/conts.daymincnt)
    h4i=int(min1i/conts.h4mincnt)
    h1i=int(min1i/conts.h1mincnt)
    min15i=int(min1i/conts.min15mincnt)
    idx=di
    if alphaname.split("_")[0]=="h4info":
        idx=h4i
    feas=dr[alphaname][idx-1:idx].reshape(-1,1)
    return feas


def getmin15seq(dr, min1i, seqlen):
    di=int(min1i/conts.daymincnt)
    h1i=int(min1i/conts.h1mincnt)
    min15i=int(min1i/conts.min15mincnt)
    min15pricefield=["min15info_open", "min15info_close", "min15info_high", "min15info_low", "min15info_twap", "min15info_vwap"]
    min15volumefield=["min15info_volume"]
    min15moneyfield=["min15info_money"]
    min15extrafield=["min15info_return", "min15info_dreturn", "min15info_5pvcorr", "min15info_10pvcorr", "min15info_20pvcorr"]
    fea=[]
    lcnt=seqlen
    curcloseprice=dr["dayinfo_close"][di-1]
    curvolumeavg=dr["dayinfo_volume"][di-20:di].mean(axis=0)
    curmoneyavg=dr["dayinfo_money"][di-20:di].mean(axis=0)
    
    for field in min15pricefield:
        data=dr[field][min15i-lcnt:min15i]/curcloseprice-1.0
        fea.append(data)
    for field in min15volumefield:
        data=dr[field][min15i-lcnt:min15i]/curvolumeavg-1.0
        fea.append(data)
    for field in min15moneyfield:
        data=dr[field][min15i-lcnt:min15i]/curmoneyavg-1.0
        fea.append(data)
    for field in min15extrafield:
        data=dr[field][min15i-lcnt:min15i]
        fea.append(data)
    feas=[np.expand_dims(x, axis=2).astype(np.float32) for x in fea]
    feas=np.concatenate(feas, axis=2)
    return feas


def getmin5seq(dr, min1i, seqlen):
    di=int(min1i/conts.daymincnt)
    h1i=int(min1i/conts.h1mincnt)
    min15i=int(min1i/conts.min15mincnt)
    min5i=int(min1i/conts.min5mincnt)
    min5pricefield=["min5info_open", "min5info_close", 'min5info_5high', 'min5info_5low', 'min5info_5twap', 'min5info_5vwap',
                    'min5info_15high', 'min5info_15low', 'min5info_15twap', 'min5info_15vwap',
                    'min5info_30high', 'min5info_30low', 'min5info_30twap', 'min5info_30vwap',
                    'min5info_60high', 'min5info_60low', 'min5info_60twap', 'min5info_60vwap']
    min5volumefield=["min5info_5volume", "min5info_5tbv", "min5info_15volume", "min5info_15tbv",
                     "min5info_30volume", "min5info_30tbv", "min5info_60volume", "min5info_60tbv"]
    volumescale=[1,1,3,3,6,6,12,12]
    min5moneyfield=["min5info_5money", "min5info_15money", "min5info_30money", "min5info_60money"]
    moneyscale=[1,1,3,3,6,6,12,12]
    min5extrafield=["min5info_return", "min5info_dreturn", "min5info_5tbvr", 
                    'min5info_5pvcorr', 'min5info_5pstd', 'min5info_5mapb', 'min5info_5prpp', 'min5info_5prfp', 
                    'min5info_5rstd', 'min5info_15tbvr', 'min5info_15pvcorr', 'min5info_15pstd', 
                    'min5info_15mapb', 'min5info_15prpp', 'min5info_15prfp', 'min5info_15rstd',
                    'min5info_30tbvr',  'min5info_30pvcorr', 'min5info_30pstd', 'min5info_30mapb', 
                    'min5info_30prpp', 'min5info_30prfp', 'min5info_30rstd', 'min5info_60tbvr',
                    'min5info_60pvcorr', 'min5info_60pstd', 'min5info_60mapb', 'min5info_60prpp', 'min5info_60prfp', 'min5info_60rstd',
                    'min5info_5tnum', 'min5info_15tnum', 'min5info_30tnum', 'min5info_60tnum']
    
    if gv["spotfea"]:
        min5pricefield+=["smin5info_open", "smin5info_close", 'smin5info_5high', 'smin5info_5low', 'smin5info_5twap', 'smin5info_5vwap',
                        'smin5info_15high', 'smin5info_15low', 'smin5info_15twap', 'smin5info_15vwap',
                        'smin5info_30high', 'smin5info_30low', 'smin5info_30twap', 'smin5info_30vwap',
                        'smin5info_60high', 'smin5info_60low', 'smin5info_60twap', 'smin5info_60vwap']
        smin5volumefield=["smin5info_5volume", "smin5info_5tbv", "smin5info_15volume", "smin5info_15tbv",
                          "smin5info_30volume", "smin5info_30tbv", "smin5info_60volume", "smin5info_60tbv"]
        smin5moneyfield=["smin5info_5money", "smin5info_15money", "smin5info_30money", "smin5info_60money"]
        min5extrafield+=["smin5info_return", "smin5info_dreturn", "smin5info_5tbvr", 
                        'smin5info_5pvcorr', 'smin5info_5pstd', 'smin5info_5mapb', 'smin5info_5prpp', 'smin5info_5prfp', 
                        'smin5info_5rstd', 'smin5info_15tbvr', 'smin5info_15pvcorr', 'smin5info_15pstd', 
                        'smin5info_15mapb', 'smin5info_15prpp', 'smin5info_15prfp', 'smin5info_15rstd',
                        'smin5info_30tbvr',  'smin5info_30pvcorr', 'smin5info_30pstd', 'smin5info_30mapb', 
                        'smin5info_30prpp', 'smin5info_30prfp', 'smin5info_30rstd', 'smin5info_60tbvr',
                        'smin5info_60pvcorr', 'smin5info_60pstd', 'smin5info_60mapb', 'smin5info_60prpp', 'smin5info_60prfp', 'smin5info_60rstd',
                        'smin5info_5tnum', 'smin5info_15tnum', 'smin5info_30tnum', 'smin5info_60tnum']
    
    fea=[]
    lcnt=seqlen
    curcloseprice=dr["min5info_close"][min5i-seqlen:min5i].mean(axis=0)
    curclosepricestd=dr["min5info_close"][min5i-seqlen:min5i].std(axis=0)
    curvolumeavg=dr["min5info_5volume"][min5i-seqlen:min5i].mean(axis=0)
    curvolumeavgstd=dr["min5info_5volume"][min5i-seqlen:min5i].std(axis=0)
    curmoneyavg=dr["min5info_5money"][min5i-seqlen:min5i].mean(axis=0)
    curmoneyavgstd=dr["min5info_5money"][min5i-seqlen:min5i].std(axis=0)
    scurvolumeavg=dr["smin5info_5volume"][min5i-seqlen:min5i].mean(axis=0)
    scurvolumeavgstd=dr["smin5info_5volume"][min5i-seqlen:min5i].std(axis=0)
    scurmoneyavg=dr["smin5info_5money"][min5i-seqlen:min5i].mean(axis=0)
    scurmoneyavgstd=dr["smin5info_5money"][min5i-seqlen:min5i].std(axis=0)
    
    for field in min5pricefield:
        data=(dr[field][min5i-lcnt:min5i]-curcloseprice)/curclosepricestd
        fea.append(data)
    for idx, field in enumerate(min5volumefield):
        data=(dr[field][min5i-lcnt:min5i]/volumescale[idx]-curvolumeavg)/curvolumeavgstd
        fea.append(data)
    for idx, field in enumerate(min5moneyfield):
        data=(dr[field][min5i-lcnt:min5i]/moneyscale[idx]-curmoneyavg)/curmoneyavgstd
        fea.append(data)
    if gv["spotfea"]:
        for idx, field in enumerate(smin5volumefield):
            data=(dr[field][min5i-lcnt:min5i]/volumescale[idx]-scurvolumeavg)/scurvolumeavgstd
            fea.append(data)
        for idx, field in enumerate(smin5moneyfield):
            data=(dr[field][min5i-lcnt:min5i]/moneyscale[idx]-scurmoneyavg)/scurmoneyavgstd
            fea.append(data)
    for field in min5extrafield:
        data=dr[field][min5i-lcnt:min5i]
        mean=np.nanmean(data,axis=1).reshape(-1,1)
        std=np.nanstd(data, axis=1).reshape(-1,1)
        data=(data-mean)/std
        fea.append(data)
    feas=[np.expand_dims(x, axis=2).astype(np.float32) for x in fea]
    feas=np.concatenate(feas, axis=2)
    return feas

def getmin1seq(dr, min1i, seqlen):
    di=int(min1i/conts.daymincnt)
    h1i=int(min1i/conts.h1mincnt)
    min15i=int(min1i/conts.min15mincnt)
    min1pricefield=["min1info_open", "min1info_close", "min1info_high", "min1info_low", "min1info_twap", "min1info_vwap", 
                    "min1info_15twap", "min1info_30twap", "min1info_15vwap", "min1info_30vwap"]
    min1volumefield=["min1info_volume", "min1info_tbv"]
    volumescale=[1,10]
    min1moneyfield=["min1info_money", "min1info_tbm"]
    moneyscale=[1,1]
    min1extrafield=["min1info_return", "min1info_dreturn", "min1info_tbvr", 
                    "min1info_15pvcorr", "min1info_30pvcorr", 
                    "min1info_15pstd", "min1info_30pstd", "min1info_15mapb", "min1info_30mapb", "min1info_15prpp", "min1info_30prpp",
                    "min1info_15prfp", "min1info_30prfp", "min1info_15rstd", "min1info_30rstd"]

    if gv["spotfea"]:
        min1pricefield+=["smin1info_open", "smin1info_close", "smin1info_high", "smin1info_low", "smin1info_twap", "smin1info_vwap", 
                        "smin1info_15twap", "smin1info_30twap", "smin1info_15vwap", "smin1info_30vwap"]
        smin1volumefield=["smin1info_volume", "smin1info_tbv"]
        smin1moneyfield=["smin1info_money", "smin1info_tbm"]
        min1extrafield+=["smin1info_return", "smin1info_dreturn", "smin1info_tbvr", 
                        "smin1info_15pvcorr", "smin1info_30pvcorr", 
                        "smin1info_15pstd", "smin1info_30pstd", "smin1info_15mapb", "smin1info_30mapb", "smin1info_15prpp", "smin1info_30prpp",
                        "smin1info_15prfp", "smin1info_30prfp", "smin1info_15rstd", "smin1info_30rstd"]
    
    
    fea=[]
    lcnt=seqlen
    curcloseprice=dr["min1info_close"][min1i-seqlen:min1i].mean(axis=0)
    curclosepricestd=dr["min1info_close"][min1i-seqlen:min1i].std(axis=0)
    curvolumeavg=dr["min1info_volume"][min1i-seqlen:min1i].mean(axis=0)
    curvolumeavgstd=dr["min1info_volume"][min1i-seqlen:min1i].std(axis=0)
    curmoneyavg=dr["min1info_money"][min1i-seqlen:min1i].mean(axis=0)
    curmoneyavgstd=dr["min1info_money"][min1i-seqlen:min1i].std(axis=0)
    scurvolumeavg=dr["smin1info_volume"][min1i-seqlen:min1i].mean(axis=0)
    scurvolumeavgstd=dr["smin1info_volume"][min1i-seqlen:min1i].std(axis=0)
    scurmoneyavg=dr["smin1info_money"][min1i-seqlen:min1i].mean(axis=0)
    scurmoneyavgstd=dr["smin1info_money"][min1i-seqlen:min1i].std(axis=0)
    
    for field in min1pricefield:
        data=(dr[field][min1i-lcnt:min1i]-curcloseprice)/curclosepricestd
        fea.append(data)
    for idx,field in enumerate(min1volumefield):
        data=(dr[field][min1i-lcnt:min1i]/volumescale[idx]-curvolumeavg)/curvolumeavgstd
        fea.append(data)
    for idx,field in enumerate(min1moneyfield):
        data=(dr[field][min1i-lcnt:min1i]/moneyscale[idx]-curmoneyavg)/curmoneyavgstd
        fea.append(data)
    if gv["spotfea"]:
        for idx,field in enumerate(smin1volumefield):
            data=(dr[field][min1i-lcnt:min1i]/volumescale[idx]-scurvolumeavg)/scurvolumeavgstd
            fea.append(data)
        for idx,field in enumerate(smin1moneyfield):
            data=(dr[field][min1i-lcnt:min1i]/moneyscale[idx]-scurmoneyavg)/scurmoneyavgstd
            fea.append(data)
    for field in min1extrafield:
        data=dr[field][min1i-lcnt:min1i]
        mean=np.nanmean(data,axis=1).reshape(-1,1)
        std=np.nanstd(data, axis=1).reshape(-1,1)
        data=(data-mean)/std
        fea.append(data)
    feas=[np.expand_dims(x, axis=2).astype(np.float32) for x in fea]
    feas=np.concatenate(feas, axis=2)
    return feas

def getx(dr, min1i):
    di=int(min1i/conts.daymincnt)
    h4i=int(min1i/conts.h4mincnt)
    h1i=int(min1i/conts.h1mincnt)
    min15i=int(min1i/conts.min15mincnt)
    min1extrafield=["min1info_return", "min1info_dreturn", "min1info_tbvr", 
                    "min1info_15pvcorr", "min1info_30pvcorr", 
                    "min1info_15twap", "min1info_30twap", "min1info_15vwap", "min1info_30vwap", 
                    "min1info_15pstd", "min1info_30pstd", "min1info_15mapb", "min1info_30mapb", "min1info_15prpp", "min1info_30prpp",
                    "min1info_15prfp", "min1info_30prfp", "min1info_15rstd", "min1info_30rstd"]
    fea=[]
    for field in min1extrafield:
        data=dr[field][min1i-1:min1i]
        mean=np.nanmean(data,axis=1)
        std=np.nanstd(data, axis=1)
        data=(data-mean)/std
        fea.append(data)
    if "marketFea" in gv and gv["marketFea"]:                                                                                                                                         
        min5return=np.clip((dr["min1info_15vwap"][min1i-1]/dr["min1info_15vwap"][min1i-6]-1.0)*100, -10, 10) 
        min15return=np.clip((dr["min1info_15vwap"][min1i-1]/dr["min1info_15vwap"][min1i-16]-1.0)*100, -10, 10) 
        min60return=np.clip((dr["min1info_15vwap"][min1i-1]/dr["min1info_15vwap"][min1i-61]-1.0)*100, -10, 10) 
        min120return=np.clip((dr["min1info_15vwap"][min1i-1]/dr["min1info_15vwap"][min1i-121]-1.0)*100, -10, 10) 
        min480return=np.clip((dr["min1info_15vwap"][min1i-1]/dr["min1info_15vwap"][min1i-481]-1.0)*100, -10, 10) 
        min1440return=np.clip((dr["min1info_15vwap"][min1i-1]/dr["min1info_15vwap"][min1i-1441]-1.0)*100, -10, 10) 
        main_symbols=["BTCUSDT", "ETHUSDT"]
        sidxs=[gv["data_dict"]["sidmap"][sid] for sid in main_symbols]
        market_fea=[min5return[sidxs], min15return[sidxs], min60return[sidxs], min120return[sidxs], min480return[sidxs], min1440return[sidxs]]
        market_fea=np.hstack(market_fea) #3*6
        market_fea=np.repeat(market_fea.reshape(1,-1), gv["data_dict"]["sids"].shape[0], axis=0)
        market_fea[~np.isfinite(market_fea)]=0
        market_fea=market_fea.transpose()
        fea.append(market_fea)
    

    feas=np.vstack(fea).transpose()
    return feas

def getvalid(dr, min1i):
    
    ipodays=2
    di=int(min1i/conts.daymincnt)
    h1i=int(min1i/conts.h1mincnt)
    min15i=int(min1i/conts.min15mincnt)
    valid=(np.isfinite(dr["min1info_close"][min1i-ipodays*conts.daymincnt:min1i]).sum(axis=0)==ipodays*conts.daymincnt)
    svalid=(np.isfinite(dr["smin1info_close"][min1i-ipodays*conts.daymincnt:min1i]).sum(axis=0)==ipodays*conts.daymincnt)
    valid=valid&svalid
    moneyflag=dr["min1info_money"][(di-ipodays)*conts.daymincnt:di*conts.daymincnt].mean(axis=0)>(gv["daily_trade_money"]/1440)
    valid=valid&moneyflag
    
    return  valid

def gety(dr, min1i, tmperiod):
    di=int(min1i/conts.daymincnt)
    h1i=int(min1i/conts.h1mincnt)
    min15i=int(min1i/conts.min15mincnt)
    
    startp=np.nanmean(dr["min1info_vwap"][min1i:min1i+gv["delaymin"]], axis=0)
    startp[~np.isfinite(startp)]=dr["min1info_twap"][min1i+gv["delaymin"]][~np.isfinite(startp)]
    endp=np.nanmean(dr["min1info_vwap"][min1i+tmperiod:min1i+tmperiod+gv["delaymin"]], axis=0).copy()
    endp[~np.isfinite(endp)]=dr["min1info_twap"][min1i+tmperiod][~np.isfinite(endp)]
    
    y=endp/startp-1
    return  y*100.0

def getyl(dr, min1i, tmperiod):
    di=int(min1i/conts.daymincnt)
    h1i=int(min1i/conts.h1mincnt)
    min15i=int(min1i/conts.min15mincnt)
    
    startp=np.nanmean(dr["min1info_vwap"][min1i:min1i+gv["delaymin"]], axis=0)
    startp[~np.isfinite(startp)]=dr["min1info_twap"][min1i+gv["delaymin"]][~np.isfinite(startp)]
    endp=np.nanmean(dr["min1info_vwap"][min1i+tmperiod-gv["delaymin"]:min1i+tmperiod], axis=0).copy()
    endp[~np.isfinite(endp)]=dr["min1info_twap"][min1i+tmperiod-1][~np.isfinite(endp)]
    
    y=endp/startp-1
    return  y*100.0

def getvolumerety(dr, min1i, tmperiod):
    di=int(min1i/conts.daymincnt)
    h1i=int(min1i/conts.h1mincnt)
    min15i=int(min1i/conts.min15mincnt)
    curvolume=dr["min1info_money"][min1i-conts.daymincnt*20:min1i].mean(axis=0)
    tomvolume=dr["min1info_money"][min1i:min1i+tmperiod].mean(axis=0)
    ret=tomvolume/curvolume-1.0
    return ret.astype(np.float32)

def gethiddenfea(dr, key, min1i):
    di=int(min1i/conts.daymincnt)
    h1i=int(min1i/conts.h1mincnt)
    min15i=int(min1i/conts.min15mincnt)
    data=dr[key][di]
    data[~np.isfinite(data)]=0
    return data
    










