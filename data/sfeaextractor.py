#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 11:14:33 2024

@author: prod
"""
import numpy as np
from functools import partial
from audtorch.metrics.functional import pearsonr
import torch
import cryptoqt.data.constants as conts
from cryptoqt.data.feaextractor import g_feafunc as g_feafunc
daymincnt=conts.daymincnt
h4mincnt=conts.h4mincnt
h1mincnt=conts.h1mincnt
min15mincnt=conts.min15mincnt
min5mincnt=conts.min5mincnt


def smin1info_twap(dr, min1s, min1e, cfg={}): 
    stockcnt=dr["sids"].shape[0]
    smin1info_twap=(dr["smin1info_open"][min1s:min1e]+dr["smin1info_close"][min1s:min1e])/2.0
    return smin1info_twap 

def smin1info_vwap(dr, min1s, min1e, cfg={}): 
    stockcnt=dr["sids"].shape[0]
    smin1info_vwap=(dr["smin1info_money"][min1s:min1e]/dr["smin1info_volume"][min1s:min1e])
    return smin1info_vwap 

def smin1info_tbvr(dr, min1s, min1e): 
    data=dr["smin1info_tbv"][min1s:min1e]/dr["smin1info_volume"][min1s:min1e]
    return data

def smin1info_return(dr, min1s, min1e, cfg={}): 
    data=dr["smin1info_close"][min1s:min1e]/dr["smin1info_open"][min1s:min1e]-1.0
    return data

def smin1info_dreturn(dr, min1s, min1e, cfg={}): 
    stockcnt=dr["sids"].shape[0]
    data=np.full((min1e-min1s, stockcnt), np.nan)
    for min1i in range(min1s, min1e):
        di=int(min1i/daymincnt-1)
        data[min1i-min1s]=dr["smin1info_close"][min1i]/dr["smin1info_close"][min1i-daymincnt]-1.0
    return data

def smin1info_npvcorr(dr, min1s, min1e, cfg={}): 
    cnt=cfg["cnt"]
    stockcnt=dr["sids"].shape[0]
    data=np.full((min1e-min1s, stockcnt), np.nan)
    for min1i in range(min1s, min1e):
        if min1i-cnt+1>=0:
            volume=dr["smin1info_volume"][min1i-cnt+1:min1i+1]
            price=dr["smin1info_vwap"][min1i-cnt+1:min1i+1]
            corr = pearsonr(torch.from_numpy(price), torch.from_numpy(volume), batch_first=False)
            data[min1i-min1s]=corr[0].numpy()
    return data

def smin1info_ntwap(dr, min1s, min1e, cfg={}): 
    cnt=cfg["cnt"]
    stockcnt=dr["sids"].shape[0]
    data=np.full((min1e-min1s, stockcnt), np.nan)
    for min1i in range(min1s, min1e):
        if min1i-cnt+1>=0:
            price=dr["smin1info_vwap"][min1i-cnt+1:min1i+1]
            data[min1i-min1s]=price.mean(axis=0)
    return data

def smin1info_nvwap(dr, min1s, min1e, cfg={}): 
    cnt=cfg["cnt"]
    stockcnt=dr["sids"].shape[0]
    data=np.full((min1e-min1s, stockcnt), np.nan)
    for min1i in range(min1s, min1e):
        if min1i-cnt+1>=0:
            price=dr["smin1info_vwap"][min1i-cnt+1:min1i+1]
            vol=dr["smin1info_volume"][min1i-cnt+1:min1i+1]
            data[min1i-min1s]=(price*vol).sum(axis=0)/vol.sum(axis=0)
    return data

def smin1info_npstd(dr, min1s, min1e, cfg={}): 
    cnt=cfg["cnt"]
    stockcnt=dr["sids"].shape[0]
    data=np.full((min1e-min1s, stockcnt), np.nan)
    for min1i in range(min1s, min1e):
        if min1i-cnt+1>=0:
            price=dr["smin1info_vwap"][min1i-cnt+1:min1i+1]
            alpha=price.std(axis=0)
            alpha=alpha-np.nanmean(alpha)
            alpha=alpha/np.nanstd(alpha)
            data[min1i-min1s]=alpha
    return data

def smin1info_nmapb(dr, min1s, min1e, cfg={}): 
    cnt=cfg["cnt"]
    stockcnt=dr["sids"].shape[0]
    data=np.full((min1e-min1s, stockcnt), np.nan)
    for min1i in range(min1s, min1e):
        if min1i-cnt+1>=0:
            money=dr["smin1info_money"][min1i-cnt+1:min1i+1]
            volume=dr["smin1info_volume"][min1i-cnt+1:min1i+1]
            vwap=dr["smin1info_vwap"][min1i-cnt+1:min1i+1]
            vwapavg=vwap.mean(axis=0)
            tvwap=money.sum(axis=0)/volume.sum(axis=0)
            alpha=np.log(vwapavg/tvwap)
            alpha=alpha-np.nanmean(alpha)
            alpha=alpha/np.nanstd(alpha)
            data[min1i-min1s]=alpha
    return data


def smin1info_nprpp(dr, min1s, min1e, cfg={}): 
    cnt=cfg["cnt"]
    stockcnt=dr["sids"].shape[0]
    data=np.full((min1e-min1s, stockcnt), np.nan)
    for min1i in range(min1s, min1e):
        if min1i-cnt+1>=0:
            vwap=dr["smin1info_vwap"][min1i-cnt+1:min1i+1]
            vwapret=vwap[1:]/vwap[:-1]-1.0
            retp=vwapret*(vwapret>0)
            alpha=(retp*retp).sum(axis=0)/((vwapret*vwapret).sum(axis=0))
            alpha=alpha-np.nanmean(alpha)
            alpha=alpha/np.nanstd(alpha)
            data[min1i-min1s]=alpha
    return data

def smin1info_nprfp(dr, min1s, min1e, cfg={}): 
    cnt=cfg["cnt"]
    stockcnt=dr["sids"].shape[0]
    data=np.full((min1e-min1s, stockcnt), np.nan)
    for min1i in range(min1s, min1e):
        if min1i-cnt+1>=0:
            vwap=dr["smin1info_vwap"][min1i-cnt+1:min1i+1]
            vwapret=vwap[1:]/vwap[:-1]-1.0
            retp=vwapret*(vwapret<0)
            alpha=(retp*retp).sum(axis=0)/((vwapret*vwapret).sum(axis=0))
            alpha=alpha-np.nanmean(alpha)
            alpha=alpha/np.nanstd(alpha)
            data[min1i-min1s]=alpha
    return data

def smin1info_nrstd(dr, min1s, min1e, cfg={}): 
    cnt=cfg["cnt"]
    stockcnt=dr["sids"].shape[0]
    data=np.full((min1e-min1s, stockcnt), np.nan)
    for min1i in range(min1s, min1e):
        if min1i-cnt+1>=0:
            price=dr["smin1info_return"][min1i-cnt+1:min1i+1]
            alpha=price.std(axis=0)
            alpha=alpha-np.nanmean(alpha)
            alpha=alpha/np.nanstd(alpha)
            data[min1i-min1s]=alpha
    return data

def smin5info_open(dr, min5s, min5e, cfg={}): 
    stockcnt=dr["sids"].shape[0]
    min1s, min1e=min5mincnt*min5s, min5mincnt*min5e
    data=dr["smin1info_open"][min1s:min1e][::min5mincnt]
    return data

def smin5info_close(dr, min5s, min5e, cfg={}): 
    stockcnt=dr["sids"].shape[0]
    min1s, min1e=min5mincnt*min5s, min5mincnt*min5e
    data=dr["smin1info_close"][min1s:min1e][min5mincnt-1::min5mincnt]
    return data

def smin5info_return(dr, min5s, min5e, cfg={}): 
    data=dr["smin5info_close"][min5s:min5e]/dr["smin5info_open"][min5s:min5e]-1.0
    return data

def smin5info_dreturn(dr, min5s, min5e, cfg={}): 
    stockcnt=dr["sids"].shape[0]
    data=np.full((min5e-min5s, stockcnt), np.nan)
    for min5i in range(min5s, min5e):
        di=int(min5i*min5mincnt/daymincnt-1)
        data[min5i-min5s]=dr["smin5info_close"][min5i]/dr["min1info_close"][min5i*min5mincnt-daymincnt]-1.0
    return data

def smin5info_nhigh(dr, min5s, min5e, cfg={}): 
    cnt=cfg["cnt"]
    stockcnt=dr["sids"].shape[0]
    data=np.full((min5e-min5s, stockcnt), np.nan)
    for min5i in range(min5s, min5e):
        min1i = min5mincnt*(min5i+1)
        if min1i-cnt>=0:
            data[min5i-min5s]=dr["smin1info_high"][min1i-cnt:min1i].max(axis=0)
    return data

def smin5info_nlow(dr, min5s, min5e, cfg={}): 
    cnt=cfg["cnt"]
    stockcnt=dr["sids"].shape[0]
    data=np.full((min5e-min5s, stockcnt), np.nan)
    for min5i in range(min5s, min5e):
        min1i = min5mincnt*(min5i+1)
        if min1i-cnt>=0:
            data[min5i-min5s]=dr["smin1info_low"][min1i-cnt:min1i].min(axis=0)
    return data

def smin5info_nvolume(dr, min5s, min5e, cfg={}): 
    cnt=cfg["cnt"]
    stockcnt=dr["sids"].shape[0]
    data=np.full((min5e-min5s, stockcnt), np.nan)
    for min5i in range(min5s, min5e):
        min1i = min5mincnt*(min5i+1)
        if min1i-cnt>=0:
            data[min5i-min5s]=dr["smin1info_volume"][min1i-cnt:min1i].sum(axis=0)
    return data

def smin5info_ntnum(dr, min5s, min5e, cfg={}): 
    cnt=cfg["cnt"]
    stockcnt=dr["sids"].shape[0]
    data=np.full((min5e-min5s, stockcnt), np.nan)
    for min5i in range(min5s, min5e):
        min1i = min5mincnt*(min5i+1)
        if min1i-cnt>=0:
            data[min5i-min5s]=dr["smin1info_tnum"][min1i-cnt:min1i].sum(axis=0)
    return data

def smin5info_ntbv(dr, min5s, min5e, cfg={}): 
    cnt=cfg["cnt"]
    stockcnt=dr["sids"].shape[0]
    data=np.full((min5e-min5s, stockcnt), np.nan)
    for min5i in range(min5s, min5e):
        min1i = min5mincnt*(min5i+1)
        if min1i-cnt>=0:
            data[min5i-min5s]=dr["smin1info_tbv"][min1i-cnt:min1i].sum(axis=0)
    return data

def smin5info_ntbvr(dr, min5s, min5e, cfg={}): 
    cnt=cfg["cnt"]
    stockcnt=dr["sids"].shape[0]
    data=np.full((min5e-min5s, stockcnt), np.nan)
    for min5i in range(min5s, min5e):
        min1i = min5mincnt*(min5i+1)
        if min1i-cnt>=0:
            data[min5i-min5s]=dr["smin1info_tbv"][min1i-cnt:min1i].sum(axis=0)/dr["smin1info_volume"][min1i-cnt:min1i].sum(axis=0)
    return data

def smin5info_nmoney(dr, min5s, min5e, cfg={}): 
    cnt=cfg["cnt"]
    stockcnt=dr["sids"].shape[0]
    data=np.full((min5e-min5s, stockcnt), np.nan)
    for min5i in range(min5s, min5e):
        min1i = min5mincnt*(min5i+1)
        if min1i-cnt>=0:
            data[min5i-min5s]=dr["smin1info_money"][min1i-cnt:min1i].sum(axis=0)
    return data 

def smin5info_ntwap(dr, min5s, min5e, cfg={}): 
    cnt=cfg["cnt"]
    stockcnt=dr["sids"].shape[0]
    data=np.full((min5e-min5s, stockcnt), np.nan)
    for min5i in range(min5s, min5e):
        min1i = min5mincnt*(min5i+1)
        if min1i-cnt>=0:
            price=dr["smin1info_vwap"][min1i-cnt:min1i]
            data[min5i-min5s]=price.mean(axis=0)
    return data

def smin5info_nvwap(dr, min5s, min5e, cfg={}): 
    cnt=cfg["cnt"]
    stockcnt=dr["sids"].shape[0]
    data=np.full((min5e-min5s, stockcnt), np.nan)
    for min5i in range(min5s, min5e):
        min1i = min5mincnt*(min5i+1)
        if min1i-cnt>=0:
            price=dr["smin1info_vwap"][min1i-cnt:min1i]
            vol=dr["smin1info_volume"][min1i-cnt:min1i]
            data[min5i-min5s]=(price*vol).sum(axis=0)/vol.sum(axis=0)
    return data

def smin5info_npvcorr(dr, min5s, min5e, cfg={}): 
    cnt=cfg["cnt"]
    stockcnt=dr["sids"].shape[0]
    data=np.full((min5e-min5s, stockcnt), np.nan)
    for min5i in range(min5s, min5e):
        min1i = min5mincnt*(min5i+1)
        if min1i-cnt>=0:
            volume=dr["smin1info_volume"][min1i-cnt:min1i]
            price=dr["smin1info_vwap"][min1i-cnt:min1i]
            corr = pearsonr(torch.from_numpy(price), torch.from_numpy(volume), batch_first=False)
            data[min5i-min5s]=corr[0].numpy()
    return data

def smin5info_npstd(dr, min5s, min5e, cfg={}): 
    cnt=cfg["cnt"]
    stockcnt=dr["sids"].shape[0]
    data=np.full((min5e-min5s, stockcnt), np.nan)
    for min5i in range(min5s, min5e):
        min1i = min5mincnt*(min5i+1)
        if min1i-cnt>=0:
            price=dr["smin1info_vwap"][min1i-cnt:min1i]
            alpha=price.std(axis=0)
            alpha=alpha-np.nanmean(alpha)
            alpha=alpha/np.nanstd(alpha)
            data[min5i-min5s]=alpha
    return data

def smin5info_nmapb(dr, min5s, min5e, cfg={}): 
    cnt=cfg["cnt"]
    stockcnt=dr["sids"].shape[0]
    data=np.full((min5e-min5s, stockcnt), np.nan)
    for min5i in range(min5s, min5e):
        min1i = min5mincnt*(min5i+1)
        if min1i-cnt>=0:
            money=dr["smin1info_money"][min1i-cnt:min1i]
            volume=dr["smin1info_volume"][min1i-cnt:min1i]
            vwap=dr["smin1info_vwap"][min1i-cnt:min1i]
            vwapavg=vwap.mean(axis=0)
            tvwap=money.sum(axis=0)/volume.sum(axis=0)
            alpha=np.log(vwapavg/tvwap)
            alpha=alpha-np.nanmean(alpha)
            alpha=alpha/np.nanstd(alpha)
            data[min5i-min5s]=alpha
    return data


def smin5info_nprpp(dr, min5s, min5e, cfg={}): 
    cnt=cfg["cnt"]
    stockcnt=dr["sids"].shape[0]
    data=np.full((min5e-min5s, stockcnt), np.nan)
    for min5i in range(min5s, min5e):
        min1i = min5mincnt*(min5i+1)
        if min1i-cnt>=0:
            vwap=dr["smin1info_vwap"][min1i-cnt:min1i]
            vwapret=vwap[1:]/vwap[:-1]-1.0
            retp=vwapret*(vwapret>0)
            alpha=(retp*retp).sum(axis=0)/((vwapret*vwapret).sum(axis=0))
            alpha=alpha-np.nanmean(alpha)
            alpha=alpha/np.nanstd(alpha)
            data[min5i-min5s]=alpha
    return data

def smin5info_nprfp(dr, min5s, min5e, cfg={}): 
    cnt=cfg["cnt"]
    stockcnt=dr["sids"].shape[0]
    data=np.full((min5e-min5s, stockcnt), np.nan)
    for min5i in range(min5s, min5e):
        min1i = min5mincnt*(min5i+1)
        if min1i-cnt>=0:
            vwap=dr["smin1info_vwap"][min1i-cnt:min1i]
            vwapret=vwap[1:]/vwap[:-1]-1.0
            retp=vwapret*(vwapret<0)
            alpha=(retp*retp).sum(axis=0)/((vwapret*vwapret).sum(axis=0))
            alpha=alpha-np.nanmean(alpha)
            alpha=alpha/np.nanstd(alpha)
            data[min5i-min5s]=alpha
    return data

def smin5info_nrstd(dr, min5s, min5e, cfg={}): 
    cnt=cfg["cnt"]
    stockcnt=dr["sids"].shape[0]
    data=np.full((min5e-min5s, stockcnt), np.nan)
    for min5i in range(min5s, min5e):
        min1i = min5mincnt*(min5i+1)
        if min1i-cnt>=0:
            price=dr["smin1info_return"][min1i-cnt:min1i]
            alpha=price.std(axis=0)
            alpha=alpha-np.nanmean(alpha)
            alpha=alpha/np.nanstd(alpha)
            data[min5i-min5s]=alpha
    return data


g_feafunc["smin1info_twap"]=smin1info_twap
g_feafunc["smin1info_vwap"]=smin1info_vwap
g_feafunc["smin1info_tbvr"]=smin1info_tbvr
g_feafunc["smin1info_return"]=smin1info_return
g_feafunc["smin1info_dreturn"]=smin1info_dreturn
for cnt in [5, 15,30]:
    key="smin1info_"+str(cnt)+"pvcorr"
    g_feafunc[key]=partial(smin1info_npvcorr, cfg={"cnt":cnt})
    key="smin1info_"+str(cnt)+"twap"
    g_feafunc[key]=partial(smin1info_ntwap, cfg={"cnt":cnt})
    key="smin1info_"+str(cnt)+"vwap"
    g_feafunc[key]=partial(smin1info_nvwap, cfg={"cnt":cnt})
    key="smin1info_"+str(cnt)+"pstd"
    g_feafunc[key]=partial(smin1info_npstd, cfg={"cnt":cnt})
    key="smin1info_"+str(cnt)+"mapb"
    g_feafunc[key]=partial(smin1info_nmapb, cfg={"cnt":cnt})
    key="smin1info_"+str(cnt)+"prpp"
    g_feafunc[key]=partial(smin1info_nprpp, cfg={"cnt":cnt})
    key="smin1info_"+str(cnt)+"prfp"
    g_feafunc[key]=partial(smin1info_nprfp, cfg={"cnt":cnt})
    key="smin1info_"+str(cnt)+"rstd"
    g_feafunc[key]=partial(smin1info_nrstd, cfg={"cnt":cnt})
    
g_feafunc["smin5info_open"]=smin5info_open
g_feafunc["smin5info_close"]=smin5info_close
g_feafunc["smin5info_return"]=smin5info_return
g_feafunc["smin5info_dreturn"]=smin5info_dreturn
for cnt in [5, 15, 30, 60]:
    key="smin5info_"+str(cnt)+"high"
    g_feafunc[key]=partial(smin5info_nhigh, cfg={"cnt":cnt})
    key="smin5info_"+str(cnt)+"low"
    g_feafunc[key]=partial(smin5info_nlow, cfg={"cnt":cnt})
    key="smin5info_"+str(cnt)+"volume"
    g_feafunc[key]=partial(smin5info_nvolume, cfg={"cnt":cnt})
    key="smin5info_"+str(cnt)+"money"
    g_feafunc[key]=partial(smin5info_nmoney, cfg={"cnt":cnt})
    key="smin5info_"+str(cnt)+"tbv"
    g_feafunc[key]=partial(smin5info_ntbv, cfg={"cnt":cnt})
    key="smin5info_"+str(cnt)+"tbvr"
    g_feafunc[key]=partial(smin5info_ntbvr, cfg={"cnt":cnt})
    key="smin5info_"+str(cnt)+"tnum"
    g_feafunc[key]=partial(smin5info_ntnum, cfg={"cnt":cnt})
    key="smin5info_"+str(cnt)+"twap"
    g_feafunc[key]=partial(smin5info_ntwap, cfg={"cnt":cnt})
    key="smin5info_"+str(cnt)+"vwap"
    g_feafunc[key]=partial(smin5info_nvwap, cfg={"cnt":cnt})
    key="smin5info_"+str(cnt)+"pvcorr"
    g_feafunc[key]=partial(smin5info_npvcorr, cfg={"cnt":cnt})
    key="smin5info_"+str(cnt)+"pstd"
    g_feafunc[key]=partial(smin5info_npstd, cfg={"cnt":cnt})
    key="smin5info_"+str(cnt)+"mapb"
    g_feafunc[key]=partial(smin5info_nmapb, cfg={"cnt":cnt})
    key="smin5info_"+str(cnt)+"prpp"
    g_feafunc[key]=partial(smin5info_nprpp, cfg={"cnt":cnt})
    key="smin5info_"+str(cnt)+"prfp"
    g_feafunc[key]=partial(smin5info_nprfp, cfg={"cnt":cnt})
    key="smin5info_"+str(cnt)+"rstd"
    g_feafunc[key]=partial(smin5info_nrstd, cfg={"cnt":cnt})








