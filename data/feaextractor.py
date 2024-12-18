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
daymincnt=conts.daymincnt
h4mincnt=conts.h4mincnt
h1mincnt=conts.h1mincnt
min15mincnt=conts.min15mincnt
min5mincnt=conts.min5mincnt

def dayinfo_pre_close(dr, ds, de): 
    stockcnt=dr["sids"].shape[0]
    data=np.full((de-ds, stockcnt), np.nan)
    for di in range(ds, de):
        if di>=1:
            data[di-ds]=dr["dayinfo_close"][di-1]
    return data

def dayinfo_tbvr(dr, ds, de): 
    data=dr["dayinfo_tbv"][ds:de]/dr["dayinfo_volume"][ds:de]
    return data

def dayinfo_tsv(dr, ds, de): 
    data=dr["dayinfo_volume"][ds:de]-dr["dayinfo_tbv"][ds:de]
    return data

def dayinfo_tsm(dr, ds, de): 
    data=dr["dayinfo_money"][ds:de]-dr["dayinfo_tbm"][ds:de]
    return data

def dayinfo_return(dr, ds, de): 
    data=dr["dayinfo_close"][ds:de]/dr["dayinfo_pre_close"][ds:de]-1.0
    return data

def dayinfo_twap(dr, ds, de): 
    stockcnt=dr["sids"].shape[0]
    min1s, min1e=daymincnt*ds, daymincnt*de
    data=dr["min1info_close"][min1s:min1e].reshape(-1,daymincnt,stockcnt).mean(axis=1)
    return data

def dayinfo_vwap(dr, ds, de): 
    stockcnt=dr["sids"].shape[0]
    min1s, min1e=daymincnt*ds, daymincnt*de
    data=(dr["min1info_close"][min1s:min1e]*dr["min1info_volume"][min1s:min1e]).reshape(-1,daymincnt,stockcnt).sum(axis=1)
    data=data/(dr["min1info_volume"][min1s:min1e].reshape(-1,daymincnt,stockcnt).sum(axis=1))
    return data

def dayinfo_ntwap(dr, ds, de, cfg={}): 
    cnt=cfg["cnt"]
    stockcnt=dr["sids"].shape[0]
    data=np.full((de-ds, stockcnt), np.nan)
    for di in range(ds, de):
        if di-cnt+1>=0:
            data[di-ds]=(dr["dayinfo_twap"][di-cnt+1:di+1].mean(axis=0))
    return data

def dayinfo_nvwap(dr, ds, de, cfg={}): 
    stockcnt=dr["sids"].shape[0]
    cnt=cfg["cnt"]
    data=np.full((de-ds, stockcnt), np.nan)
    for di in range(ds, de):
        if di-cnt+1>=0:
            data[di-ds]=(dr["dayinfo_vwap"][di-cnt+1:di+1].mean(axis=0))
    return data

def dayinfo_nhigh(dr, ds, de, cfg={}): 
    cnt=cfg["cnt"]
    stockcnt=dr["sids"].shape[0]
    data=np.full((de-ds, stockcnt), np.nan)
    for di in range(ds, de):
        if di-cnt+1>=0:
            data[di-ds]=(dr["dayinfo_high"][di-cnt+1:di+1].max(axis=0))
    return data

def dayinfo_nlow(dr, ds, de, cfg={}): 
    cnt=cfg["cnt"]
    stockcnt=dr["sids"].shape[0]
    data=np.full((de-ds, stockcnt), np.nan)
    for di in range(ds, de):
        if di-cnt+1>=0:
            data[di-ds]=(dr["dayinfo_low"][di-cnt+1:di+1].min(axis=0))
    return data

def dayinfo_nvolume(dr, ds, de, cfg={}): 
    cnt=cfg["cnt"]
    stockcnt=dr["sids"].shape[0]
    data=np.full((de-ds, stockcnt), np.nan)
    for di in range(ds, de):
        if di-cnt+1>=0:
            data[di-ds]=(dr["dayinfo_volume"][di-cnt+1:di+1].mean(axis=0))
    return data

def dayinfo_ntbv(dr, ds, de, cfg={}): 
    cnt=cfg["cnt"]
    stockcnt=dr["sids"].shape[0]
    data=np.full((de-ds, stockcnt), np.nan)
    for di in range(ds, de):
        if di-cnt+1>=0:
            data[di-ds]=(dr["dayinfo_tbv"][di-cnt+1:di+1].mean(axis=0))
    return data

def dayinfo_ntbm(dr, ds, de, cfg={}): 
    cnt=cfg["cnt"]
    stockcnt=dr["sids"].shape[0]
    data=np.full((de-ds, stockcnt), np.nan)
    for di in range(ds, de):
        if di-cnt+1>=0:
            data[di-ds]=(dr["dayinfo_tbm"][di-cnt+1:di+1].mean(axis=0))
    return data

def dayinfo_ntbvr(dr, ds, de, cfg={}): 
    cnt=cfg["cnt"]
    stockcnt=dr["sids"].shape[0]
    data=np.full((de-ds, stockcnt), np.nan)
    for di in range(ds, de):
        if di-cnt+1>=0:
            data[di-ds]=(dr["dayinfo_tbv"][di-cnt+1:di+1]/dr["dayinfo_volume"][di-cnt+1:di+1]).mean(axis=0)
    return data

def dayinfo_ntsv(dr, ds, de, cfg={}): 
    cnt=cfg["cnt"]
    stockcnt=dr["sids"].shape[0]
    data=np.full((de-ds, stockcnt), np.nan)
    for di in range(ds, de):
        if di-cnt+1>=0:
            data[di-ds]=(dr["dayinfo_volume"][di-cnt+1:di+1]-dr["dayinfo_tbv"][di-cnt+1:di+1]).mean(axis=0)
    return data

def dayinfo_ntsm(dr, ds, de, cfg={}): 
    cnt=cfg["cnt"]
    stockcnt=dr["sids"].shape[0]
    data=np.full((de-ds, stockcnt), np.nan)
    for di in range(ds, de):
        if di-cnt+1>=0:
            data[di-ds]=(dr["dayinfo_money"][di-cnt+1:di+1]-dr["dayinfo_tbm"][di-cnt+1:di+1]).mean(axis=0)
    return data

def dayinfo_ntnum(dr, ds, de, cfg={}): 
    cnt=cfg["cnt"]
    stockcnt=dr["sids"].shape[0]
    data=np.full((de-ds, stockcnt), np.nan)
    for di in range(ds, de):
        if di-cnt+1>=0:
            data[di-ds]=(dr["dayinfo_tnum"][di-cnt+1:di+1].mean(axis=0))
    return data

def dayinfo_nreturn(dr, ds, de, cfg={}): 
    cnt=cfg["cnt"]
    stockcnt=dr["sids"].shape[0]
    data=np.full((de-ds, stockcnt), np.nan)
    for di in range(ds, de):
        if di-cnt>=0:
            data[di-ds]=(dr["dayinfo_"+str(cnt)+"twap"][di]/dr["dayinfo_"+str(cnt)+"twap"][di-cnt]-1.0)
    return data

def dayinfo_nmoney(dr, ds, de, cfg={}): 
    cnt=cfg["cnt"]
    stockcnt=dr["sids"].shape[0]
    data=np.full((de-ds, stockcnt), np.nan)
    for di in range(ds, de):
        if di-cnt+1>=0:
            data[di-ds]=(dr["dayinfo_money"][di-cnt+1:di+1].mean(axis=0))
    return data 

def dayinfo_npvcorr(dr, ds, de, cfg={}): 
    cnt=cfg["cnt"]
    stockcnt=dr["sids"].shape[0]
    data=np.full((de-ds, stockcnt), np.nan)
    for di in range(ds, de):
        if di-cnt+1>=0:
            volume=dr["dayinfo_volume"][di-cnt+1:di+1]
            price=dr["dayinfo_close"][di-cnt+1:di+1]
            corr = pearsonr(torch.from_numpy(price), torch.from_numpy(volume), batch_first=False)
            data[di-ds]=corr[0].numpy()
    return data 

def h4info_open(dr, h4s, h4e, cfg={}): 
    stockcnt=dr["sids"].shape[0]
    min1s, min1e=h4mincnt*h4s, h4mincnt*h4e
    data=dr["min1info_open"][min1s:min1e][::h4mincnt]
    return data

def h4info_close(dr, h4s, h4e, cfg={}): 
    stockcnt=dr["sids"].shape[0]
    min1s, min1e=h4mincnt*h4s, h4mincnt*h4e
    data=dr["min1info_close"][min1s:min1e][h4mincnt-1::h4mincnt]
    return data

def h4info_high(dr, h4s, h4e, cfg={}): 
    stockcnt=dr["sids"].shape[0]
    min1s, min1e=h4mincnt*h4s, h4mincnt*h4e
    data=dr["min1info_high"][min1s:min1e].reshape(-1, h4mincnt, stockcnt).max(axis=1)
    return data

def h4info_low(dr, h4s, h4e, cfg={}): 
    stockcnt=dr["sids"].shape[0]
    min1s, min1e=h4mincnt*h4s, h4mincnt*h4e
    data=dr["min1info_low"][min1s:min1e].reshape(-1, h4mincnt, stockcnt).min(axis=1)
    return data

def h4info_twap(dr, h4s, h4e, cfg={}): 
    stockcnt=dr["sids"].shape[0]
    min1s, min1e=h4mincnt*h4s, h4mincnt*h4e
    data=dr["min1info_close"][min1s:min1e].reshape(-1, h4mincnt, stockcnt).mean(axis=1)
    return data

def h4info_vwap(dr, h4s, h4e, cfg={}): 
    stockcnt=dr["sids"].shape[0]
    min1s, min1e=h4mincnt*h4s, h4mincnt*h4e
    data=(dr["min1info_close"][min1s:min1e]*dr["min1info_volume"][min1s:min1e]).reshape(-1,h4mincnt,stockcnt).sum(axis=1)
    data=data/(dr["min1info_volume"][min1s:min1e].reshape(-1,h4mincnt,stockcnt).sum(axis=1))
    return data

def h4info_volume(dr, h4s, h4e, cfg={}): 
    stockcnt=dr["sids"].shape[0]
    min1s, min1e=h4mincnt*h4s, h4mincnt*h4e
    data=dr["min1info_volume"][min1s:min1e].reshape(-1, h4mincnt, stockcnt).sum(axis=1)
    return data

def h4info_money(dr, h4s, h4e, cfg={}): 
    stockcnt=dr["sids"].shape[0]
    min1s, min1e=h4mincnt*h4s, h4mincnt*h4e
    min1info_money=(dr["min1info_open"][min1s:min1e]+dr["min1info_close"][min1s:min1e])/2.0*dr["min1info_volume"][min1s:min1e]
    data=min1info_money.reshape(-1, h4mincnt, stockcnt).sum(axis=1)
    return data 

def h4info_tnum(dr, h4s, h4e, cfg={}): 
    stockcnt=dr["sids"].shape[0]
    min1s, min1e=h4mincnt*h4s, h4mincnt*h4e
    data=dr["min1info_tnum"][min1s:min1e].reshape(-1, h4mincnt, stockcnt).sum(axis=1)
    return data

def h4info_tbv(dr, h4s, h4e, cfg={}): 
    stockcnt=dr["sids"].shape[0]
    min1s, min1e=h4mincnt*h4s, h4mincnt*h4e
    data=dr["min1info_tbv"][min1s:min1e].reshape(-1, h4mincnt, stockcnt).sum(axis=1)
    return data

def h4info_tbm(dr, h4s, h4e, cfg={}): 
    stockcnt=dr["sids"].shape[0]
    min1s, min1e=h4mincnt*h4s, h4mincnt*h4e
    data=dr["min1info_tbm"][min1s:min1e].reshape(-1, h4mincnt, stockcnt).sum(axis=1)
    return data

def h4info_tbvr(dr, h4s, h4e, cfg={}): 
    data=dr["h4info_tbv"][h4s:h4e]/dr["h4info_volume"][h4s:h4e]
    return data

def h4info_tsv(dr, h4s, h4e, cfg={}): 
    data=dr["h4info_volume"][h4s:h4e]-dr["h4info_tbv"][h4s:h4e]
    return data

def h4info_tsm(dr, h4s, h4e, cfg={}): 
    data=dr["h4info_money"][h4s:h4e]-dr["h4info_tbm"][h4s:h4e]
    return data

def h4info_return(dr, h4s, h4e, cfg={}): 
    if h4s<=0:
        data=dr["h4info_close"][h4s:h4e]
        data[0,:]=0
        data[1:h4e]=data[1:h4e]/dr["h4info_close"][0:h4e-1]-1.0
    else:
        data=dr["h4info_close"][h4s:h4e]/dr["h4info_close"][h4s-1:h4e-1]-1.0

    return data

def h4info_dreturn(dr, h4s, h4e, cfg={}): 
    stockcnt=dr["sids"].shape[0]
    data=np.full((h4e-h4s, stockcnt), np.nan)
    for h4i in range(h4s, h4e):
        di=int(h4i*h4mincnt/daymincnt-1)
        data[h4i-h4s]=dr["h4info_close"][h4i]/dr["dayinfo_close"][di]-1.0
    return data

def h4info_npvcorr(dr, h4s, h4e, cfg={}): 
    cnt=cfg["cnt"]
    stockcnt=dr["sids"].shape[0]
    data=np.full((h4e-h4s, stockcnt), np.nan)
    for h4i in range(h4s, h4e):
        if h4i-cnt+1>=0:
            volume=dr["h4info_volume"][h4i-cnt+1:h4i+1]
            price=dr["h4info_vwap"][h4i-cnt+1:h4i+1]
            corr = pearsonr(torch.from_numpy(price), torch.from_numpy(volume), batch_first=False)
            data[h4i-h4s]=corr[0].numpy()
    return data

def h1info_open(dr, h1s, h1e, cfg={}): 
    stockcnt=dr["sids"].shape[0]
    min1s, min1e=h1mincnt*h1s, h1mincnt*h1e
    data=dr["min1info_open"][min1s:min1e][::h1mincnt]
    return data

def h1info_close(dr, h1s, h1e, cfg={}): 
    stockcnt=dr["sids"].shape[0]
    min1s, min1e=h1mincnt*h1s, h1mincnt*h1e
    data=dr["min1info_close"][min1s:min1e][h1mincnt-1::h1mincnt]
    return data

def h1info_high(dr, h1s, h1e, cfg={}): 
    stockcnt=dr["sids"].shape[0]
    min1s, min1e=h1mincnt*h1s, h1mincnt*h1e
    data=dr["min1info_high"][min1s:min1e].reshape(-1, h1mincnt, stockcnt).max(axis=1)
    return data

def h1info_low(dr, h1s, h1e, cfg={}): 
    stockcnt=dr["sids"].shape[0]
    min1s, min1e=h1mincnt*h1s, h1mincnt*h1e
    data=dr["min1info_low"][min1s:min1e].reshape(-1, h1mincnt, stockcnt).min(axis=1)
    return data

def h1info_twap(dr, h1s, h1e, cfg={}): 
    stockcnt=dr["sids"].shape[0]
    min1s, min1e=h1mincnt*h1s, h1mincnt*h1e
    data=dr["min1info_close"][min1s:min1e].reshape(-1, h1mincnt, stockcnt).mean(axis=1)
    return data

def h1info_vwap(dr, h1s, h1e, cfg={}): 
    stockcnt=dr["sids"].shape[0]
    min1s, min1e=h1mincnt*h1s, h1mincnt*h1e
    data=(dr["min1info_close"][min1s:min1e]*dr["min1info_volume"][min1s:min1e]).reshape(-1,h1mincnt,stockcnt).sum(axis=1)
    data=data/(dr["min1info_volume"][min1s:min1e].reshape(-1,h1mincnt,stockcnt).sum(axis=1))
    return data

def h1info_volume(dr, h1s, h1e, cfg={}): 
    stockcnt=dr["sids"].shape[0]
    min1s, min1e=h1mincnt*h1s, h1mincnt*h1e
    data=dr["min1info_volume"][min1s:min1e].reshape(-1, h1mincnt, stockcnt).sum(axis=1)
    return data

def h1info_tnum(dr, h1s, h1e, cfg={}): 
    stockcnt=dr["sids"].shape[0]
    min1s, min1e=h1mincnt*h1s, h1mincnt*h1e
    data=dr["min1info_tnum"][min1s:min1e].reshape(-1, h1mincnt, stockcnt).sum(axis=1)
    return data

def h1info_tbv(dr, h1s, h1e, cfg={}): 
    stockcnt=dr["sids"].shape[0]
    min1s, min1e=h1mincnt*h1s, h1mincnt*h1e
    data=dr["min1info_tbv"][min1s:min1e].reshape(-1, h1mincnt, stockcnt).sum(axis=1)
    return data

def h1info_tbvr(dr, h1s, h1e, cfg={}): 
    data=dr["h1info_tbv"][h1s:h1e]/dr["h1info_volume"][h1s:h1e]
    return data

def h1info_tbm(dr, h1s, h1e, cfg={}): 
    stockcnt=dr["sids"].shape[0]
    min1s, min1e=h1mincnt*h1s, h1mincnt*h1e
    data=dr["min1info_tbm"][min1s:min1e].reshape(-1, h1mincnt, stockcnt).sum(axis=1)
    return data

def h1info_tsv(dr, h1s, h1e, cfg={}): 
    data=dr["h1info_volume"][h1s:h1e]-dr["h1info_tbv"][h1s:h1e]
    return data

def h1info_tsm(dr, h1s, h1e, cfg={}): 
    data=dr["h1info_money"][h1s:h1e]-dr["h1info_tbm"][h1s:h1e]
    return data

def h1info_money(dr, h1s, h1e, cfg={}): 
    stockcnt=dr["sids"].shape[0]
    min1s, min1e=h1mincnt*h1s, h1mincnt*h1e
    min1info_money=(dr["min1info_open"][min1s:min1e]+dr["min1info_close"][min1s:min1e])/2.0*dr["min1info_volume"][min1s:min1e]
    data=min1info_money.reshape(-1, h1mincnt, stockcnt).sum(axis=1)
    return data 

def h1info_return(dr, h1s, h1e, cfg={}): 
    if h1s<=0:
        data=dr["h1info_close"][h1s:h1e]
        data[0,:]=0
        data[1:h1e]=data[1:h1e]/dr["h1info_close"][0:h1e-1]-1.0
    else:
        data=dr["h1info_close"][h1s:h1e]/dr["h1info_close"][h1s-1:h1e-1]-1.0

    return data

def h1info_dreturn(dr, h1s, h1e, cfg={}): 
    stockcnt=dr["sids"].shape[0]
    data=np.full((h1e-h1s, stockcnt), np.nan)
    for h1i in range(h1s, h1e):
        di=int(h1i*h1mincnt/daymincnt-1)
        data[h1i-h1s]=dr["h1info_close"][h1i]/dr["dayinfo_close"][di]-1.0
    return data

def h1info_npvcorr(dr, h1s, h1e, cfg={}): 
    cnt=cfg["cnt"]
    stockcnt=dr["sids"].shape[0]
    data=np.full((h1e-h1s, stockcnt), np.nan)
    for h1i in range(h1s, h1e):
        if h1i-cnt+1>=0:
            volume=dr["h1info_volume"][h1i-cnt+1:h1i+1]
            price=dr["h1info_vwap"][h1i-cnt+1:h1i+1]
            corr = pearsonr(torch.from_numpy(price), torch.from_numpy(volume), batch_first=False)
            data[h1i-h1s]=corr[0].numpy()
    return data

def min15info_open(dr, min15s, min15e, cfg={}): 
    stockcnt=dr["sids"].shape[0]
    min1s, min1e=min15mincnt*min15s, min15mincnt*min15e
    data=dr["min1info_open"][min1s:min1e][::min15mincnt]
    return data

def min15info_close(dr, min15s, min15e, cfg={}): 
    stockcnt=dr["sids"].shape[0]
    min1s, min1e=min15mincnt*min15s, min15mincnt*min15e
    data=dr["min1info_close"][min1s:min1e][min15mincnt-1::min15mincnt]
    return data

def min15info_high(dr, min15s, min15e, cfg={}): 
    stockcnt=dr["sids"].shape[0]
    min1s, min1e=min15mincnt*min15s, min15mincnt*min15e
    data=dr["min1info_high"][min1s:min1e].reshape(-1, min15mincnt, stockcnt).max(axis=1)
    return data

def min15info_low(dr, min15s, min15e, cfg={}): 
    stockcnt=dr["sids"].shape[0]
    min1s, min1e=min15mincnt*min15s, min15mincnt*min15e
    data=dr["min1info_low"][min1s:min1e].reshape(-1, min15mincnt, stockcnt).min(axis=1)
    return data

def min15info_twap(dr, min15s, min15e, cfg={}): 
    stockcnt=dr["sids"].shape[0]
    min1s, min1e=min15mincnt*min15s, min15mincnt*min15e
    data=dr["min1info_close"][min1s:min1e].reshape(-1, min15mincnt, stockcnt).mean(axis=1)
    return data

def min15info_vwap(dr, min15s, min15e, cfg={}): 
    stockcnt=dr["sids"].shape[0]
    min1s, min1e=min15mincnt*min15s, min15mincnt*min15e
    data=(dr["min1info_close"][min1s:min1e]*dr["min1info_volume"][min1s:min1e]).reshape(-1,min15mincnt,stockcnt).sum(axis=1)
    data=data/(dr["min1info_volume"][min1s:min1e].reshape(-1,min15mincnt,stockcnt).sum(axis=1))
    return data

def min15info_volume(dr, min15s, min15e, cfg={}): 
    stockcnt=dr["sids"].shape[0]
    min1s, min1e=min15mincnt*min15s, min15mincnt*min15e
    data=dr["min1info_volume"][min1s:min1e].reshape(-1, min15mincnt, stockcnt).sum(axis=1)
    return data

def min15info_tnum(dr, min15s, min15e, cfg={}): 
    stockcnt=dr["sids"].shape[0]
    min1s, min1e=min15mincnt*min15s, min15mincnt*min15e
    data=dr["min1info_tnum"][min1s:min1e].reshape(-1, min15mincnt, stockcnt).sum(axis=1)
    return data

def min15info_tbv(dr, min15s, min15e, cfg={}): 
    stockcnt=dr["sids"].shape[0]
    min1s, min1e=min15mincnt*min15s, min15mincnt*min15e
    data=dr["min1info_tbv"][min1s:min1e].reshape(-1, min15mincnt, stockcnt).sum(axis=1)
    return data

def min15info_tbvr(dr, min15s, min15e, cfg={}): 
    data=dr["min15info_tbv"][min15s:min15e]/dr["min15info_volume"][min15s:min15e]
    return data

def min15info_tbm(dr, min15s, min15e, cfg={}): 
    stockcnt=dr["sids"].shape[0]
    min1s, min1e=min15mincnt*min15s, min15mincnt*min15e
    data=dr["min1info_tbm"][min1s:min1e].reshape(-1, min15mincnt, stockcnt).sum(axis=1)
    return data

def min15info_tsv(dr, min15s, min15e, cfg={}): 
    data=dr["min15info_volume"][min15s:min15e]-dr["min15info_tbv"][min15s:min15e]
    return data

def min15info_tsm(dr, min15s, min15e, cfg={}): 
    data=dr["min15info_money"][min15s:min15e]-dr["min15info_tbm"][min15s:min15e]
    return data

def min15info_money(dr, min15s, min15e, cfg={}): 
    stockcnt=dr["sids"].shape[0]
    min1s, min1e=min15mincnt*min15s, min15mincnt*min15e
    min1info_money=(dr["min1info_open"][min1s:min1e]+dr["min1info_close"][min1s:min1e])/2.0*dr["min1info_volume"][min1s:min1e]
    data=min1info_money.reshape(-1, min15mincnt, stockcnt).sum(axis=1)
    return data 

def min15info_return(dr, min15s, min15e, cfg={}): 
    if min15s<=0:
        data=dr["min15info_close"][min15s:min15e]
        data[0,:]=0
        data[1:min15e]=data[1:min15e]/dr["min15info_close"][0:min15e-1]-1.0
    else:
        data=dr["min15info_close"][min15s:min15e]/dr["min15info_close"][min15s-1:min15e-1]-1.0
    return data

def min15info_dreturn(dr, min15s, min15e, cfg={}): 
    stockcnt=dr["sids"].shape[0]
    data=np.full((min15e-min15s, stockcnt), np.nan)
    for min15i in range(min15s, min15e):
        di=int(min15i*min15mincnt/daymincnt-1)
        data[min15i-min15s]=dr["min15info_close"][min15i]/dr["dayinfo_close"][di]-1.0
    return data

def min15info_npvcorr(dr, min15s, min15e, cfg={}): 
    cnt=cfg["cnt"]
    stockcnt=dr["sids"].shape[0]
    data=np.full((min15e-min15s, stockcnt), np.nan)
    for min15i in range(min15s, min15e):
        if min15i-cnt+1>=0:
            volume=dr["min15info_volume"][min15i-cnt+1:min15i+1]
            price=dr["min15info_vwap"][min15i-cnt+1:min15i+1]
            corr = pearsonr(torch.from_numpy(price), torch.from_numpy(volume), batch_first=False)
            data[min15i-min15s]=corr[0].numpy()
    return data


def min1info_twap(dr, min1s, min1e, cfg={}): 
    stockcnt=dr["sids"].shape[0]
    min1info_twap=(dr["min1info_open"][min1s:min1e]+dr["min1info_close"][min1s:min1e])/2.0
    return min1info_twap 

def min1info_vwap(dr, min1s, min1e, cfg={}): 
    stockcnt=dr["sids"].shape[0]
    min1info_vwap=(dr["min1info_money"][min1s:min1e]/dr["min1info_volume"][min1s:min1e])
    return min1info_vwap 

def min1info_tbvr(dr, min1s, min1e): 
    data=dr["min1info_tbv"][min1s:min1e]/dr["min1info_volume"][min1s:min1e]
    return data

def min1info_return(dr, min1s, min1e, cfg={}): 
    data=dr["min1info_close"][min1s:min1e]/dr["min1info_open"][min1s:min1e]-1.0
    return data

def min1info_dreturn(dr, min1s, min1e, cfg={}): 
    stockcnt=dr["sids"].shape[0]
    data=np.full((min1e-min1s, stockcnt), np.nan)
    for min1i in range(min1s, min1e):
        di=int(min1i/daymincnt-1)
        data[min1i-min1s]=dr["min1info_close"][min1i]/dr["min1info_close"][min1i-daymincnt]-1.0
    return data

def min1info_npvcorr(dr, min1s, min1e, cfg={}): 
    cnt=cfg["cnt"]
    stockcnt=dr["sids"].shape[0]
    data=np.full((min1e-min1s, stockcnt), np.nan)
    for min1i in range(min1s, min1e):
        if min1i-cnt+1>=0:
            volume=dr["min1info_volume"][min1i-cnt+1:min1i+1]
            price=dr["min1info_vwap"][min1i-cnt+1:min1i+1]
            corr = pearsonr(torch.from_numpy(price), torch.from_numpy(volume), batch_first=False)
            data[min1i-min1s]=corr[0].numpy()
    return data

def min1info_ntwap(dr, min1s, min1e, cfg={}): 
    cnt=cfg["cnt"]
    stockcnt=dr["sids"].shape[0]
    data=np.full((min1e-min1s, stockcnt), np.nan)
    for min1i in range(min1s, min1e):
        if min1i-cnt+1>=0:
            price=dr["min1info_vwap"][min1i-cnt+1:min1i+1]
            data[min1i-min1s]=price.mean(axis=0)
    return data

def min1info_nvwap(dr, min1s, min1e, cfg={}): 
    cnt=cfg["cnt"]
    stockcnt=dr["sids"].shape[0]
    data=np.full((min1e-min1s, stockcnt), np.nan)
    for min1i in range(min1s, min1e):
        if min1i-cnt+1>=0:
            price=dr["min1info_vwap"][min1i-cnt+1:min1i+1]
            vol=dr["min1info_volume"][min1i-cnt+1:min1i+1]
            data[min1i-min1s]=(price*vol).sum(axis=0)/vol.sum(axis=0)
    return data

def min1info_npstd(dr, min1s, min1e, cfg={}): 
    cnt=cfg["cnt"]
    stockcnt=dr["sids"].shape[0]
    data=np.full((min1e-min1s, stockcnt), np.nan)
    for min1i in range(min1s, min1e):
        if min1i-cnt+1>=0:
            price=dr["min1info_vwap"][min1i-cnt+1:min1i+1]
            alpha=price.std(axis=0)
            alpha=alpha-np.nanmean(alpha)
            alpha=alpha/np.nanstd(alpha)
            data[min1i-min1s]=alpha
    return data

def min1info_nmapb(dr, min1s, min1e, cfg={}): 
    cnt=cfg["cnt"]
    stockcnt=dr["sids"].shape[0]
    data=np.full((min1e-min1s, stockcnt), np.nan)
    for min1i in range(min1s, min1e):
        if min1i-cnt+1>=0:
            money=dr["min1info_money"][min1i-cnt+1:min1i+1]
            volume=dr["min1info_volume"][min1i-cnt+1:min1i+1]
            vwap=dr["min1info_vwap"][min1i-cnt+1:min1i+1]
            vwapavg=vwap.mean(axis=0)
            tvwap=money.sum(axis=0)/volume.sum(axis=0)
            alpha=np.log(vwapavg/tvwap)
            alpha=alpha-np.nanmean(alpha)
            alpha=alpha/np.nanstd(alpha)
            data[min1i-min1s]=alpha
    return data


def min1info_nprpp(dr, min1s, min1e, cfg={}): 
    cnt=cfg["cnt"]
    stockcnt=dr["sids"].shape[0]
    data=np.full((min1e-min1s, stockcnt), np.nan)
    for min1i in range(min1s, min1e):
        if min1i-cnt+1>=0:
            vwap=dr["min1info_vwap"][min1i-cnt+1:min1i+1]
            vwapret=vwap[1:]/vwap[:-1]-1.0
            retp=vwapret*(vwapret>0)
            alpha=(retp*retp).sum(axis=0)/((vwapret*vwapret).sum(axis=0))
            alpha=alpha-np.nanmean(alpha)
            alpha=alpha/np.nanstd(alpha)
            data[min1i-min1s]=alpha
    return data

def min1info_nprfp(dr, min1s, min1e, cfg={}): 
    cnt=cfg["cnt"]
    stockcnt=dr["sids"].shape[0]
    data=np.full((min1e-min1s, stockcnt), np.nan)
    for min1i in range(min1s, min1e):
        if min1i-cnt+1>=0:
            vwap=dr["min1info_vwap"][min1i-cnt+1:min1i+1]
            vwapret=vwap[1:]/vwap[:-1]-1.0
            retp=vwapret*(vwapret<0)
            alpha=(retp*retp).sum(axis=0)/((vwapret*vwapret).sum(axis=0))
            alpha=alpha-np.nanmean(alpha)
            alpha=alpha/np.nanstd(alpha)
            data[min1i-min1s]=alpha
    return data

def min1info_nrstd(dr, min1s, min1e, cfg={}): 
    cnt=cfg["cnt"]
    stockcnt=dr["sids"].shape[0]
    data=np.full((min1e-min1s, stockcnt), np.nan)
    for min1i in range(min1s, min1e):
        if min1i-cnt+1>=0:
            price=dr["min1info_return"][min1i-cnt+1:min1i+1]
            alpha=price.std(axis=0)
            alpha=alpha-np.nanmean(alpha)
            alpha=alpha/np.nanstd(alpha)
            data[min1i-min1s]=alpha
    return data

def min5info_open(dr, min5s, min5e, cfg={}): 
    stockcnt=dr["sids"].shape[0]
    min1s, min1e=min5mincnt*min5s, min5mincnt*min5e
    data=dr["min1info_open"][min1s:min1e][::min5mincnt]
    return data

def min5info_close(dr, min5s, min5e, cfg={}): 
    stockcnt=dr["sids"].shape[0]
    min1s, min1e=min5mincnt*min5s, min5mincnt*min5e
    data=dr["min1info_close"][min1s:min1e][min5mincnt-1::min5mincnt]
    return data

def min5info_return(dr, min5s, min5e, cfg={}): 
    data=dr["min5info_close"][min5s:min5e]/dr["min5info_open"][min5s:min5e]-1.0
    return data

def min5info_dreturn(dr, min5s, min5e, cfg={}): 
    stockcnt=dr["sids"].shape[0]
    data=np.full((min5e-min5s, stockcnt), np.nan)
    for min5i in range(min5s, min5e):
        di=int(min5i*min5mincnt/daymincnt-1)
        data[min5i-min5s]=dr["min5info_close"][min5i]/dr["min1info_close"][min5i*min5mincnt-daymincnt]-1.0
    return data

def min5info_nhigh(dr, min5s, min5e, cfg={}): 
    cnt=cfg["cnt"]
    stockcnt=dr["sids"].shape[0]
    data=np.full((min5e-min5s, stockcnt), np.nan)
    for min5i in range(min5s, min5e):
        min1i = min5mincnt*(min5i+1)
        if min1i-cnt>=0:
            data[min5i-min5s]=dr["min1info_high"][min1i-cnt:min1i].max(axis=0)
    return data

def min5info_nlow(dr, min5s, min5e, cfg={}): 
    cnt=cfg["cnt"]
    stockcnt=dr["sids"].shape[0]
    data=np.full((min5e-min5s, stockcnt), np.nan)
    for min5i in range(min5s, min5e):
        min1i = min5mincnt*(min5i+1)
        if min1i-cnt>=0:
            data[min5i-min5s]=dr["min1info_low"][min1i-cnt:min1i].min(axis=0)
    return data

def min5info_nvolume(dr, min5s, min5e, cfg={}): 
    cnt=cfg["cnt"]
    stockcnt=dr["sids"].shape[0]
    data=np.full((min5e-min5s, stockcnt), np.nan)
    for min5i in range(min5s, min5e):
        min1i = min5mincnt*(min5i+1)
        if min1i-cnt>=0:
            data[min5i-min5s]=dr["min1info_volume"][min1i-cnt:min1i].sum(axis=0)
    return data

def min5info_ntnum(dr, min5s, min5e, cfg={}): 
    cnt=cfg["cnt"]
    stockcnt=dr["sids"].shape[0]
    data=np.full((min5e-min5s, stockcnt), np.nan)
    for min5i in range(min5s, min5e):
        min1i = min5mincnt*(min5i+1)
        if min1i-cnt>=0:
            data[min5i-min5s]=dr["min1info_tnum"][min1i-cnt:min1i].sum(axis=0)
    return data

def min5info_ntbv(dr, min5s, min5e, cfg={}): 
    cnt=cfg["cnt"]
    stockcnt=dr["sids"].shape[0]
    data=np.full((min5e-min5s, stockcnt), np.nan)
    for min5i in range(min5s, min5e):
        min1i = min5mincnt*(min5i+1)
        if min1i-cnt>=0:
            data[min5i-min5s]=dr["min1info_tbv"][min1i-cnt:min1i].sum(axis=0)
    return data

def min5info_ntbvr(dr, min5s, min5e, cfg={}): 
    cnt=cfg["cnt"]
    stockcnt=dr["sids"].shape[0]
    data=np.full((min5e-min5s, stockcnt), np.nan)
    for min5i in range(min5s, min5e):
        min1i = min5mincnt*(min5i+1)
        if min1i-cnt>=0:
            data[min5i-min5s]=dr["min1info_tbv"][min1i-cnt:min1i].sum(axis=0)/dr["min1info_volume"][min1i-cnt:min1i].sum(axis=0)
    return data

def min5info_nmoney(dr, min5s, min5e, cfg={}): 
    cnt=cfg["cnt"]
    stockcnt=dr["sids"].shape[0]
    data=np.full((min5e-min5s, stockcnt), np.nan)
    for min5i in range(min5s, min5e):
        min1i = min5mincnt*(min5i+1)
        if min1i-cnt>=0:
            data[min5i-min5s]=dr["min1info_money"][min1i-cnt:min1i].sum(axis=0)
    return data 

def min5info_ntwap(dr, min5s, min5e, cfg={}): 
    cnt=cfg["cnt"]
    stockcnt=dr["sids"].shape[0]
    data=np.full((min5e-min5s, stockcnt), np.nan)
    for min5i in range(min5s, min5e):
        min1i = min5mincnt*(min5i+1)
        if min1i-cnt>=0:
            price=dr["min1info_vwap"][min1i-cnt:min1i]
            data[min5i-min5s]=price.mean(axis=0)
    return data

def min5info_nvwap(dr, min5s, min5e, cfg={}): 
    cnt=cfg["cnt"]
    stockcnt=dr["sids"].shape[0]
    data=np.full((min5e-min5s, stockcnt), np.nan)
    for min5i in range(min5s, min5e):
        min1i = min5mincnt*(min5i+1)
        if min1i-cnt>=0:
            price=dr["min1info_vwap"][min1i-cnt:min1i]
            vol=dr["min1info_volume"][min1i-cnt:min1i]
            data[min5i-min5s]=(price*vol).sum(axis=0)/vol.sum(axis=0)
    return data

def min5info_npvcorr(dr, min5s, min5e, cfg={}): 
    cnt=cfg["cnt"]
    stockcnt=dr["sids"].shape[0]
    data=np.full((min5e-min5s, stockcnt), np.nan)
    for min5i in range(min5s, min5e):
        min1i = min5mincnt*(min5i+1)
        if min1i-cnt>=0:
            volume=dr["min1info_volume"][min1i-cnt:min1i]
            price=dr["min1info_vwap"][min1i-cnt:min1i]
            corr = pearsonr(torch.from_numpy(price), torch.from_numpy(volume), batch_first=False)
            data[min5i-min5s]=corr[0].numpy()
    return data

def min5info_npstd(dr, min5s, min5e, cfg={}): 
    cnt=cfg["cnt"]
    stockcnt=dr["sids"].shape[0]
    data=np.full((min5e-min5s, stockcnt), np.nan)
    for min5i in range(min5s, min5e):
        min1i = min5mincnt*(min5i+1)
        if min1i-cnt>=0:
            price=dr["min1info_vwap"][min1i-cnt:min1i]
            alpha=price.std(axis=0)
            alpha=alpha-np.nanmean(alpha)
            alpha=alpha/np.nanstd(alpha)
            data[min5i-min5s]=alpha
    return data

def min5info_nmapb(dr, min5s, min5e, cfg={}): 
    cnt=cfg["cnt"]
    stockcnt=dr["sids"].shape[0]
    data=np.full((min5e-min5s, stockcnt), np.nan)
    for min5i in range(min5s, min5e):
        min1i = min5mincnt*(min5i+1)
        if min1i-cnt>=0:
            money=dr["min1info_money"][min1i-cnt:min1i]
            volume=dr["min1info_volume"][min1i-cnt:min1i]
            vwap=dr["min1info_vwap"][min1i-cnt:min1i]
            vwapavg=vwap.mean(axis=0)
            tvwap=money.sum(axis=0)/volume.sum(axis=0)
            alpha=np.log(vwapavg/tvwap)
            alpha=alpha-np.nanmean(alpha)
            alpha=alpha/np.nanstd(alpha)
            data[min5i-min5s]=alpha
    return data


def min5info_nprpp(dr, min5s, min5e, cfg={}): 
    cnt=cfg["cnt"]
    stockcnt=dr["sids"].shape[0]
    data=np.full((min5e-min5s, stockcnt), np.nan)
    for min5i in range(min5s, min5e):
        min1i = min5mincnt*(min5i+1)
        if min1i-cnt>=0:
            vwap=dr["min1info_vwap"][min1i-cnt:min1i]
            vwapret=vwap[1:]/vwap[:-1]-1.0
            retp=vwapret*(vwapret>0)
            alpha=(retp*retp).sum(axis=0)/((vwapret*vwapret).sum(axis=0))
            alpha=alpha-np.nanmean(alpha)
            alpha=alpha/np.nanstd(alpha)
            data[min5i-min5s]=alpha
    return data

def min5info_nprfp(dr, min5s, min5e, cfg={}): 
    cnt=cfg["cnt"]
    stockcnt=dr["sids"].shape[0]
    data=np.full((min5e-min5s, stockcnt), np.nan)
    for min5i in range(min5s, min5e):
        min1i = min5mincnt*(min5i+1)
        if min1i-cnt>=0:
            vwap=dr["min1info_vwap"][min1i-cnt:min1i]
            vwapret=vwap[1:]/vwap[:-1]-1.0
            retp=vwapret*(vwapret<0)
            alpha=(retp*retp).sum(axis=0)/((vwapret*vwapret).sum(axis=0))
            alpha=alpha-np.nanmean(alpha)
            alpha=alpha/np.nanstd(alpha)
            data[min5i-min5s]=alpha
    return data

def min5info_nrstd(dr, min5s, min5e, cfg={}): 
    cnt=cfg["cnt"]
    stockcnt=dr["sids"].shape[0]
    data=np.full((min5e-min5s, stockcnt), np.nan)
    for min5i in range(min5s, min5e):
        min1i = min5mincnt*(min5i+1)
        if min1i-cnt>=0:
            price=dr["min1info_return"][min1i-cnt:min1i]
            alpha=price.std(axis=0)
            alpha=alpha-np.nanmean(alpha)
            alpha=alpha/np.nanstd(alpha)
            data[min5i-min5s]=alpha
    return data

g_feafunc=dict(
    dayinfo_tbvr=dayinfo_tbvr,
    dayinfo_tsv=dayinfo_tsv,
    dayinfo_tsm=dayinfo_tsm,
    dayinfo_pre_close=dayinfo_pre_close, 
    dayinfo_return=dayinfo_return, 
             dayinfo_twap=dayinfo_twap, 
             dayinfo_vwap=dayinfo_vwap)
for cnt in [5, 10, 20]:
    key="dayinfo_"+str(cnt)+"twap"
    g_feafunc[key]=partial(dayinfo_ntwap, cfg={"cnt":cnt})
    key="dayinfo_"+str(cnt)+"vwap"
    g_feafunc[key]=partial(dayinfo_nvwap, cfg={"cnt":cnt})
    key="dayinfo_"+str(cnt)+"high"
    g_feafunc[key]=partial(dayinfo_nhigh, cfg={"cnt":cnt})
    key="dayinfo_"+str(cnt)+"low"
    g_feafunc[key]=partial(dayinfo_nlow, cfg={"cnt":cnt})
    key="dayinfo_"+str(cnt)+"volume"
    g_feafunc[key]=partial(dayinfo_nvolume, cfg={"cnt":cnt})
    key="dayinfo_"+str(cnt)+"money"
    g_feafunc[key]=partial(dayinfo_nmoney, cfg={"cnt":cnt})
    key="dayinfo_"+str(cnt)+"return"
    g_feafunc[key]=partial(dayinfo_nreturn, cfg={"cnt":cnt})
    key="dayinfo_"+str(cnt)+"pvcorr"
    g_feafunc[key]=partial(dayinfo_npvcorr, cfg={"cnt":cnt})
    key="dayinfo_"+str(cnt)+"tnum"
    g_feafunc[key]=partial(dayinfo_ntnum, cfg={"cnt":cnt})
    key="dayinfo_"+str(cnt)+"tbv"
    g_feafunc[key]=partial(dayinfo_ntbv, cfg={"cnt":cnt})
    key="dayinfo_"+str(cnt)+"tbm"
    g_feafunc[key]=partial(dayinfo_ntbm, cfg={"cnt":cnt})
    key="dayinfo_"+str(cnt)+"tsv"
    g_feafunc[key]=partial(dayinfo_ntsv, cfg={"cnt":cnt})
    key="dayinfo_"+str(cnt)+"tsm"
    g_feafunc[key]=partial(dayinfo_ntsm, cfg={"cnt":cnt})
    key="dayinfo_"+str(cnt)+"tbvr"
    g_feafunc[key]=partial(dayinfo_ntbvr, cfg={"cnt":cnt})
    
    key="h4info_"+str(cnt)+"pvcorr"
    g_feafunc[key]=partial(h4info_npvcorr, cfg={"cnt":cnt})
    key="h1info_"+str(cnt)+"pvcorr"
    g_feafunc[key]=partial(h1info_npvcorr, cfg={"cnt":cnt})
    key="min15info_"+str(cnt)+"pvcorr"
    g_feafunc[key]=partial(min15info_npvcorr, cfg={"cnt":cnt})

g_feafunc["h4info_open"]=h4info_open
g_feafunc["h4info_close"]=h4info_close
g_feafunc["h4info_high"]=h4info_high
g_feafunc["h4info_low"]=h4info_low
g_feafunc["h4info_twap"]=h4info_twap
g_feafunc["h4info_vwap"]=h4info_vwap
g_feafunc["h4info_volume"]=h4info_volume
g_feafunc["h4info_money"]=h4info_money
g_feafunc["h4info_return"]=h4info_return
g_feafunc["h4info_dreturn"]=h4info_dreturn
g_feafunc["h4info_tnum"]=h4info_tnum
g_feafunc["h4info_tbv"]=h4info_tbv
g_feafunc["h4info_tbm"]=h4info_tbm
g_feafunc["h4info_tsv"]=h4info_tsv
g_feafunc["h4info_tsm"]=h4info_tsm
g_feafunc["h4info_tbvr"]=h4info_tbvr

g_feafunc["h1info_open"]=h1info_open
g_feafunc["h1info_close"]=h1info_close
g_feafunc["h1info_high"]=h1info_high
g_feafunc["h1info_low"]=h1info_low
g_feafunc["h1info_twap"]=h1info_twap
g_feafunc["h1info_vwap"]=h1info_vwap
g_feafunc["h1info_volume"]=h1info_volume
g_feafunc["h1info_money"]=h1info_money
g_feafunc["h1info_return"]=h1info_return
g_feafunc["h1info_dreturn"]=h1info_dreturn
g_feafunc["h1info_tnum"]=h1info_tnum
g_feafunc["h1info_tbv"]=h1info_tbv
g_feafunc["h1info_tbm"]=h1info_tbm
g_feafunc["h1info_tsv"]=h1info_tsv
g_feafunc["h1info_tsm"]=h1info_tsm
g_feafunc["h1info_tbvr"]=h1info_tbvr

g_feafunc["min15info_open"]=min15info_open
g_feafunc["min15info_close"]=min15info_close
g_feafunc["min15info_high"]=min15info_high
g_feafunc["min15info_low"]=min15info_low
g_feafunc["min15info_twap"]=min15info_twap
g_feafunc["min15info_vwap"]=min15info_vwap
g_feafunc["min15info_volume"]=min15info_volume
g_feafunc["min15info_money"]=min15info_money
g_feafunc["min15info_return"]=min15info_return
g_feafunc["min15info_dreturn"]=min15info_dreturn
g_feafunc["min15info_tnum"]=min15info_tnum
g_feafunc["min15info_tbv"]=min15info_tbv
g_feafunc["min15info_tbm"]=min15info_tbm
g_feafunc["min15info_tsv"]=min15info_tsv
g_feafunc["min15info_tsm"]=min15info_tsm
g_feafunc["min15info_tbvr"]=min15info_tbvr

g_feafunc["min1info_twap"]=min1info_twap
g_feafunc["min1info_vwap"]=min1info_vwap
g_feafunc["min1info_tbvr"]=min1info_tbvr
g_feafunc["min1info_return"]=min1info_return
g_feafunc["min1info_dreturn"]=min1info_dreturn
for cnt in [5, 15,30]:
    key="min1info_"+str(cnt)+"pvcorr"
    g_feafunc[key]=partial(min1info_npvcorr, cfg={"cnt":cnt})
    key="min1info_"+str(cnt)+"twap"
    g_feafunc[key]=partial(min1info_ntwap, cfg={"cnt":cnt})
    key="min1info_"+str(cnt)+"vwap"
    g_feafunc[key]=partial(min1info_nvwap, cfg={"cnt":cnt})
    key="min1info_"+str(cnt)+"pstd"
    g_feafunc[key]=partial(min1info_npstd, cfg={"cnt":cnt})
    key="min1info_"+str(cnt)+"mapb"
    g_feafunc[key]=partial(min1info_nmapb, cfg={"cnt":cnt})
    key="min1info_"+str(cnt)+"prpp"
    g_feafunc[key]=partial(min1info_nprpp, cfg={"cnt":cnt})
    key="min1info_"+str(cnt)+"prfp"
    g_feafunc[key]=partial(min1info_nprfp, cfg={"cnt":cnt})
    key="min1info_"+str(cnt)+"rstd"
    g_feafunc[key]=partial(min1info_nrstd, cfg={"cnt":cnt})
    
g_feafunc["min5info_open"]=min5info_open
g_feafunc["min5info_close"]=min5info_close
g_feafunc["min5info_return"]=min5info_return
g_feafunc["min5info_dreturn"]=min5info_dreturn
for cnt in [5, 15, 30, 60]:
    key="min5info_"+str(cnt)+"high"
    g_feafunc[key]=partial(min5info_nhigh, cfg={"cnt":cnt})
    key="min5info_"+str(cnt)+"low"
    g_feafunc[key]=partial(min5info_nlow, cfg={"cnt":cnt})
    key="min5info_"+str(cnt)+"volume"
    g_feafunc[key]=partial(min5info_nvolume, cfg={"cnt":cnt})
    key="min5info_"+str(cnt)+"money"
    g_feafunc[key]=partial(min5info_nmoney, cfg={"cnt":cnt})
    key="min5info_"+str(cnt)+"tbv"
    g_feafunc[key]=partial(min5info_ntbv, cfg={"cnt":cnt})
    key="min5info_"+str(cnt)+"tbvr"
    g_feafunc[key]=partial(min5info_ntbvr, cfg={"cnt":cnt})
    key="min5info_"+str(cnt)+"tnum"
    g_feafunc[key]=partial(min5info_ntnum, cfg={"cnt":cnt})
    key="min5info_"+str(cnt)+"twap"
    g_feafunc[key]=partial(min5info_ntwap, cfg={"cnt":cnt})
    key="min5info_"+str(cnt)+"vwap"
    g_feafunc[key]=partial(min5info_nvwap, cfg={"cnt":cnt})
    key="min5info_"+str(cnt)+"pvcorr"
    g_feafunc[key]=partial(min5info_npvcorr, cfg={"cnt":cnt})
    key="min5info_"+str(cnt)+"pstd"
    g_feafunc[key]=partial(min5info_npstd, cfg={"cnt":cnt})
    key="min5info_"+str(cnt)+"mapb"
    g_feafunc[key]=partial(min5info_nmapb, cfg={"cnt":cnt})
    key="min5info_"+str(cnt)+"prpp"
    g_feafunc[key]=partial(min5info_nprpp, cfg={"cnt":cnt})
    key="min5info_"+str(cnt)+"prfp"
    g_feafunc[key]=partial(min5info_nprfp, cfg={"cnt":cnt})
    key="min5info_"+str(cnt)+"rstd"
    g_feafunc[key]=partial(min5info_nrstd, cfg={"cnt":cnt})








