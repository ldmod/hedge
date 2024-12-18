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
import cryptoqt.data.sec_klines.sec_klines as sk 
import h5py

def readh5res(secdr, tm, path="/home/prod/cryptoqt/prm/model_states/infmodel/prmv3/res/res.h5"):
    f=h5py.File(path, "r", swmr=True)
    f["tm"].refresh()
    ridx=np.where(f["tm"][:]==tm)[0]
    if ridx.shape[0] <0 :
        f.close()
        print("read res fail:", tm, path, flush=True)
        return None
    ridx=ridx[0]
    dd={}
    for key in f.keys():
        if key != "tm":
            f[key].refresh()
            dd[key]=f[key][ridx].astype(np.float64)
    f.close()
    return dd

def s1info_bvolume(secdr, s1s, s1e): 
    data=(secdr["s1info_volume"][s1s:s1e]-secdr["s1info_svolume"][s1s:s1e])
    return data
def s1info_bmoney(secdr, s1s, s1e): 
    data=(secdr["s1info_money"][s1s:s1e]-secdr["s1info_smoney"][s1s:s1e])
    return data

def s1info_bvr(secdr, s1s, s1e): 
    data=secdr["s1info_bvolume"][s1s:s1e]/secdr["s1info_volume"][s1s:s1e]
    return data

def s1info_bmr(secdr, s1s, s1e): 
    data=secdr["s1info_bmoney"][s1s:s1e]/secdr["s1info_money"][s1s:s1e]
    return data

def s1info_return(secdr, s1s, s1e, cfg={}): 
    data=secdr["s1info_vwap"][s1s:s1e]/secdr["s1info_vwap"][s1s-1:s1e-1]-1.0
    data*=10000.0
    return data

def s1info_breturn(secdr, s1s, s1e, cfg={}): 
    data=secdr["s1info_bvwap"][s1s:s1e]/secdr["s1info_bvwap"][s1s-1:s1e-1]-1.0
    data*=10000.0
    return data

def s1info_sreturn(secdr, s1s, s1e, cfg={}): 
    data=secdr["s1info_svwap"][s1s:s1e]/secdr["s1info_svwap"][s1s-1:s1e-1]-1.0
    data*=10000.0
    return data

def s1info_spdiff(secdr, s1s, s1e, cfg={}): 
    data=secdr["s1info_shigh"][s1s:s1e]/secdr["s1info_slow"][s1s:s1e]-1.0
    data*=10000.0
    return data

def s1info_bpdiff(secdr, s1s, s1e, cfg={}): 
    data=secdr["s1info_bhigh"][s1s:s1e]/secdr["s1info_blow"][s1s:s1e]-1.0
    data*=10000.0
    return data

def s1info_bspdiff(secdr, s1s, s1e, cfg={}): 
    data=secdr["s1info_bvwap"][s1s:s1e]/secdr["s1info_svwap"][s1s:s1e]-1.0
    data*=10000.0
    return data

def s1info_npvcorr(secdr, s1s, s1e, cfg={}): 
    cnt=cfg["cnt"]
    stockcnt=secdr["s1info_vwap"].shape[1]
    data=torch.full((s1e-s1s, stockcnt), torch.nan, device='cuda')
    for s1i in range(s1s, s1e):
        if s1i-cnt+1>=0:
            volume=secdr["s1info_volume"][s1i-cnt+1:s1i+1]
            price=secdr["s1info_vwap"][s1i-cnt+1:s1i+1]
            corr = pearsonr(price, volume, batch_first=False)
            data[s1i-s1s]=corr[0]
    return data

def s1info_ntwap(secdr, s1s, s1e, cfg={}): 
    cnt=cfg["cnt"]
    stockcnt=secdr["s1info_vwap"].shape[1]
    data=torch.full((s1e-s1s, stockcnt), torch.nan, device='cuda')
    for s1i in range(s1s, s1e):
        if s1i-cnt+1>=0:
            price=secdr["s1info_vwap"][s1i-cnt+1:s1i+1]
            data[s1i-s1s]=price.mean(dim=0)
    return data

def s1info_nstwap(secdr, s1s, s1e, cfg={}): 
    cnt=cfg["cnt"]
    stockcnt=secdr["s1info_vwap"].shape[1]
    data=torch.full((s1e-s1s, stockcnt), torch.nan, device='cuda')
    for s1i in range(s1s, s1e):
        if s1i-cnt+1>=0:
            price=secdr["s1info_svwap"][s1i-cnt+1:s1i+1]
            data[s1i-s1s]=price.mean(dim=0)
    return data

def s1info_nbtwap(secdr, s1s, s1e, cfg={}): 
    cnt=cfg["cnt"]
    stockcnt=secdr["s1info_vwap"].shape[1]
    data=torch.full((s1e-s1s, stockcnt), torch.nan, device='cuda')
    for s1i in range(s1s, s1e):
        if s1i-cnt+1>=0:
            price=secdr["s1info_bvwap"][s1i-cnt+1:s1i+1]
            data[s1i-s1s]=price.mean(dim=0)
    return data

def s1info_nshigh(secdr, s1s, s1e, cfg={}): 
    cnt=cfg["cnt"]
    stockcnt=secdr["s1info_vwap"].shape[1]
    data=torch.full((s1e-s1s, stockcnt), torch.nan, device='cuda')
    for s1i in range(s1s, s1e):
        if s1i-cnt+1>=0:
            price=secdr["s1info_shigh"][s1i-cnt+1:s1i+1]
            data[s1i-s1s]=price.max(dim=0)[0]
    return data

def s1info_nslow(secdr, s1s, s1e, cfg={}): 
    cnt=cfg["cnt"]
    stockcnt=secdr["s1info_vwap"].shape[1]
    data=torch.full((s1e-s1s, stockcnt), torch.nan, device='cuda')
    for s1i in range(s1s, s1e):
        if s1i-cnt+1>=0:
            price=secdr["s1info_slow"][s1i-cnt+1:s1i+1]
            data[s1i-s1s]=price.min(dim=0)[0]
    return data

def s1info_nbhigh(secdr, s1s, s1e, cfg={}): 
    cnt=cfg["cnt"]
    stockcnt=secdr["s1info_vwap"].shape[1]
    data=torch.full((s1e-s1s, stockcnt), torch.nan, device='cuda')
    for s1i in range(s1s, s1e):
        if s1i-cnt+1>=0:
            price=secdr["s1info_bhigh"][s1i-cnt+1:s1i+1]
            data[s1i-s1s]=price.max(dim=0)[0]
    return data

def s1info_nblow(secdr, s1s, s1e, cfg={}): 
    cnt=cfg["cnt"]
    stockcnt=secdr["s1info_vwap"].shape[1]
    data=torch.full((s1e-s1s, stockcnt), torch.nan, device='cuda')
    for s1i in range(s1s, s1e):
        if s1i-cnt+1>=0:
            price=secdr["s1info_blow"][s1i-cnt+1:s1i+1]
            data[s1i-s1s]=price.min(dim=0)[0]
    return data

def s1info_nvolume(secdr, s1s, s1e, cfg={}): 
    cnt=cfg["cnt"]
    stockcnt=secdr["s1info_vwap"].shape[1]
    data=torch.full((s1e-s1s, stockcnt), torch.nan, device='cuda')
    for s1i in range(s1s, s1e):
        if s1i-cnt+1>=0:
            price=secdr["s1info_volume"][s1i-cnt+1:s1i+1]
            data[s1i-s1s]=price.sum(dim=0)
    return data

def s1info_nsvolume(secdr, s1s, s1e, cfg={}): 
    cnt=cfg["cnt"]
    stockcnt=secdr["s1info_vwap"].shape[1]
    data=torch.full((s1e-s1s, stockcnt), torch.nan, device='cuda')
    for s1i in range(s1s, s1e):
        if s1i-cnt+1>=0:
            price=secdr["s1info_svolume"][s1i-cnt+1:s1i+1]
            data[s1i-s1s]=price.sum(dim=0)
    return data

def s1info_nbvolume(secdr, s1s, s1e, cfg={}): 
    cnt=cfg["cnt"]
    stockcnt=secdr["s1info_vwap"].shape[1]
    data=torch.full((s1e-s1s, stockcnt), torch.nan, device='cuda')
    for s1i in range(s1s, s1e):
        if s1i-cnt+1>=0:
            price=secdr["s1info_bvolume"][s1i-cnt+1:s1i+1]
            data[s1i-s1s]=price.sum(dim=0)
    return data

def s1info_nmoney(secdr, s1s, s1e, cfg={}): 
    cnt=cfg["cnt"]
    stockcnt=secdr["s1info_vwap"].shape[1]
    data=torch.full((s1e-s1s, stockcnt), torch.nan, device='cuda')
    for s1i in range(s1s, s1e):
        if s1i-cnt+1>=0:
            price=secdr["s1info_money"][s1i-cnt+1:s1i+1]
            data[s1i-s1s]=price.sum(dim=0)
    return data

def s1info_nsmoney(secdr, s1s, s1e, cfg={}): 
    cnt=cfg["cnt"]
    stockcnt=secdr["s1info_vwap"].shape[1]
    data=torch.full((s1e-s1s, stockcnt), torch.nan, device='cuda')
    for s1i in range(s1s, s1e):
        if s1i-cnt+1>=0:
            price=secdr["s1info_smoney"][s1i-cnt+1:s1i+1]
            data[s1i-s1s]=price.sum(dim=0)
    return data

def s1info_nbmoney(secdr, s1s, s1e, cfg={}): 
    cnt=cfg["cnt"]
    stockcnt=secdr["s1info_vwap"].shape[1]
    data=torch.full((s1e-s1s, stockcnt), torch.nan, device='cuda')
    for s1i in range(s1s, s1e):
        if s1i-cnt+1>=0:
            price=secdr["s1info_bmoney"][s1i-cnt+1:s1i+1]
            data[s1i-s1s]=price.sum(dim=0)
    return data

def s1info_nvwap(secdr, s1s, s1e, cfg={}): 
    cnt=cfg["cnt"]
    stockcnt=secdr["s1info_vwap"].shape[1]
    data=torch.full((s1e-s1s, stockcnt), torch.nan, device='cuda')
    for s1i in range(s1s, s1e):
        if s1i-cnt+1>=0:
            money=secdr["s1info_money"][s1i-cnt+1:s1i+1]
            vol=secdr["s1info_volume"][s1i-cnt+1:s1i+1]
            data[s1i-s1s]=money.sum(dim=0)/vol.sum(dim=0)

            flag=(vol.sum(dim=0)==0)
            if s1i == s1s:
                data[s1i-s1s][flag]=secdr["s1info_"+str(cnt)+"vwap"][s1s-1][flag]
            else:
                data[s1i-s1s][flag]=data[s1i-s1s-1][flag]
    return data

def s1info_nsvwap(secdr, s1s, s1e, cfg={}): 
    cnt=cfg["cnt"]
    stockcnt=secdr["s1info_vwap"].shape[1]
    data=torch.full((s1e-s1s, stockcnt), torch.nan, device='cuda')
    for s1i in range(s1s, s1e):
        if s1i-cnt+1>=0:
            money=secdr["s1info_smoney"][s1i-cnt+1:s1i+1]
            vol=secdr["s1info_svolume"][s1i-cnt+1:s1i+1]
            data[s1i-s1s]=money.sum(dim=0)/vol.sum(dim=0)

            flag=(vol.sum(dim=0)==0)
            if s1i == s1s:
                data[s1i-s1s][flag]=secdr["s1info_"+str(cnt)+"svwap"][s1s-1][flag]
            else:
                data[s1i-s1s][flag]=data[s1i-s1s-1][flag]
    return data

def s1info_nbvwap(secdr, s1s, s1e, cfg={}): 
    cnt=cfg["cnt"]
    stockcnt=secdr["s1info_vwap"].shape[1]
    data=torch.full((s1e-s1s, stockcnt), torch.nan, device='cuda')
    for s1i in range(s1s, s1e):
        if s1i-cnt+1>=0:
            money=secdr["s1info_bmoney"][s1i-cnt+1:s1i+1]
            vol=secdr["s1info_bvolume"][s1i-cnt+1:s1i+1]
            data[s1i-s1s]=money.sum(dim=0)/vol.sum(dim=0)
            flag=(vol.sum(dim=0)==0)
            if s1i == s1s:
                data[s1i-s1s][flag]=secdr["s1info_"+str(cnt)+"bvwap"][s1s-1][flag]
            else:
                data[s1i-s1s][flag]=data[s1i-s1s-1][flag]
    return data

def s1info_npstd(secdr, s1s, s1e, cfg={}): 
    #section fea
    cnt=cfg["cnt"]
    stockcnt=secdr["s1info_vwap"].shape[1]
    data=torch.full((s1e-s1s, stockcnt), torch.nan, device='cuda')
    for s1i in range(s1s, s1e):
        if s1i-cnt+1>=0:
            price=secdr["s1info_vwap"][s1i-cnt+1:s1i+1]
            alpha=price.std(dim=0)
            data[s1i-s1s]=alpha
    return data


def s1info_nmapb(secdr, s1s, s1e, cfg={}): 
    cnt=cfg["cnt"]
    stockcnt=secdr["s1info_vwap"].shape[1]
    data=torch.full((s1e-s1s, stockcnt), torch.nan, device='cuda')
    for s1i in range(s1s, s1e):
        if s1i-cnt+1>=0:
            money=secdr["s1info_money"][s1i-cnt+1:s1i+1]
            volume=secdr["s1info_volume"][s1i-cnt+1:s1i+1]
            vwap=secdr["s1info_vwap"][s1i-cnt+1:s1i+1]
            vwapavg=vwap.mean(dim=0) #zheliyouyiwen,henduo shi bu zhi
            tvwap=money.sum(dim=0)/volume.sum(dim=0)
            alpha=torch.log(vwapavg/tvwap)
            data[s1i-s1s]=alpha
    return data


def s1info_nprpp(secdr, s1s, s1e, cfg={}): 
    cnt=cfg["cnt"]
    stockcnt=secdr["s1info_vwap"].shape[1]
    data=torch.full((s1e-s1s, stockcnt), torch.nan, device='cuda')
    for s1i in range(s1s, s1e):
        if s1i-cnt+1>=0:
            vwap=secdr["s1info_vwap"][s1i-cnt+1:s1i+1]
            vwapret=vwap[1:]/vwap[:-1]-1.0
            retp=vwapret*(vwapret>0)
            alpha=(retp*retp).sum(dim=0)/((vwapret*vwapret).sum(dim=0))
            data[s1i-s1s]=alpha
    return data

def s1info_nprfp(secdr, s1s, s1e, cfg={}): 
    cnt=cfg["cnt"]
    stockcnt=secdr["s1info_vwap"].shape[1]
    data=torch.full((s1e-s1s, stockcnt), np.nan)
    for s1i in range(s1s, s1e):
        if s1i-cnt+1>=0:
            vwap=secdr["s1info_vwap"][s1i-cnt+1:s1i+1]
            vwapret=vwap[1:]/vwap[:-1]-1.0
            retp=vwapret*(vwapret<0)
            alpha=(retp*retp).sum(dim=0)/((vwapret*vwapret).sum(dim=0))
            data[s1i-s1s]=alpha
    return data

def s1info_nrstd(secdr, s1s, s1e, cfg={}): 
    cnt=cfg["cnt"]
    stockcnt=secdr["s1info_vwap"].shape[1]
    data=torch.full((s1e-s1s, stockcnt), torch.nan, device='cuda')
    for s1i in range(s1s, s1e):
        if s1i-cnt+1>=0:
            price=secdr["s1info_return"][s1i-cnt+1:s1i+1]
            alpha=price.std(dim=0)
            data[s1i-s1s]=alpha
    return data

g_feafunc_first=dict()

g_feafunc=dict(
            s1info_bvolume=s1info_bvolume,
            s1info_bmoney=s1info_bmoney,
            s1info_bvr=s1info_bvr,
            s1info_bmr=s1info_bmr,
            s1info_spdiff=s1info_spdiff,
            s1info_bpdiff=s1info_bpdiff,
            s1info_bspdiff=s1info_bspdiff,
            s1info_return=s1info_return,
               s1info_breturn=s1info_breturn,
               s1info_sreturn=s1info_sreturn,)
for cnt in [5, 15,30]:
    key="s1info_"+str(cnt)+"pvcorr"
    g_feafunc[key]=partial(s1info_npvcorr, cfg={"cnt":cnt})
    key="s1info_"+str(cnt)+"twap"
    g_feafunc[key]=partial(s1info_ntwap, cfg={"cnt":cnt})
    key="s1info_"+str(cnt)+"stwap" 
    g_feafunc[key]=partial(s1info_nstwap, cfg={"cnt":cnt})
    key="s1info_"+str(cnt)+"btwap"
    g_feafunc[key]=partial(s1info_nbtwap, cfg={"cnt":cnt})
    key="s1info_"+str(cnt)+"vwap"
    g_feafunc[key]=partial(s1info_nvwap, cfg={"cnt":cnt})
    key="s1info_"+str(cnt)+"svwap"
    g_feafunc[key]=partial(s1info_nsvwap, cfg={"cnt":cnt})
    key="s1info_"+str(cnt)+"bvwap"
    g_feafunc[key]=partial(s1info_nbvwap, cfg={"cnt":cnt})
    key="s1info_"+str(cnt)+"bhigh"
    g_feafunc[key]=partial(s1info_nbhigh, cfg={"cnt":cnt})
    key="s1info_"+str(cnt)+"shigh"
    g_feafunc[key]=partial(s1info_nshigh, cfg={"cnt":cnt})
    key="s1info_"+str(cnt)+"blow"
    g_feafunc[key]=partial(s1info_nblow, cfg={"cnt":cnt})
    key="s1info_"+str(cnt)+"slow"
    g_feafunc[key]=partial(s1info_nslow, cfg={"cnt":cnt})
    key="s1info_"+str(cnt)+"volume"
    g_feafunc[key]=partial(s1info_nvolume, cfg={"cnt":cnt})
    key="s1info_"+str(cnt)+"bvolume"
    g_feafunc[key]=partial(s1info_nbvolume, cfg={"cnt":cnt})
    key="s1info_"+str(cnt)+"svolume"
    g_feafunc[key]=partial(s1info_nsvolume, cfg={"cnt":cnt})
    key="s1info_"+str(cnt)+"money"
    g_feafunc[key]=partial(s1info_nmoney, cfg={"cnt":cnt})
    key="s1info_"+str(cnt)+"bmoney"
    g_feafunc[key]=partial(s1info_nbmoney, cfg={"cnt":cnt})
    key="s1info_"+str(cnt)+"smoney"
    g_feafunc[key]=partial(s1info_nsmoney, cfg={"cnt":cnt})
    
    key="s1info_"+str(cnt)+"pstd"
    g_feafunc[key]=partial(s1info_npstd, cfg={"cnt":cnt})
    key="s1info_"+str(cnt)+"mapb"
    g_feafunc[key]=partial(s1info_nmapb, cfg={"cnt":cnt})
    key="s1info_"+str(cnt)+"prpp"
    g_feafunc[key]=partial(s1info_nprpp, cfg={"cnt":cnt})
    key="s1info_"+str(cnt)+"prfp"
    g_feafunc[key]=partial(s1info_nprfp, cfg={"cnt":cnt})
    key="s1info_"+str(cnt)+"rstd"
    g_feafunc[key]=partial(s1info_nrstd, cfg={"cnt":cnt})
    








