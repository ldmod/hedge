#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 13:32:20 2024

@author: ld
"""
import numpy as np
import cryptoqt.data.constants as conts
import cryptoqt.data.tools as tools
from audtorch.metrics.functional import pearsonr
import torch
import scipy.stats as stats
import pandas as pd
daymincnt=conts.daymincnt
h4mincnt=conts.h4mincnt
h1mincnt=conts.h1mincnt
min15mincnt=conts.min15mincnt

def readcsv(dr, min1i, path="/home/crypto/cryptoqt/smlp/model_states/infmodel/tsmlp15/res"):
    di=int(min1i/conts.daymincnt)
    h4i=int(min1i/conts.h4mincnt)
    h1i=int(min1i/conts.h1mincnt)
    min15i=int(min1i/conts.min15mincnt) 
    fname=path+"/"+str(dr["min1info_tm"][min1i])+"_pred.csv"
    df=pd.read_csv(fname)
    alpha=df["pred0"].to_numpy()
    return alpha

def readcsv_v2(dr, min1i, path="/home/crypto/smlpv2/cryptoqt/smlp/model_states/infmodel/tsmlpv215/res"):
    di=int(min1i/conts.daymincnt)
    h4i=int(min1i/conts.h4mincnt)
    h1i=int(min1i/conts.h1mincnt)
    min15i=int(min1i/conts.min15mincnt) 
    fname=path+"/"+str(dr["min1info_tm"][min1i])+"_pred.csv"
    df=pd.read_csv(fname)
    alpha=df["pred0"].to_numpy()
    return alpha

def readcsv_v2avg(dr, min1i, path="/home/crypto/smlpv2/cryptoqt/smlp/model_states/infmodel/tsmlpv215/res", fields = ["pred0", "pred1", "pred2"]):
    di=int(min1i/conts.daymincnt)
    h4i=int(min1i/conts.h4mincnt)
    h1i=int(min1i/conts.h1mincnt)
    min15i=int(min1i/conts.min15mincnt) 
    fname=path+"/"+str(dr["min1info_tm"][min1i])+"_pred.csv"
    df=pd.read_csv(fname)
    alpha=np.zeros(df["pred0"].shape)
    for field in fields:
        alpha+=tools.npnorm(df[field].to_numpy())
    return alpha

def readcsv_test(dr, min1i, path="/home/crypto/smlpv2/cryptoqt/smlp/model_states/infmodel/tsmlpv215/res"):
    di=int(min1i/conts.daymincnt)
    h4i=int(min1i/conts.h4mincnt)
    h1i=int(min1i/conts.h1mincnt)
    min15i=int(min1i/conts.min15mincnt) 
    fname=path+"/"+str(dr["min1info_tm"][min1i])+"_pred.csv"
    df=pd.read_csv(fname)
    alpha=df["pred0"].to_numpy()+df["pred1"].to_numpy()+df["pred2"].to_numpy()
    return alpha


def nmin15r5(dr, min1i):
    di=int(min1i/conts.daymincnt)
    h4i=int(min1i/conts.h4mincnt)
    h1i=int(min1i/conts.h1mincnt)
    min15i=int(min1i/conts.min15mincnt) 
    delay=5
    dclose=dr["min15info_close"]
    ret5day=dclose[min15i-1]/dclose[min15i-delay-1]-1.0
    alpha=ret5day-np.nanmean(ret5day)
    alpha=alpha/np.nanstd(alpha)
    return -1.0*alpha

def nmin15r1(dr, min1i):
    di=int(min1i/conts.daymincnt)
    h4i=int(min1i/conts.h4mincnt)
    h1i=int(min1i/conts.h1mincnt)
    min15i=int(min1i/conts.min15mincnt) 
    delay=1
    dclose=dr["min15info_close"]
    ret5day=dclose[min15i-1]/dclose[min15i-delay-1]-1.0
    alpha=ret5day-np.nanmean(ret5day)
    alpha=alpha/np.nanstd(alpha)
    return -1.0*alpha

def dli_15mincorr(dr, min1i):
    di=int(min1i/conts.daymincnt)
    h4i=int(min1i/conts.h4mincnt)
    h1i=int(min1i/conts.h1mincnt)
    min15i=int(min1i/conts.min15mincnt) 
    delta=15
    money=dr["min1info_money"][min1i-delta:min1i]
    volume=dr["min1info_volume"][min1i-delta:min1i]
    vwap=money/volume
    alpha=pearsonr(torch.from_numpy(vwap), torch.from_numpy(volume), batch_first=False).numpy()[0]
    alpha=alpha/np.nanstd(alpha) 
    alpha[~np.isfinite(alpha)]=0
    return -1.0*alpha

def dli_60mincorr(dr, min1i):
    di=int(min1i/conts.daymincnt)
    h4i=int(min1i/conts.h4mincnt)
    h1i=int(min1i/conts.h1mincnt)
    min15i=int(min1i/conts.min15mincnt) 
    delta=60
    money=dr["min1info_money"][min1i-delta:min1i]
    volume=dr["min1info_volume"][min1i-delta:min1i]
    vwap=money/volume
    alpha=pearsonr(torch.from_numpy(vwap), torch.from_numpy(volume), batch_first=False).numpy()[0]
    alpha=alpha/np.nanstd(alpha) 
    alpha[~np.isfinite(alpha)]=0
    return -1.0*alpha

def dli_15mapb(dr, min1i):
    di=int(min1i/conts.daymincnt)
    h4i=int(min1i/conts.h4mincnt)
    h1i=int(min1i/conts.h1mincnt)
    min15i=int(min1i/conts.min15mincnt) 
    delta=15
    money=dr["min1info_money"][min1i-delta:min1i]
    volume=dr["min1info_volume"][min1i-delta:min1i]
    vwap=money/volume
    vwapavg=vwap.mean(axis=0)
    tvwap=money.sum(axis=0)/volume.sum(axis=0)
    alpha=np.log(vwapavg/tvwap)
    alpha[~np.isfinite(alpha)]=0
    return alpha

def dli_60mapb(dr, min1i):
    di=int(min1i/conts.daymincnt)
    h4i=int(min1i/conts.h4mincnt)
    h1i=int(min1i/conts.h1mincnt)
    min15i=int(min1i/conts.min15mincnt) 
    delta=60
    money=dr["min1info_money"][min1i-delta:min1i]
    volume=dr["min1info_volume"][min1i-delta:min1i]
    vwap=money/volume
    vwapavg=vwap.mean(axis=0)
    tvwap=money.sum(axis=0)/volume.sum(axis=0)
    alpha=np.log(vwapavg/tvwap)
    alpha[~np.isfinite(alpha)]=0
    return alpha

def dli_15marpp(dr, min1i):
    di=int(min1i/conts.daymincnt)
    h4i=int(min1i/conts.h4mincnt)
    h1i=int(min1i/conts.h1mincnt)
    min15i=int(min1i/conts.min15mincnt) 
    delta=15
    money=dr["min1info_money"][min1i-delta:min1i]
    volume=dr["min1info_volume"][min1i-delta:min1i]
    high=dr["min1info_high"][min1i-delta:min1i].max(axis=0)
    low=dr["min1info_low"][min1i-delta:min1i].min(axis=0)
    vwap=money/volume
    vwapavg=vwap.mean(axis=0)
    rp=(vwap-low)/(high-low)
    alpha=(vwapavg-low)/(high-low)
    alpha=alpha-np.nanmean(alpha)
    alpha=alpha/np.nanstd(alpha)
    alpha[~np.isfinite(alpha)]=0
    return alpha

def dli_60marpp(dr, min1i):
    di=int(min1i/conts.daymincnt)
    h4i=int(min1i/conts.h4mincnt)
    h1i=int(min1i/conts.h1mincnt)
    min15i=int(min1i/conts.min15mincnt) 
    delta=60
    money=dr["min1info_money"][min1i-delta:min1i]
    volume=dr["min1info_volume"][min1i-delta:min1i]
    high=dr["min1info_high"][min1i-delta:min1i].max(axis=0)
    low=dr["min1info_low"][min1i-delta:min1i].min(axis=0)
    vwap=money/volume
    vwapavg=vwap.mean(axis=0)
    rp=(vwap-low)/(high-low)
    alpha=(vwapavg-low)/(high-low)
    alpha=alpha-np.nanmean(alpha)
    alpha=alpha/np.nanstd(alpha)
    alpha[~np.isfinite(alpha)]=0
    return alpha

def dli_15mtrend(dr, min1i):
    di=int(min1i/conts.daymincnt)
    h4i=int(min1i/conts.h4mincnt)
    h1i=int(min1i/conts.h1mincnt)
    min15i=int(min1i/conts.min15mincnt) 
    delta=15
    money=dr["min1info_money"][min1i-delta:min1i]
    volume=dr["min1info_volume"][min1i-delta:min1i]
    vwap=money/volume
    start=vwap[0]
    end=vwap[-1]
    lsum=np.abs(vwap[1:]-vwap[:-1]).sum(axis=0)
    trend=(end-start)/lsum
    alpha=trend
    alpha=alpha-np.nanmean(alpha)
    alpha=alpha/np.nanstd(alpha)
    alpha[~np.isfinite(alpha)]=0
    return -1.0*alpha

def dli_60mtrend(dr, min1i):
    di=int(min1i/conts.daymincnt)
    h4i=int(min1i/conts.h4mincnt)
    h1i=int(min1i/conts.h1mincnt)
    min15i=int(min1i/conts.min15mincnt) 
    delta=60
    money=dr["min1info_money"][min1i-delta:min1i]
    volume=dr["min1info_volume"][min1i-delta:min1i]
    vwap=money/volume
    start=vwap[0]
    end=vwap[-1]
    lsum=np.abs(vwap[1:]-vwap[:-1]).sum(axis=0)
    trend=(end-start)/lsum
    alpha=trend
    alpha=alpha-np.nanmean(alpha)
    alpha=alpha/np.nanstd(alpha)
    alpha[~np.isfinite(alpha)]=0
    return -1.0*alpha


def dli_15pstd(dr, min1i):
    di=int(min1i/conts.daymincnt)
    h4i=int(min1i/conts.h4mincnt)
    h1i=int(min1i/conts.h1mincnt)
    min15i=int(min1i/conts.min15mincnt) 
    delta=15
    money=dr["min1info_money"][min1i-delta:min1i]
    volume=dr["min1info_volume"][min1i-delta:min1i]
    vwap=money/volume
    vwapret=vwap[1:]/vwap[:-1]-1.0
    alpha=vwapret.std(axis=0)
    alpha=alpha-np.nanmean(alpha)
    alpha=alpha/np.nanstd(alpha)
    alpha[~np.isfinite(alpha)]=0
    return -1.0*alpha

def dli_60pstd(dr, min1i):
    di=int(min1i/conts.daymincnt)
    h4i=int(min1i/conts.h4mincnt)
    h1i=int(min1i/conts.h1mincnt)
    min15i=int(min1i/conts.min15mincnt) 
    delta=60
    money=dr["min1info_money"][min1i-delta:min1i]
    volume=dr["min1info_volume"][min1i-delta:min1i]
    vwap=money/volume
    vwapret=vwap[1:]/vwap[:-1]-1.0
    alpha=vwapret.std(axis=0)
    alpha=alpha-np.nanmean(alpha)
    alpha=alpha/np.nanstd(alpha)
    alpha[~np.isfinite(alpha)]=0
    return -1.0*alpha

def dli_15vstd(dr, min1i):
    di=int(min1i/conts.daymincnt)
    h4i=int(min1i/conts.h4mincnt)
    h1i=int(min1i/conts.h1mincnt)
    min15i=int(min1i/conts.min15mincnt) 
    delta=15
    volume=dr["min1info_volume"][min1i-delta:min1i]
    alpha=volume.std(axis=0)
    alpha=alpha-np.nanmean(alpha)
    alpha=alpha/np.nanstd(alpha)
    alpha[~np.isfinite(alpha)]=0
    return -1.0*alpha

def dli_60vstd(dr, min1i):
    di=int(min1i/conts.daymincnt)
    h4i=int(min1i/conts.h4mincnt)
    h1i=int(min1i/conts.h1mincnt)
    min15i=int(min1i/conts.min15mincnt) 
    delta=60
    volume=dr["min1info_volume"][min1i-delta:min1i]
    alpha=volume.std(axis=0)
    alpha=alpha-np.nanmean(alpha)
    alpha=alpha/np.nanstd(alpha)
    alpha[~np.isfinite(alpha)]=0
    return -1.0*alpha

def dli_15pskew(dr, min1i):
    di=int(min1i/conts.daymincnt)
    h4i=int(min1i/conts.h4mincnt)
    h1i=int(min1i/conts.h1mincnt)
    min15i=int(min1i/conts.min15mincnt) 
    delta=15
    money=dr["min1info_money"][min1i-delta:min1i]
    volume=dr["min1info_volume"][min1i-delta:min1i]
    vwap=money/volume
    vwapret=vwap[1:]/vwap[:-1]-1.0
    alpha=stats.skew(vwapret,axis=0)
    alpha=alpha-np.nanmean(alpha)
    alpha=alpha/np.nanstd(alpha)
    alpha[~np.isfinite(alpha)]=0
    return -1.0*alpha

def dli_60pskew(dr, min1i):
    di=int(min1i/conts.daymincnt)
    h4i=int(min1i/conts.h4mincnt)
    h1i=int(min1i/conts.h1mincnt)
    min15i=int(min1i/conts.min15mincnt) 
    delta=60
    money=dr["min1info_money"][min1i-delta:min1i]
    volume=dr["min1info_volume"][min1i-delta:min1i]
    vwap=money/volume
    vwapret=vwap[1:]/vwap[:-1]-1.0
    alpha=stats.skew(vwapret,axis=0)
    alpha=alpha-np.nanmean(alpha)
    alpha=alpha/np.nanstd(alpha)
    alpha[~np.isfinite(alpha)]=0
    return -1.0*alpha

def dli_15pkurtosis(dr, min1i):
    di=int(min1i/conts.daymincnt)
    h4i=int(min1i/conts.h4mincnt)
    h1i=int(min1i/conts.h1mincnt)
    min15i=int(min1i/conts.min15mincnt) 
    delta=15
    money=dr["min1info_money"][min1i-delta:min1i]
    volume=dr["min1info_volume"][min1i-delta:min1i]
    vwap=money/volume
    vwapret=vwap[1:]/vwap[:-1]-1.0
    alpha=stats.kurtosis(vwapret,axis=0)
    alpha=alpha-np.nanmean(alpha)
    alpha=alpha/np.nanstd(alpha)
    alpha[~np.isfinite(alpha)]=0
    return -1.0*alpha

def dli_60pkurtosis(dr, min1i):
    di=int(min1i/conts.daymincnt)
    h4i=int(min1i/conts.h4mincnt)
    h1i=int(min1i/conts.h1mincnt)
    min15i=int(min1i/conts.min15mincnt) 
    delta=60
    money=dr["min1info_money"][min1i-delta:min1i]
    volume=dr["min1info_volume"][min1i-delta:min1i]
    vwap=money/volume
    vwapret=vwap[1:]/vwap[:-1]-1.0
    alpha=stats.kurtosis(vwapret,axis=0)
    alpha=alpha-np.nanmean(alpha)
    alpha=alpha/np.nanstd(alpha)
    alpha[~np.isfinite(alpha)]=0
    return -1.0*alpha

def dli_15pscorr(dr, min1i):
    di=int(min1i/conts.daymincnt)
    h4i=int(min1i/conts.h4mincnt)
    h1i=int(min1i/conts.h1mincnt)
    min15i=int(min1i/conts.min15mincnt) 
    delta=60
    money=dr["min1info_money"][min1i-delta:min1i]
    volume=dr["min1info_volume"][min1i-delta:min1i]
    vwap=money/volume
    vwapret=vwap[1:]/vwap[:-1]-1.0
    alpha=pearsonr(torch.from_numpy(vwapret[:-2]), torch.from_numpy(vwapret[2:]), batch_first=False).numpy()[0]
    alpha=alpha-np.nanmean(alpha)
    alpha=alpha/np.nanstd(alpha)
    alpha[~np.isfinite(alpha)]=0
    return -1.0*alpha

def dli_15prpp(dr, min1i):
    di=int(min1i/conts.daymincnt)
    h4i=int(min1i/conts.h4mincnt)
    h1i=int(min1i/conts.h1mincnt)
    min15i=int(min1i/conts.min15mincnt) 
    delta=15
    money=dr["min1info_money"][min1i-delta:min1i]
    volume=dr["min1info_volume"][min1i-delta:min1i]
    vwap=money/volume
    vwapret=vwap[1:]/vwap[:-1]-1.0
    retp=vwapret*(vwapret>0)
    alpha=(retp*retp).sum(axis=0)/((vwapret*vwapret).sum(axis=0))
    alpha=alpha-np.nanmean(alpha)
    alpha=alpha/np.nanstd(alpha)
    alpha[~np.isfinite(alpha)]=0
    return alpha

def dli_15prfp(dr, min1i):
    di=int(min1i/conts.daymincnt)
    h4i=int(min1i/conts.h4mincnt)
    h1i=int(min1i/conts.h1mincnt)
    min15i=int(min1i/conts.min15mincnt) 
    delta=15
    money=dr["min1info_money"][min1i-delta:min1i]
    volume=dr["min1info_volume"][min1i-delta:min1i]
    vwap=money/volume
    vwapret=vwap[1:]/vwap[:-1]-1.0
    retp=vwapret*(vwapret<0)
    alpha=(retp*retp).sum(axis=0)/((vwapret*vwapret).sum(axis=0))
    alpha=alpha-np.nanmean(alpha)
    alpha=alpha/np.nanstd(alpha)
    alpha[~np.isfinite(alpha)]=0
    return alpha

def dli_15mr(dr, min1i):
    di=int(min1i/conts.daymincnt)
    h4i=int(min1i/conts.h4mincnt)
    h1i=int(min1i/conts.h1mincnt)
    min15i=int(min1i/conts.min15mincnt) 
    delta=15
    money=dr["min1info_money"][min1i-delta:min1i]
    moneysum=money.sum(axis=0)
    alpha=stats.rankdata(moneysum, nan_policy="omit")
    alpha=alpha-np.nanmean(alpha)
    alpha=alpha/np.nanstd(alpha)
    alpha[~np.isfinite(alpha)]=0
    return alpha

def dli_15mrr(dr, min1i):
    di=int(min1i/conts.daymincnt)
    h4i=int(min1i/conts.h4mincnt)
    h1i=int(min1i/conts.h1mincnt)
    min15i=int(min1i/conts.min15mincnt) 
    delta=15
    money=dr["min1info_money"][min1i-delta:min1i]
    moneyrank=stats.rankdata(money, axis=0, nan_policy="omit")
    w=np.repeat(np.arange(moneyrank.shape[0]).reshape(-1,1), moneyrank.shape[1], axis=1).astype(np.float32)
    alpha=pearsonr(torch.from_numpy(moneyrank), torch.from_numpy(w), batch_first=False).numpy()[0]
    alpha=alpha-np.nanmean(alpha)
    alpha=alpha/np.nanstd(alpha)
    alpha[~np.isfinite(alpha)]=0
    return alpha

def dli_15mhighret(dr, min1i):
    di=int(min1i/conts.daymincnt)
    h4i=int(min1i/conts.h4mincnt)
    h1i=int(min1i/conts.h1mincnt)
    min15i=int(min1i/conts.min15mincnt) 
    delta=8
    high=dr["min15info_high"][min15i-delta:min15i]  
    ret=high[-1]/high[-2]-1.0
    retstd=(high[1:]/high[:-1]-1.0).std(axis=0)
    alpha=ret/retstd
    alpha=alpha-np.nanmean(alpha)
    alpha=alpha/np.nanstd(alpha)
    alpha[~np.isfinite(alpha)]=0
    return alpha








