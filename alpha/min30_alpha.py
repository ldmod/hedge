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
import pandas as pd
daymincnt=conts.daymincnt
h4mincnt=conts.h4mincnt
h1mincnt=conts.h1mincnt
min15mincnt=conts.min15mincnt


def readcsv(dr, min1i, si=0, path="/home/crypto/smlpv2/cryptoqt/smlp/model_states/infmodel/tsmlpv230/res"):
    di=int(min1i/conts.daymincnt)
    h4i=int(min1i/conts.h4mincnt)
    h1i=int(min1i/conts.h1mincnt)
    min15i=int(min1i/conts.min15mincnt) 
    fname=path+"/"+str(dr["min1info_tm"][min1i])+"_pred.csv"
    df=pd.read_csv(fname)
    alpha=df["pred"+str(si)].to_numpy()
    return alpha

def readcsvavg(dr, min1i, si=0, path="/home/crypto/smlpv2/cryptoqt/smlp/model_states/infmodel/tsmlpv230/res"):
    di=int(min1i/conts.daymincnt)
    h4i=int(min1i/conts.h4mincnt)
    h1i=int(min1i/conts.h1mincnt)
    min15i=int(min1i/conts.min15mincnt) 
    fname=path+"/"+str(dr["min1info_tm"][min1i])+"_pred.csv"
    df=pd.read_csv(fname)
    alpha=df["pred0"].to_numpy()+df["pred1"].to_numpy()+df["pred2"].to_numpy()
    return alpha

def readcsvselect(dr, min1i, si=0, path="/home/crypto/smlpv2/cryptoqt/smlp/model_states/infmodel/tsmlpv230/res"):
    di=int(min1i/conts.daymincnt)
    h4i=int(min1i/conts.h4mincnt)
    h1i=int(min1i/conts.h1mincnt)
    min15i=int(min1i/conts.min15mincnt) 
    fname=path+"/"+str(dr["min1info_tm"][min1i])+"_pred.csv"
    df=pd.read_csv(fname)
    alpha=df["pred"+str(si)].to_numpy()
    # alpha=df["pred0"].to_numpy()+df["pred1"].to_numpy()
    return alpha

def dli_min30dr5(dr, min1i):
    di=int(min1i/conts.daymincnt)
    h4i=int(min1i/conts.h4mincnt)
    h1i=int(min1i/conts.h1mincnt)
    min15i=int(min1i/conts.min15mincnt) 
    min30i=int(min1i/conts.min15mincnt/2) 
    delay=5
    dclose=dr["min1info_close"]
    ret5day=dclose[min1i-1]/dclose[min1i-delay*30-1]-1.0
    alpha=ret5day-np.nanmean(ret5day)
    alpha=alpha/np.nanstd(alpha)
    return -1.0*alpha

def dli_1440mincorr(dr, min1i):
    di=int(min1i/conts.daymincnt)
    h4i=int(min1i/conts.h4mincnt)
    h1i=int(min1i/conts.h1mincnt)
    min15i=int(min1i/conts.min15mincnt) 
    delta=1440
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

def dli_30mapb(dr, min1i):
    di=int(min1i/conts.daymincnt)
    h4i=int(min1i/conts.h4mincnt)
    h1i=int(min1i/conts.h1mincnt)
    min15i=int(min1i/conts.min15mincnt) 
    delta=30
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

def dli_30marpp(dr, min1i):
    di=int(min1i/conts.daymincnt)
    h4i=int(min1i/conts.h4mincnt)
    h1i=int(min1i/conts.h1mincnt)
    min15i=int(min1i/conts.min15mincnt) 
    delta=30
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

def dli_30mtrend(dr, min1i):
    di=int(min1i/conts.daymincnt)
    h4i=int(min1i/conts.h4mincnt)
    h1i=int(min1i/conts.h1mincnt)
    min15i=int(min1i/conts.min15mincnt) 
    delta=30
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



















