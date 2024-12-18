#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 20:52:18 2024

@author: prod
"""
import cryptoqt.data.constants as conts
import numpy as np
from cryptoqt.prm.alpha.yamlcfg import gv
import scipy.stats as stats
import cryptoqt.prm.alpha.tools as tools
import cryptoqt.data.sec_klines.sec_klines as sk
import cryptoqt.alpha.secfeaextractor as secfea
import torch

class InsDict:
    def __init__(self):
        self.dd={}
    def adddata(self, keys, curlen):
        for key in keys:
            assert not key in self.dd.keys()
            self.dd[key]=tools.CircleTensor(maxlen=gv["insmaxlen"], insstart=curlen)
        
def updatesecdr(secdr, s1i):
    fields=[field  for field in sk.kline_fields if field != "valid" ]
    
    if not "insdict" in gv:
        gv["insdict"]=InsDict()
        insdict=gv["insdict"]
        insdict.adddata(fields, s1i-gv["insmaxlen"])
        for key in secfea.g_feafunc.keys():
            insdict.adddata([key], s1i-gv["insmaxlen"])
            insdict.dd[key].append(torch.zeros((int(gv["insmaxlen"]/16), len(gv["sids"]))).cuda(), s1i-gv["insmaxlen"])


    insdict=gv["insdict"]
    for key in fields:
        curlen=insdict.dd[key].e
        data=torch.from_numpy(secdr[key][curlen:s1i].copy()).cuda()
        insdict.dd[key].append(data, curlen)
    # #     
    for key in secfea.g_feafunc.keys():
        curlen=insdict.dd[key].e
        data=secfea.g_feafunc[key](insdict.dd, curlen, s1i)
        insdict.dd[key].append(data, curlen)
    return

def appendfea(feas, data):
    data[~data.isfinite()]=0
    data=data.unsqueeze(dim=2)
    feas.append(data)
def gets1seq(dd, s1i, seqlen):
    feas=[]
    pavgcnt=gv["pavgcnt"]
    curcloseprice=dd["s1info_vwap"][s1i-pavgcnt:s1i].mean(axis=0)
    curclosepricestd=dd["s1info_vwap"][s1i-pavgcnt:s1i].std(axis=0)
    curvolumeavg=dd["s1info_volume"][s1i-pavgcnt:s1i].mean(axis=0)
    curvolumeavgstd=dd["s1info_volume"][s1i-pavgcnt:s1i].std(axis=0)
    curmoneyavg=dd["s1info_money"][s1i-pavgcnt:s1i].mean(axis=0)
    curmoneyavgstd=dd["s1info_money"][s1i-pavgcnt:s1i].std(axis=0)
    
    pricefields=[key for key in dd.keys() if key.endswith('wap') or key.endswith('high') or key.endswith('low')]
    volumefields=[key for key in dd.keys() if key.endswith('volume')]
    moneyfields=[key for key in dd.keys() if key.endswith('money')]
    otherfield=[key for key in dd.keys() if key not in pricefields+volumefields+moneyfields]
    
    for field in pricefields:
        data=(dd[field][s1i-seqlen:s1i]-curcloseprice)/curclosepricestd
        appendfea(feas, data)
    for idx,field in enumerate(volumefields):
        data=(dd[field][s1i-seqlen:s1i]-curvolumeavg)/curvolumeavgstd
        appendfea(feas, data)
    for idx,field in enumerate(moneyfields):
        data=(dd[field][s1i-seqlen:s1i]-curmoneyavg)/curmoneyavgstd
        appendfea(feas, data)
    for idx,field in enumerate(otherfield):
        data=dd[field][s1i-seqlen:s1i]
        #section norm
        fea_secnorm=tools.tensornorm(data, dim=1)
        appendfea(feas, fea_secnorm)
        #time norm
        fea_tmnorm=tools.tensornorm(data, dim=0)
        appendfea(feas, fea_tmnorm)

    feas=torch.cat(feas, dim=2)

    return feas

def getvalid(dd, s1i):
    validmins=5
    valid=((dd["s1info_vwap"][s1i-conts.minseccnt*validmins:s1i]>0).sum(axis=0)==validmins*conts.minseccnt)
    vwapflag=(dd["s1info_vwap"][s1i-conts.minseccnt*validmins:s1i].isfinite().sum(axis=0)==validmins*conts.minseccnt)
    valid=valid*vwapflag
    moneyflag=dd["s1info_money"][s1i-conts.minseccnt*validmins:s1i].sum(axis=0)>(10000000/1440)
    valid=valid&moneyflag
    return  valid

def getrx(dd, x, s1i):
    pavgcnt=gv["pavgcnt"]
    # oldvwap=(dd["s1info_money"][s1i-pavgcnt:s1i].sum(dim=0)/dd["s1info_volume"][s1i-pavgcnt:s1i].sum(dim=0)).cpu()
    oldvwap=dd["s1info_vwap"][s1i-pavgcnt:s1i].mean(axis=0).cpu().double()
    rx=x.clone().double()
    rx[:,0]=oldvwap*(1+x[:,0]/10000.0)
    rx[:,1]=oldvwap*(1+x[:,1]/10000.0)
    rx[:,2]=oldvwap*(1+x[:,2]/10000.0)
    rx[:,3]=oldvwap*(1+x[:,3]/10000.0)
    return rx
    
def getyvwap(dd, s1i, tmperiod, longterm):
    pavgcnt=gv["pavgcnt"]
    # oldvwap=dd["s1info_money"][s1i-pavgcnt:s1i].sum(dim=0)/dd["s1info_volume"][s1i-pavgcnt:s1i].sum(dim=0)
    oldvwap=dd["s1info_vwap"][s1i-pavgcnt:s1i].mean(axis=0)
    money=dd["s1info_money"][s1i+gv["delaysec"]:s1i+tmperiod+gv["delaysec"]].sum(dim=0)
    volume=dd["s1info_volume"][s1i+gv["delaysec"]:s1i+tmperiod+gv["delaysec"]].sum(dim=0)
    vwap=money/volume
    #
    lvwap=dd["s1info_money"][s1i+gv["delaysec"]:s1i+longterm+gv["delaysec"]].sum(dim=0)/\
        dd["s1info_volume"][s1i+gv["delaysec"]:s1i+longterm+gv["delaysec"]].sum(dim=0)
    yl=(lvwap/vwap-1.0)*10000.0
    
    yvwap=(vwap/oldvwap-1.0)*10000.0
    vf_y=(yvwap.isfinite())
    return  yvwap, yl, vf_y

def getys(dd, s1i, tmperiod, longterm):
    pavgcnt=gv["pavgcnt"]
    # oldvwap=dd["s1info_money"][s1i-pavgcnt:s1i].sum(dim=0)/dd["s1info_volume"][s1i-pavgcnt:s1i].sum(dim=0)
    oldvwap=dd["s1info_vwap"][s1i-pavgcnt:s1i].mean(axis=0)
    money=dd["s1info_smoney"][s1i+gv["delaysec"]:s1i+tmperiod+gv["delaysec"]].sum(dim=0)
    volume=dd["s1info_svolume"][s1i+gv["delaysec"]:s1i+tmperiod+gv["delaysec"]].sum(dim=0)
    svwap=money/volume
    #
    lsvwap=dd["s1info_smoney"][s1i+gv["delaysec"]:s1i+longterm+gv["delaysec"]].sum(dim=0)/\
        dd["s1info_svolume"][s1i+gv["delaysec"]:s1i+longterm+gv["delaysec"]].sum(dim=0)
    ysl=(lsvwap/svwap-1.0)*10000.0
    #
    slow=dd["s1info_slow"][s1i+gv["delaysec"]:s1i+tmperiod+gv["delaysec"]].min(dim=0)[0]
    ysvwap=(svwap/oldvwap-1.0)*10000.0
    yslow=(slow/oldvwap-1.0)*10000.0
    vf_ys=(ysvwap.isfinite() & yslow.isfinite())
    return  ysvwap, yslow, ysl, vf_ys

def getyb(dd, s1i, tmperiod, longterm):
    pavgcnt=gv["pavgcnt"]
    # oldvwap=dd["s1info_money"][s1i-pavgcnt:s1i].sum(dim=0)/dd["s1info_volume"][s1i-pavgcnt:s1i].sum(dim=0)
    oldvwap=dd["s1info_vwap"][s1i-pavgcnt:s1i].mean(axis=0)
    money=dd["s1info_bmoney"][s1i+gv["delaysec"]:s1i+tmperiod+gv["delaysec"]].sum(dim=0)
    volume=dd["s1info_bvolume"][s1i+gv["delaysec"]:s1i+tmperiod+gv["delaysec"]].sum(dim=0)
    bvwap=money/volume
    #
    lbvwap=dd["s1info_bmoney"][s1i+gv["delaysec"]:s1i+longterm+gv["delaysec"]].sum(dim=0)/\
        dd["s1info_bvolume"][s1i+gv["delaysec"]:s1i+longterm+gv["delaysec"]].sum(dim=0)
    ybl=(lbvwap/bvwap-1.0)*10000.0
    
    bhigh=dd["s1info_bhigh"][s1i+gv["delaysec"]:s1i+tmperiod+gv["delaysec"]].max(dim=0)[0]
    ybvwap=(bvwap/oldvwap-1.0)*10000.0
    ybhigh=(bhigh/oldvwap-1.0)*10000.0
    vf_yb=(ybvwap.isfinite() & ybhigh.isfinite())
    return  ybvwap, ybhigh, ybl, vf_yb

    










