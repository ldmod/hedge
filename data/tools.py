#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 11:04:14 2024

@author: prod
"""
import time
import datetime
import pytz
import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
import random
import pandas as pd
from audtorch.metrics.functional import pearsonr
beijing = pytz.timezone("Asia/Shanghai")
pst = pytz.timezone("US/Pacific")

class TimeProfilingLoop(object):
    def __init__(self, name):
        self.deltas=[]
        self._s=datetime.datetime.now()
        self._subt={}
        self._name=name
    def end(self):
        self._e=datetime.datetime.now()
        delta=(self._e-self._s).seconds*1000.0+(self._e-self._s).microseconds/1000.0
        self.deltas.append(delta)
    def restart(self):
        self._s=datetime.datetime.now()
    def add(self, name):
        if not name in self._subt:
            self._subt[name]=TimeProfilingLoop(name)  
        else:
            self._subt[name].restart()
        return self._subt[name]
    def to_string(self):
        # assert (self._e != None) << self._name+"timeend"
        deltastr=str(sum(self.deltas))+":"+str(sum(self.deltas)/len(self.deltas))+":"+str(len(self.deltas))
        s='{'+self._name+':'+deltastr
        for k,v in self._subt.items():
            s+=v.to_string()+'--'
        s+='}'
        return s
    
def tmu2s(tm, fmat=None):
    tm=datetime.datetime.fromtimestamp(tm/1000)
    if fmat is None:
        tm=tm.strftime("%Y%m%d%H%M%S")
    else:
        tm=tm.strftime(fmat)
    return tm

def tmu2s_utc(tm, fmat=None):
    tm=tm-8*3600*1000
    tm=datetime.datetime.fromtimestamp(tm/1000)
    if fmat is None:
        tm=tm.strftime("%Y%m%d%H%M%S")
    else:
        tm=tm.strftime(fmat)
    return tm

def tmu2i(tm):
    tm=datetime.datetime.fromtimestamp(tm/1000)
    tm=int(tm.strftime("%Y%m%d%H%M%S"))
    return tm

def tmu2ms(tm):
    tm=datetime.datetime.fromtimestamp(tm/1000.0)
    tm=tm.strftime("%Y%m%d%H%M%S.%f")
    return tm

def tmi2s(tm):
    tt=str(tm)
    tt=tt[:4]+"-"+tt[4:6]+"-"+tt[6:8]+"T"+tt[8:10]+":"+tt[10:12]+":"+tt[12:14]
    return tt

def string_toTimestamp(st):
    return time.mktime(time.strptime(st, "%Y-%m-%dT%H:%M:%S"))

def tmi2u(tm):
    tt=str(tm)
    tt=tt[:4]+"-"+tt[4:6]+"-"+tt[6:8]+"T"+tt[8:10]+":"+tt[10:12]+":"+tt[12:14]
    tt=int(string_toTimestamp(tt)*1000)
    return tt


def moving_average(x, w):
    if torch.is_tensor(x):
        x=x.cpu().detach().numpy()
    return np.convolve(x, np.ones(w), 'same') / w

def mova(x, w):
    if torch.is_tensor(x):
        x=x.cpu().detach().numpy()
    return np.convolve(x, np.ones(w), 'same') / w
def nt(a):
    return a.cpu().detach().numpy().flatten()
def pltg(a, b=20,range=None,clip=False):
    if clip:
        range=(np.percentile(a,1),np.percentile(a,99))
    if torch.is_tensor(a):
        return plt.hist(a.flatten().cpu().detach().numpy(),bins=b,range=range)
    else:
        return plt.hist(a.flatten(),bins=b,range=range)
def plotg(a, clip=False):
    if clip:
        range=(np.percentile(a,1),np.percentile(a,99))
    if torch.is_tensor(a):
        return plt.plot(a.flatten().cpu().detach().numpy())
    else:
        return plt.plot(a.flatten())
def isfinite(a):
    if torch.is_tensor(a):
        return a[torch.isfinite(a)]
    else:
        return a[np.isfinite(a)]

def tensornorm(x, eps=1e-8, dim=0):
    mean=x.nanmean(dim=dim).unsqueeze(dim=dim)
    std=(((x-mean)**2).nanmean(dim=dim)).unsqueeze(dim=dim)**0.5
    x=(x-mean)/std.clamp(eps)
    return x

def npnorm(x, eps=1e-8):
    mean=np.nanmean(x)
    std=np.nanstd(x)
    x=(x-mean)/max(std, eps)
    return x

def readh5(path, sleeptime=5):
    cnt=0
    while cnt<10000:
        cnt+=1
        try:
            f=h5py.File(path, "r", libver='latest', swmr=True)
            break
        except:
            print("read wati:", cnt, path, flush=True)
            time.sleep(sleeptime)
    return f

def writeh5(path, sleeptime=5):
    cnt=0
    while cnt<10000:
        cnt+=1
        try:
            f=h5py.File(path, "a", libver='latest')
            break
        except:
            print("write wati:", cnt, path, flush=True)
            time.sleep(sleeptime)
    return f

def open_csv(path, mode="r"):
    while True:
        try:
            store = pd.HDFStore(path, mode)
            break
        except Exception as e:
            print("open csv fail:", path, str(e), flush=True)
            time.sleep(1)
    return store


def open_csv_copy(path, mode="r"):
    while True:
        try:
            store = pd.HDFStore(path, mode)
            break
        except Exception as e:
            print("open csv fail:", path, str(e), flush=True)
            time.sleep(1)
    copy_data={}
    for key in store.keys():
        copy_data[key[1:]]=store[key]
    store.close()
    return copy_data

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True



