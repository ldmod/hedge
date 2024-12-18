# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 21:15:11 2021

@author: ld
"""
"""
"""
import pandas as pd
import numpy as np
import os
import pickle
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED, ALL_COMPLETED
import time
import multiprocessing
import cryptoqt.data.constants as conts
import matplotlib.pyplot as plt
from datetime import datetime
pd.set_option('display.max_columns', None)
pd.set_option("display.max_colwidth", 100)
pd.set_option('display.width', 10000)

def stats(path):
    plist=os.listdir(path)
    plist=sorted(plist)
    tsidx=plist.index("pred_2024-01-03.csv")
    teidx=plist.index("pred_2024-05-29.csv")
    rets=[]
    xs=[]
    for idx in range(tsidx, teidx+1):
        fname=path+plist[idx]
        f=pd.read_csv(fname)
        a=f.iloc[:50]["tret"].mean()
        b=f.iloc[:20]["tret"].mean()
        c=f.iloc[:10]["tret"].mean()
        rets.append(b)
        xs.append(f["date"].iloc[0])
    xsdate = [datetime.strptime(d, '%Y-%m-%d').date() for d in xs]
    plt.plot(xsdate, np.cumsum(rets))
    plt.tick_params(axis='both',which='both',labelsize=10)
    plt.gcf().autofmt_xdate()  # 自动旋转日期标记
    plt.show()
    for i,date in enumerate(xs):
        print("ret:", xs[i], rets[i])
    rets=np.array(rets)
    return rets
    
class QueryData(object):
    def __init__(self, datadict, min1info_tm):
        self.data_dict=datadict
        self.min1info_tm=min1info_tm

    def gettibydate(self, date):
        ti=self.min1info_tm.tolist().index(date)
        return ti
    
    def getdatamin1(self, min1i):
        di=int(min1i/conts.daymincnt-1)
        close=self.data_dict["min1info_close"][min1i-1]
        preclose=self.data_dict["dayinfo_close"][di]
        ret=close/preclose-1.0
        volume=self.data_dict["min1info_volume"][min1i-1]
        money=close*volume
        date=np.repeat(self.min1info_tm[min1i+1].reshape(-1), close.shape[0])
        uid=self.data_dict["sids"]
        df=pd.DataFrame({})
        df["uid"]=uid
        df["date"]=date
        df["close"]=close
        df["ret"]=ret
        df["money"]=money
        df["volume"]=volume
        df["preclose"]=preclose
        
        
        return df

        
if __name__ == "__main__":
    path="/home/prod/newversion/smlp/model_states/infmodel/tnowtest_simple/res/"
    path2="/home/prod/newversion/seq/model_states/infmodel/tnowseq/res/"
    path3="/home/prod/newversion/addhid/model_states/infmodel/tnowtest_simple/res/"
    a=pd.read_csv(path+"pred_2024-05-09.csv")
    rets=stats(path)
    rets2=stats(path2)
    rets3=stats(path3)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
