#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 18:00:38 2024

@author: nb
"""
import sys
import os
sys.path.append(os.path.abspath(__file__+"../../../../"))
import cryptoqt.data.tools as tools
import h5py
import numpy as np
import pandas as pd
class H5Swmr(object):
    def __init__(self, path, columns, mode="r"):
        self.path = path
        self.columns = columns
        self.mode=mode
        if os.path.exists(self.path):
            cmd=f'h5clear -s {self.path}'
            os.system(cmd)
            self.mode="a" if self.mode=='w' else self.mode
            try:
                self.h5f=h5py.File(self.path, self.mode, swmr=True, libver='latest')
            except Exception as ex:
                print(f"except:{self.path} open h5fs Exception",  
                      flush=True)
                raise
            if self.mode =='a':
                self.h5f.swmr_mode =True
        else:
            self.h5f=h5py.File(self.path, self.mode, libver='latest')
            for key, vtype in self.columns:
                dtype = tools.s2type(vtype)
                self.h5f.create_dataset(key, shape=(0, ), 
                                          maxshape = (None, ), chunks = (1024000,), dtype=dtype)
            self.h5f.swmr_mode =True

    def append_dict(self, dd):
        old_len = self.h5f[list(self.h5f.keys())[0]].shape[0] 
        new_len = old_len + 1
        for key in dd.keys():
            value=dd[key]
            assert self.h5f[key].shape[0] == old_len
            self.h5f[key].resize(new_len, axis=0)
            self.h5f[key][old_len:new_len] = value
            
        self.h5f.flush()
        
    def append(self, df_append):
        mk=0
        for key, vtype in self.columns:
            if key in df_append.columns and tools.s2type(vtype)==df_append[key].dtype:
                mk+=1

        if mk==len(self.columns):
            old_len = self.h5f[list(self.h5f.keys())[0]].shape[0] 
            new_len = old_len + df_append.shape[0]
            for key in df_append.columns:
                value=df_append[key].values
                self.h5f[key].resize(new_len, axis=0)
                self.h5f[key][old_len:new_len] = value
        else:
            print("append error:", df_append, self.columns, flush=True)
            
        self.h5f.flush()
        
    def lasted_items(self, cnt, tmField="opentm"):
        dd={}
        dataLen = self.h5f[tmField].shape[0]
        for key in self.h5f.keys():
            value=self.h5f[key][dataLen-cnt:dataLen]
            dd[key]=value
        dd=pd.DataFrame(dd)
        dd["tmi"]=dd[tmField].apply(lambda x:tools.tmu2i(x))
        return dd

    def front_items(self, cnt, tmField="opentm"):
        dd={}
        for key in self.h5f.keys():
            value=self.h5f[key][:cnt]
            dd[key]=value
        dd=pd.DataFrame(dd)
        dd["tmi"]=dd[tmField].apply(lambda x:tools.tmu2i(x))
        return dd
    
    def get_items(self, startIdx, endIdx, tmField="opentm"):
        dd={}
        for key in self.h5f.keys():
            value=self.h5f[key][startIdx:endIdx]
            dd[key]=value
        dd=pd.DataFrame(dd)
        dd["tmi"]=dd[tmField].apply(lambda x:tools.tmu2i(x))
        return dd
            
    def refresh(self):
        for key in self.h5f.keys():
            self.h5f[key].refresh()
            
    def flush(self):
        self.h5f.flush()
             
    def close(self):
        self.h5f.close()
        
if __name__ == "__main__":
    cfg={"path":"/dev/shm/cryptoqt/test.h5", "columns":[["price", "np.float32"], ["tm", "np.int32"]]}
    h5f=H5Swmr(cfg["path"], cfg["columns"], mode="a")
    lasted_items = h5f.lasted_items(50)
    print(lasted_items)
    df = pd.DataFrame({'tm': np.arange(3000).astype(np.int32), 'price': np.random.rand(3000).astype(np.float32)})
    h5f.append(df)
    h5f.close()
    
    
    
    
    