#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 14:41:29 2023

@author: dli
"""
import os
import sys
import numpy as np
import torch
import cryptoqt.data.updatedata as ud
from functools import partial
import time
import os
import socks
import socket
import cryptoqt.data.tools as tools
import threading
from queue import Queue
import json
import logging
import mmap

g_endmap={}
g_path="/dev/shm/hedge/cryptocache"
g_endsize=int(4*365*1440)
g_infos=["min1info", "smin1info", "min1info_extra", "smin1info_extra", "smin5info", "min5info"]
def create_memmap(dr):
    path=g_path
    os.makedirs(path, exist_ok=True)
    fs=g_infos
    shapeinfos={}
    for fname in fs:
        size=int(g_endsize/5) if fname in ["smin5info", "min5info"] else g_endsize
        fp=ud.g_h5path+"/"+fname+".h5"
        f=tools.readh5(fp)
        for idx,key in enumerate(f.keys()):
            dr[key]=np.memmap(path+'/'+key+'.npy', dtype=f[key].dtype, mode='write', shape=(size,)+f[key].shape[1:])
            if f[key].dtype==np.float32:
                dr[key].fill(np.nan)
            dr[key][:f[key].shape[0]]=f[key][:][:f[key].shape[0]]
            g_endmap[key]=f[key].shape[0]
            # infos.append((key, f[key].dtype, (size,)+f[key].shape[1:]))
            shapeinfos[key]=(f[key].dtype, f[key].shape)
        f.close()
    writeshapeinfos(shapeinfos)
        
def update_memmap(dr):
    path=g_path
    fs=g_infos
    shapeinfos={}
    for fname in fs:
        fp=ud.g_h5path+"/"+fname+".h5"
        h5=tools.readh5(fp)
        for idx,key in enumerate(h5.keys()):
            s=g_endmap[key]
            dr[key][s:h5[key].shape[0]]=h5[key][s:h5[key].shape[0]]
            g_endmap[key]=h5[key].shape[0]
            shapeinfos[key]=(h5[key].dtype, h5[key].shape)
        h5.close()
    writeshapeinfos(shapeinfos)
 
def writeshapeinfos(shapeinfos):
    f=tools.writeh5(g_path+"/tmsize.h5")
    np.save(g_path+"_shapeinfos", np.array(shapeinfos, dtype=object))
    f.close()
    
def readshapeinfos():
    f=tools.readh5(g_path+"/tmsize.h5")
    shapeinfos=np.load(g_path+"_shapeinfos.npy", allow_pickle=True).item()
    f.close()
    return shapeinfos
    
def gettmsize():
    shapeinfos=readshapeinfos()
    tmsize=shapeinfos["min1info_close"][1][0]
    return tmsize
  
def load_memmap(dr):
    path=g_path
    if os.path.exists(g_path+"_shapeinfos.npy"):
        keyshapes=readshapeinfos()
        for key in keyshapes.keys():
            dtype,shape=keyshapes[key]
            size=int(g_endsize/5) if key[:8] in ["smin5inf", "min5info"] else g_endsize
            dataitem=np.memmap(path+'/'+key+'.npy', dtype=dtype, mode='r+', shape=(size,)+shape[1:])
            dr[key]=dataitem
            g_endmap[key]=shape[0]
        tmsize=gettmsize()
    else:
        create_memmap(dr)
    return 
      
def read_memmap(dr):
    path=g_path
    keyshapes=readshapeinfos()
    for key in keyshapes.keys():
        dtype,shape=keyshapes[key]
        size=int(g_endsize/5) if key[:8] in ["smin5inf", "min5info"] else g_endsize
        dataitem=np.memmap(path+'/'+key+'.npy', dtype=dtype, mode='r', shape=(size,)+shape[1:])
        dr[key]=dataitem
    tmsize=gettmsize()
    return tmsize





            