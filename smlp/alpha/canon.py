import os
import sys
import h5py
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import cryptoqt.smlp.alpha.catransinit as cinit
import cryptoqt.smlp.alpha.tools as tools
import cryptoqt.smlp.alpha.seqfea as seqfea
import cryptoqt.smlp.alpha.common as common
import cryptoqt.smlp.alpha.mergemodel as mmodel
import cryptoqt.smlp.alpha.yamlcfg as yamlcfg
from cryptoqt.smlp.alpha.yamlcfg import gv
import random
import datetime
import gc
import torch
# from memory_profiler import profile
# from alpha.gpu_mem_track import MemTracker
import inspect
import time 
import torch.distributed as dist
import cryptoqt.data.querydata as qd
import matplotlib.pyplot as plt
import cryptoqt.data.constants as conts
import h5py
from  cryptoqt.smlp.alpha.tools import flog as flog

def istype(ss, ttype):
    try:
        res = ttype(ss)
        return True
    except ValueError:
        return False
 
def tmdata(min1i):
    di=int(min1i/conts.daymincnt)
    h1i=int(min1i/conts.h1mincnt)
    h4i=int(min1i/conts.h4mincnt)
    min15i=int(min1i/conts.min15mincnt)
    min5i=int(min1i/conts.min5mincnt)
    dd={}
    for key in gv["data_dict"].keys():
        if  not key in ['alldates', 'uidsmap', 
                    'min1info_tm']:
            if key.startswith("dayinfo"):
                dd[key]=gv["data_dict"][key][:di]
            elif key.startswith("h4info"):
                    dd[key]=gv["data_dict"][key][:h4i]
            elif key.startswith("h1info"):
                dd[key]=gv["data_dict"][key][:h1i]
            elif key.startswith("min15info"):
                dd[key]=gv["data_dict"][key][:min15i]
            elif key.startswith("min1info"):
                dd[key]=gv["data_dict"][key][:min1i]
            elif key.startswith("min5info"):
                dd[key]=gv["data_dict"][key][:min5i]
            elif key.startswith("smin1info"):
                dd[key]=gv["data_dict"][key][:min1i]
            elif key.startswith("smin5info"):
                dd[key]=gv["data_dict"][key][:min5i]
    dd["sids"]=gv["data_dict"]["sids"]
    gv["savedr"]=dd
    return

class Conon1445():
    def __init__(self, cfg):
        self.dr=cfg["dr"]
        gv["data_dict"]=self.dr
        gv["ts"]=self.dr["min1info_tm"]
        gv["lasteddownload"]=cfg.get("lasteddownload")
        gv["min1info_tm"]=self.dr["min1info_tm"]
        gv["baseuids"]=self.dr["sids"]
        tpf=tools.TimeProfilingLoop("init")
        self.yamlpath=cfg.get("yamlcfgpath", "")
        cinit.initcfg(self.yamlpath)
        cinit.pconfig()
        self.name=gv["name"]
        #######modify config ############

        #######parse startdate##########
        self.startdti=gv["startdti"]
        self.lastdi=self.startdti*gv["tmdelta"]
        #######parse startdate##########
        lastedmodelpath=None
        if os.path.exists(gv["modelpath"]):
            ms=os.listdir(gv["modelpath"])
            modelpaths=[]
            for path in ms:
                if "_date_" in path:
                    modelpaths.append((path, path.split("_date_")[1]))
            if len(modelpaths) > 0:
                lastedmodelpath=gv["modelpath"]+sorted(modelpaths, key=lambda x:x[1])[-1][0]     
        
        self.pred=cinit.initmodel(self.dr, lastedmodelpath)
        if not lastedmodelpath is None:
            prefix, modeldate = self.modelpath_parser(lastedmodelpath)
            if modeldate > int(gv["min1info_tm"][-1]) :
                self.lastdi=len(gv["min1info_tm"])-1
                flog("modelday large than basedates")
            else:
                self.lastdi=gv["min1info_tm"].tolist().index(modeldate)

        tpf.end()
        flog("init tm:", tpf.to_string())
        return
            
    def modelpath_parser(self, path):
        prefix=path.split('_date_')[0]
        modeldate=int(path.split('_date_')[1])
        return prefix, modeldate
        
        
    def inffer(self, min1i):

        x, vf, hidden=self.pred.pred(min1i)  #pred 
        ########### model pred ##############
        flog("pred end:", gv["data_dict"]["min1info_tm"][min1i])
        
        return x,vf, hidden
    
    def trainnd(self, min1i):
        gv["dynamiclr"]=1.0 
        self.pred.tune(min1i) # train lastday 1445
            
        ########### model train ##############
        flog("tune end:", min1i, gv["data_dict"]["min1info_tm"][min1i])
        return
        
    def savecp(self, min1i):
        eidx=min1i
        di=int(min1i/conts.daymincnt)
            
        if gv["realtime"] or (min1i % gv["savedelta"] == 0) :
            self.pred.checkpoint(gv["model_path"]+"_date_"+str(gv["ts"][min1i]))
        
        flog("after_inffer:", min1i, gv["data_dict"]["min1info_tm"][min1i])
        return

    def onlinepred(self):
        return
        
    def generate(self, ts, te):
        curti=self.lastdi
        for ti in range(ts, te+1, gv["tmdelta"]):
            if ti <= curti:
                continue
            tfp=tools.TimeProfilingLoop("pred")
            min1i=ti
            tmdata(min1i)
            
            
            self.trainnd(min1i)
            x, vf, hidden = self.inffer(min1i)
            
            if "savepred"in gv and gv["savepred"]:
                self.alpha=x.cpu().numpy().copy()
                valid=vf.cpu().numpy().copy()
                
                self.alpha[~valid] = np.nan
                self.alpha[~np.isfinite(self.alpha)] = np.nan
                # self.alpha[:] = self.alpha[:] - np.nanmean(self.alpha[:])
                df=pd.DataFrame({})
                df["sid"]=gv["data_dict"]["sids"][:]
                df["pred0"]=self.alpha[:,0]
                df["pred1"]=self.alpha[:,1]
                df["pred2"]=self.alpha[:,2]
                df.to_csv(gv["modelpath"]+"./res/"+str(gv["min1info_tm"][ti])+"_pred.csv")
            
                if int(ti/gv["tmdelta"]) %100==0:
                    gc.collect()
                    time.sleep(0.1)
                    torch.cuda.empty_cache()
            tfp.end()
            flog("tm cose:", min1i, gv["data_dict"]["min1info_tm"][min1i], "tmcose:",tfp.to_string())
            self.savecp(min1i)
            self.lastdi=min1i
        
        return

    def getptopred(self, alpha, vf, min1i):
        x=alpha.flatten().copy()
        x[~vf]=0
        x[~np.isfinite(x)]=0
        argx=np.argsort(x)
        argx=argx[::-1]
        queryd=gv["queryd"]
        df=queryd.getdatamin1(min1i)
        df=df.iloc[argx]
        df["pred_score"]=x[argx]
        return df
        
        

def create(cfg):
    return Conon1445(cfg)

