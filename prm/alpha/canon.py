import os
import sys
import h5py
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import cryptoqt.prm.alpha.catransinit as cinit
import cryptoqt.prm.alpha.tools as tools
import cryptoqt.prm.alpha.seqfea as seqfea
import cryptoqt.prm.alpha.common as common
import cryptoqt.prm.alpha.mergemodel as mmodel
import cryptoqt.prm.alpha.yamlcfg as yamlcfg
from cryptoqt.prm.alpha.yamlcfg import gv
import random
import datetime
import gc
import torch
import inspect
import time 
import torch.distributed as dist
import cryptoqt.data.querydata as qd
import matplotlib.pyplot as plt
import cryptoqt.data.constants as conts
import h5py
from  cryptoqt.prm.alpha.tools import flog as flog
import cryptoqt.data.sec_klines.sec_klines as sk
import cryptoqt.prm.alpha.data2ins as d2i

class Conon1445():
    def __init__(self, cfg):
        self.dr=cfg["dr"]
        gv["data_dict"]=self.dr
        gv["secdr"]=cfg["secdr"]
        gv["sids"]=self.dr["sids"]
        tpf=tools.TimeProfilingLoop("init")
        self.yamlpath=cfg.get("yamlcfgpath", "")
        cinit.initcfg(self.yamlpath)
        cinit.pconfig()
        self.name=gv["name"]
        #######modify config ############

        #######parse startdate##########
        self.starttm=gv["starttm"]
        self.lastsi=sk.gtmidx_i(self.starttm)
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
            self.lastsi=sk.gtmidx_i(modeldate)
        self.openresh5(sk.gtm_i(self.lastsi))
        tpf.end()
        flog("init tm:", tpf.to_string())
        return
            
    def openresh5(self, lasttm):
        respath=gv["modelpath"]+"/res/res.h5"
        if os.path.exists(respath):
            cmd=f'h5clear -s {respath}'
            os.system(cmd)
            self.resh5=h5py.File(respath, 'a', swmr=True, libver='latest')
            self.resh5.swmr_mode =True
        else:
            self.resh5=h5py.File(respath, 'w', libver='latest')
            self.resh5.create_dataset("tm", shape=(0,), 
                                      maxshape = (None,), chunks = True, dtype=np.int64)
            for key in gv["outvalue"]:
                self.resh5.create_dataset(key, shape=(0,gv["sids"].shape[0]), 
                                          maxshape = (None,gv["sids"].shape[0]), chunks = True, dtype=np.float64)
            self.resh5.swmr_mode =True
        
        rzidx=np.where(self.resh5["tm"][:]>lasttm)[0]
        if rzidx.shape[0]>0:
            rzidx=rzidx[0]
            key="tm"
            self.resh5[key].resize(rzidx, axis=0)
            self.resh5[key].flush()
            for key in gv["outvalue"]:
                self.resh5[key].resize(rzidx, axis=0)
                self.resh5[key].flush()
        self.resh5.flush()
        return
            
        
    def modelpath_parser(self, path):
        prefix=path.split('_date_')[0]
        modeldate=int(path.split('_date_')[1])
        return prefix, modeldate
        
        
    def inffer(self, s1i):

        x, vf, xetr=self.pred.pred(s1i)  #pred 
        ########### model pred ##############
        flog("pred end:", sk.gtm_i(s1i), datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
        
        return x,vf, xetr
    
    def trainnd(self, s1i):
        gv["dynamiclr"]=1.0 
        self.pred.tune(s1i) # train lastday 1445
            
        ########### model train ##############
        flog("tune end:", s1i, sk.gtm_i(s1i), datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
        return
        
    def savecp(self, s1i):
        eidx=s1i
            
        if gv["realtime"] or (s1i % gv["savedelta"] == 0) :
            self.pred.checkpoint(gv["model_path"]+"_date_"+str(sk.gtm_i(s1i)))
        
        flog("after_inffer:", s1i, sk.gtm_i(s1i))
        return

    def onlinepred(self):
        return
        
    def generate(self, ts, te):
        curti=self.lastsi
        for ti in range(ts, te+1, gv["tmdelta"]):
            if ti <= curti:
                continue
            tfp=tools.TimeProfilingLoop("all")
            s1i=ti
            updatetfp=tfp.add("update")
            d2i.updatesecdr(gv["secdr"], s1i)
            updatetfp.end()
            # continue
            
            inffertfp=tfp.add("inffer")
            x, vf, xetr = self.inffer(s1i)
            inffertfp.end()
            
            savetfp=tfp.add("save")
            if "savepred"in gv and gv["savepred"]:
                self.alpha=x.cpu().numpy().copy()
                valid=vf.cpu().numpy().copy()
                self.alpha[~valid]=np.nan
                tmlen=self.resh5["tm"].shape[0]
                for idx in range(len(gv["outvalue"])):
                    key = gv["outvalue"][idx]
                    assert tmlen==self.resh5[key].shape[0]
                    h5append(self.resh5[key], self.alpha[:,idx])
                h5append(self.resh5["tm"], sk.gtm_i(s1i))
                self.resh5.flush()
            savetfp.end()
            
            traintfp=tfp.add("train")
            self.trainnd(s1i)
            traintfp.end()
                    
            tfp.end()
            flog("tm cose:", s1i, sk.gtm_i(s1i), datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),  "tmcose:",tfp.to_string())
            self.savecp(s1i)
            self.lastsi=s1i
        
        return
def h5append(fieldv, value):
    oldsize=fieldv.shape[0]
    fieldv.resize(oldsize+1, axis=0)
    fieldv[oldsize:oldsize+1]=value
    fieldv.flush()
        

def create(cfg):
    return Conon1445(cfg)

