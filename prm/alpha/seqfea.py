# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 21:15:11 2021

@author: ld


"""
# Import libraries
import time
import datetime
import torch
import torch.nn as nn # ??-??o?￥??????????
from torch.optim import SGD # ????????¨SGD
import torch.utils.data as Data # ??°???é￠??¤????
import math
import pandas as pd
import numpy as np
import os
from functools import partial
import sys
import copy
from threading import Thread, Lock
import random
import imp
from scipy import spatial
torch.set_printoptions(precision=6,threshold=1000000)
from typing import List, Optional
from torch import Tensor
import pickle as pkl
from operator import itemgetter
from audtorch.metrics.functional import pearsonr
from scipy import stats
import numba
import matplotlib.pyplot as plt
import einops
from scipy.stats import rankdata
import io
import torch.nn.functional as F
from pathlib import Path
from torch.cuda.amp import autocast as autocast
import torch.nn.utils.weight_norm as weight_norm
from functools import reduce
from torch.autograd import Function
from torch.autograd import gradcheck
import cryptoqt.prm.alpha.tools  as tools
from cryptoqt.prm.alpha.tools import *
from cryptoqt.prm.alpha.common import FeatureModel, stockgradop, stocktimenormop, logop, SeqCorr, AvgPools
import cryptoqt.prm.alpha.data_parallel as dp
from cryptoqt.prm.alpha.yamlcfg import gv
import pandas as pd
from functools import reduce
import threading
from scipy.stats import rankdata
import inspect
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED, ALL_COMPLETED
import alpha.data2ins as d2i

class SeqExtractor(FeatureModel):
    def __init__(self, orifeanum, inputsize, crosssize, xstride, hiddenstride, corrsize=32, poolnames=["avg","std","maxv","minv","s","e"], 
                 section=False, mt=0,
                 corr=False,
                 deviceid=0):
        super(SeqExtractor, self).__init__()
        self.orifeanum=orifeanum
        self.hiddenstride=hiddenstride
        self.xstride=xstride
        self.poolnames=poolnames
        self.corr=corr
        self.do=nn.Dropout(gv["dropout"])
        self.inputxlen=orifeanum*len(poolnames)+inputsize*hiddenstride+gv["posembedsize"]
        if corr:
            self.inputxlen+=corrsize
            self.seqcorr=SeqCorr(orifeanum, corrsize)
        if section:
            self.csm1=SectionCss(self.inputxlen, crosssize, bnlen=0, dropout=gv["dropout"])
        else:
            self.csm1=CssModel(self.inputxlen, crosssize, bnlen=0, dropout=gv["dropout"])
        if gv["syncbn"]:
            self.bnlayer=nn.SyncBatchNorm(self.inputxlen, track_running_stats=False)
            poolsbnsize=orifeanum*len(poolnames)+corrsize*corr
            self.poolsbn=nn.SyncBatchNorm(poolsbnsize, track_running_stats=False)
        else:
            self.bnlayer=nn.BatchNorm1d(self.inputxlen, track_running_stats=False)
        self.posembed=nn.Embedding(10240, gv["posembedsize"])
        self.deviceid=deviceid
        self.device=torch.device('cuda:'+str(self.deviceid))
    
    def norm(self, out):
        outshape=out.shape
        out=out.reshape(-1,out.shape[-1])
        out=(out-out.mean(dim=0))/out.std(dim=0).clamp(gv["eps"])
        out = F.batch_norm(out, running_mean=None, running_var=None, weight=None, bias=None, training=True)
        out=out.reshape(outshape)
        return out
    
    def pools(self, x, stride):
        x=x.reshape(-1, stride, x.shape[1], x.shape[2]).float()
        x=x.permute(1,0,2,3)
        cpools=[]
        if "avg" in self.poolnames:
            cpools.append(x.mean(dim=0))
        if "std" in self.poolnames:
            cpools.append(x.std(dim=0))
        if "maxv" in self.poolnames:
            cpools.append(x.max(dim=0).values)
        if "minv" in self.poolnames:
            cpools.append(x.min(dim=0).values)
        if "s" in self.poolnames:
            cpools.append(x[0])
        if "e" in self.poolnames:
            cpools.append(x[-1])
        if self.corr:
            corr=self.seqcorr(x)
            cpools.append(corr)
        out=torch.cat(cpools, dim=2)
        return out
    def forward(self, x, hidden):
        device=hidden.device

        xpools=self.pools(x, self.xstride)
        xpools=self.do(xpools)
        
        hidden=hidden.reshape(-1, self.hiddenstride, hidden.shape[1], hidden.shape[2])
        hidden=hidden.permute(1,0,2,3)
        hidden=hidden.permute(1,2,3,0).reshape(hidden.shape[1], hidden.shape[2], -1)
        pos=torch.arange(hidden.shape[0]).unsqueeze(1).repeat([1,hidden.shape[1]]).to(device).long()
        posembed=self.posembed(pos)*gv["posembed"]
        # 

        inputx=torch.cat([xpools, hidden, posembed], dim=2)
        # inputx=torch.cat([hidden, posembed], dim=2)
        x=self.csm1(inputx)
        loss=None
            
        return x, loss
    
    
class S1Model(nn.Module):
    def __init__(self, crosssize=32, mseqlen=8, deviceid=0, section=False, reserved=0):
        super(S1Model, self).__init__()
        self.cvnum=122 
        self.sidembed=nn.Embedding(10000, gv["sidembedsize"])
        # self.sidembed.weight.data.fill_(0.0)
        self.dcss1=CssModel(self.cvnum + gv["sidembedsize"], gv["s15hiddensize"], bnlen=0, dropout=gv["dropout"])
        self.seqlen=gv["s1seqlen"]
        cvnumnew = self.cvnum + gv["s15hiddensize"] * gv["hidpool"]
        self.seqextract1=SeqExtractor(cvnumnew, gv["s15hiddensize"], int(crosssize), mseqlen, mseqlen, 
                                       poolnames=["avg","std","maxv","minv"], 
                                      # poolnames=[], 
                                      corrsize=gv["s15corrsize"],
                                      corr=True,
                                      mt=0, deviceid=deviceid, section=section)
        self.seqextract2=SeqExtractor(cvnumnew, gv["s15hiddensize"], int(crosssize), mseqlen*mseqlen, mseqlen, 
                                       poolnames=["avg","std","maxv","minv"], 
                                      # poolnames=[], 
                                      corrsize=gv["s15corrsize"],
                                      corr=True,
                                      mt=0, deviceid=deviceid, section=section)
        self.seqextract3=SeqExtractor(cvnumnew, gv["s15corrsize"], int(crosssize), mseqlen*mseqlen*mseqlen, mseqlen, 
                                      # poolnames=["avg","std","maxv","minv"], 
                                      poolnames=[], 
                                      corrsize=gv["s15corrsize"],
                                      corr=True,
                                      mt=0, deviceid=deviceid, section=section)
        self.do=nn.Dropout(gv["dropout"])
        self.reserved=4
        

    def forward(self, tidx, valid):
        sidembedding = self.sidembed(torch.arange(valid.shape[0]).cuda().long())
        dcv=d2i.gets1seq(gv["insdict"].dd, tidx, self.seqlen)
        dcv[~dcv.isfinite()]=0
        dcv=dcv[:,valid,:]
        sidembedding=sidembedding[valid]*gv["useseqsidembed"]
        inputs=[]
        tmpdcv=torch.cat([dcv, sidembedding.unsqueeze(dim=0).repeat(dcv.shape[0], 1,1)], dim=-1)
        dx=self.dcss1(tmpdcv)
        if gv["hidpool"]:
            dcv=torch.cat([dcv, dx], dim=-1)
        dx, dloss=self.seqextract1(dcv, dx)
        inputs.append(dx[-self.reserved:])
        dx, dloss=self.seqextract2(dcv, dx)
        inputs.append(dx[-self.reserved:])
        if gv["seqlayer3"]:
            dx, dloss=self.seqextract3(dcv, dx)
            inputs.append(dx[-self.reserved:])
        for i in range(len(inputs)):
            xshape=inputs[i].shape
            inputs[i]=inputs[i].permute(1,0,2).reshape(xshape[1], -1)
        return inputs
    
class S5Model(nn.Module):
    def __init__(self, crosssize=32, mseqlen=8, deviceid=0, section=False, reserved=0):
        super(S5Model, self).__init__()
        self.cvnum=122
        self.dcss1=CssModel(self.cvnum, gv["s15hiddensize"], bnlen=0, dropout=gv["dropout"])
        self.seqlen=gv["s5seqlen"]
        cvnumnew = self.cvnum + gv["s15hiddensize"] * gv["hidpool"]
        self.seqextract1=SeqExtractor(cvnumnew, gv["s15hiddensize"], int(crosssize), mseqlen, mseqlen, 
                                       poolnames=["avg","std","maxv","minv"], 
                                      # poolnames=[], 
                                      corrsize=gv["s15corrsize"],
                                      corr=True,
                                      mt=0, deviceid=deviceid, section=section)
        self.seqextract2=SeqExtractor(cvnumnew, gv["s15hiddensize"], int(crosssize), mseqlen*mseqlen, mseqlen, 
                                       poolnames=["avg","std","maxv","minv"], 
                                      # poolnames=[], 
                                      corr=True,
                                      corrsize=gv["s15corrsize"],
                                      mt=0, deviceid=deviceid, section=section)
        self.seqextract3=SeqExtractor(cvnumnew, gv["s15corrsize"], int(crosssize), mseqlen*mseqlen*mseqlen, mseqlen, 
                                      # poolnames=["avg","std","maxv","minv"], 
                                      poolnames=[], 
                                      corrsize=gv["s15corrsize"],
                                      corr=True,
                                      mt=0, deviceid=deviceid, section=section)
        self.do=nn.Dropout(gv["dropout"])
        self.reserved=4
        

    def forward(self, tidx, valid):
        dcv=d2i.gets5seq(gv["insdict"].dd, tidx, self.seqlen)
        dcv[~dcv.isfinite()]=0
        dcv=dcv[:,valid,:]
        inputs=[]
        dx=self.dcss1(dcv)
        if gv["hidpool"]:
            dcv=torch.cat([dcv, dx], dim=-1)
        dx, dloss=self.seqextract1(dcv, dx)
        inputs.append(dx[-self.reserved:])
        dx, dloss=self.seqextract2(dcv, dx)
        inputs.append(dx[-self.reserved:])
        if gv["seqlayer3"]:
            dx, dloss=self.seqextract3(dcv, dx)
            inputs.append(dx[-self.reserved:])
        for i in range(len(inputs)):
            xshape=inputs[i].shape
            inputs[i]=inputs[i].permute(1,0,2).reshape(xshape[1], -1)
        return inputs

g_lock = threading.Lock()   
def emptygpumem():
    g_lock.acquire()
    torch.cuda.empty_cache()
    g_lock.release()

    
def printself():
    with open(__file__, 'r', encoding='unicode_escape') as file_obj:
        contents = file_obj.read()
    print(contents)   

def main():
    m=MLPmodel()
    torch.save(m, "basemodel")
    
    
if __name__ == "__main__":
    main()


