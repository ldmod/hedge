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
import cryptoqt.smlp.alpha.tools as tools
from cryptoqt.smlp.alpha.tools import *
from cryptoqt.smlp.alpha.yamlcfg import gv
import pandas as pd
#from functorch import vmap
from functools import reduce
import threading
from scipy.stats import rankdata
# from memory_profiler import profile
# from .gpu_mem_track import MemTracker as MemTracker
import inspect
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED, ALL_COMPLETED



class OriDataPool(nn.Module):
    def __init__(self, orifeanum, kernel_size=2, stride=2):
        super(OriDataPool, self).__init__()
        self.maxpool=nn.MaxPool1d(kernel_size, stride)
        self.avgpool=nn.AvgPool1d(kernel_size, stride)
        self.orifeanum=orifeanum
        self.kernel_size, self.stride = kernel_size, stride
    
    def forward(self, x):
        pools=[]
        off=0
        for k in range(off, off+5):
            pools.append(self.avgpool(x[:, k:k+1, :]))
        off+=5
        pools.append(x[:,off:off+1,::self.stride]) #open
        off+=1
        pools.append(x[:,off:off+1,self.stride-1::self.stride]) #close
        off+=1
        pools.append(self.maxpool(x[:,off:off+1,:])) #high
        off+=1
        pools.append(-1.0*self.maxpool(-1.0*x[:,off:off+1,:])) #low
        off+=1
        pools.append(x[:,off:off+1,1::self.stride]) #AdjSettlePrice
        off+=1
        pools.append(self.avgpool(x[:, off:off+1, :]))
        off+=1
        # volume=x[:, 4:4+1, :]
        # pools.append(self.avgpool(x[:, off:off+1, :]*volume)/self.avgpool(volume).clamp(gv["eps"])) 
        pools.append(self.avgpool(x[:, off:off+1, :]))
        off+=1
        for k in range(off, off+4):
            pools.append(self.avgpool(x[:, k:k+1, :]))
        off+=4

        x=torch.cat(pools, dim=1)
        return x
    
class SeqConvs(nn.Module):
    def __init__(self, orifeanum, inputsize, crosssize, kernel_size=2, stride=2):
        super(SeqConvs, self).__init__()
        self.conv1=SeqCssModel(inputsize, crosssize, kernel_size, stride)
        self.oripool=OriDataPool(orifeanum, kernel_size, stride)
        self.maxpool=nn.MaxPool1d(kernel_size, stride)
        self.avgpool=nn.AvgPool1d(kernel_size, stride)
        self.orifeanum=orifeanum
    
    def forward(self, x):
        x=x.permute(1,2,0)
        x1=x[:, :self.orifeanum, :]
        x1=self.oripool(x1)
        x=self.conv1(x)
        x=torch.cat([x1, x], dim=1)
        x=x.permute(2,0,1)
        return x

class AvgPools(nn.Module):
    def __init__(self, min_kernel_size, kernelnum, expand=4):
        super(AvgPools, self).__init__()
        self.avgms=nn.ModuleList([])
        for i in range(0,kernelnum):
            x=min_kernel_size*expand**i
            self.avgms.append(nn.AvgPool1d(x,1))
    
    def forward(self, x):
        x=x.permute(1,2,0)
        max_kernel_size=self.avgms[-1].kernel_size[0]
        xshape=x.shape
        x=torch.cat([x[:,:,0:1].repeat([1,1,max_kernel_size]), x], dim=2)
        outs=[]
        for avgm in self.avgms:
            kernel_size=avgm.kernel_size[0]
            out=avgm(x[:,:,-xshape[2]-kernel_size+1:])
            outs.append(out)
        outs=torch.cat(outs, dim=1)
        outs=outs.permute(2,0,1)
        return outs
    
class SeqPools(nn.Module):
    def __init__(self, inputsize, crosssize):
        super(SeqPools, self).__init__()
        self.csm1=CssModel(inputsize, crosssize)
        self.avgms1=AvgPools(gv["minkernel"], gv["expandnum"], gv["expandsize"])
    
    def forward(self, inp, x):
        cs1=self.csm1(x)
        csseq1=self.avgms1(cs1)
        x=torch.cat([inp, cs1, csseq1], dim=2)
        return x
    
class SeqConvs(nn.Module):
    def __init__(self, orifeanum, inputsize, crosssize, kernel_size=2, stride=2):
        super(SeqConvs, self).__init__()
        self.conv1=SeqCssModel(inputsize, crosssize, kernel_size, stride)
        self.oripool=OriDataPool(orifeanum, kernel_size, stride)
        self.maxpool=nn.MaxPool1d(kernel_size, stride)
        self.avgpool=nn.AvgPool1d(kernel_size, stride)
        self.orifeanum=orifeanum
    
    def forward(self, x):
        x=x.permute(1,2,0)
        x1=x[:, :self.orifeanum, :]
        x1=self.oripool(x1)
        x=self.conv1(x)
        x=torch.cat([x1, x], dim=1)
        x=x.permute(2,0,1)
        return x
    
class SectionCss(nn.Module):
    def __init__(self, inputsize, crosssize, sectionsize=0, hiddensize=128, dropout=0.1, bnlen=0):
        super(SectionCss, self).__init__()
        sectionsize = gv["tickernum"] if sectionsize==0 else sectionsize
        self.css1=CssModel(inputsize*2, crosssize, dropout=dropout, bnlen=bnlen)
        self.litecss=LiteCssModel(sectionsize, hiddensize, dropout=dropout)
    
    def forward(self, x):
        xdim=list(range(len(x.shape)))
        pxdim=xdim[:-2]+[xdim[-1],xdim[-2]]
        xsection=self.litecss(x.permute(pxdim)).permute(pxdim)
        x=torch.cat([x,xsection], dim=-1)
        x=self.css1(x)
        return x

class SectionCss2(nn.Module):
    def __init__(self, sectionsize, inputsize, crosssize, hiddensize=128, dropout=0.1, bnlen=0):
        super(SectionCss2, self).__init__()
        self.css1=CssModel(inputsize+crosssize, crosssize, dropout=dropout, bnlen=bnlen)
        self.sectionm1=CssModel(sectionsize, 4)
        self.sectionm2=CssModel(inputsize, int(crosssize/4))
        self.sectionsize=sectionsize
    
    def forward(self, x):
        xdim=list(range(len(x.shape)))
        pxdim=xdim[:-2]+[xdim[-1],xdim[-2]]
        xsection=self.sectionm1(x.permute(pxdim)).permute(pxdim)
        xsection=self.sectionm2(xsection).reshape(x.shape[:-2]+(1,-1))
        x=torch.cat([x,xsection.repeat([1 for i in range(len(xsection.shape)-2)]+[self.sectionsize, 1])], dim=-1)
        x=self.css1(x)
        return x
    
class SeqCorr(nn.Module):
    def __init__(self, inputsize, crosssize, dropout=0.1):
        super(SeqCorr, self).__init__()
        if gv["replacecat"]:
            self.css1=OptLinearModel(inputsize, crosssize)
            self.css2=OptLinearModel(inputsize, crosssize)
        else:
            self.css1=nn.Linear(in_features=inputsize, out_features=crosssize, bias=True)
            self.css2=nn.Linear(in_features=inputsize, out_features=crosssize, bias=True)
        self.do=nn.Dropout(gv["dropout"])
    
    def forward(self, x):
        h1=self.css1(x)
        h2=self.css2(x)
        h1=self.do(h1)
        h2=self.do(h2)
        cos=nn.CosineSimilarity(dim=0)(h1,h2)
        return cos

class SeqExtractorDay(nn.Module):
    def __init__(self, dxcvnum, xcvnum, inputsize, crosssize, dxs, xstride, hiddenstride):
        super(SeqExtractorDay, self).__init__()
        self.xcvnum=xcvnum
        self.hiddenstride=hiddenstride
        self.xstride=xstride
        self.dxs=dxs
        dxinputlen=dxcvnum*6
        self.inputxlen=xcvnum*6+inputsize*hiddenstride+dxinputlen
        self.csm1=CssModel(self.inputxlen, crosssize, bnlen=1, dropout=gv["dropout"])
        self.bnlayer=nn.BatchNorm1d(self.inputxlen, track_running_stats=False)
    
    def pools(self, x, stride):
        x=x.reshape(stride, -1, x.shape[1], x.shape[2]).float()
        avg=x.mean(dim=0)
        std=x.std(dim=0)
        maxv=x.max(dim=0).values
        minv=x.min(dim=0).values
        s=x[0]
        e=x[-1]
        out=torch.cat([avg, std, maxv, minv, s, e], dim=2)
        outshape=out.shape
        out=out.reshape(-1,out.shape[-1])
        out=(out-out.mean(dim=0))/out.std(dim=0).clamp(gv["eps"])
        out=out.reshape(outshape)
        return out
    def forward(self, dx, x, hidden):
        dxpools=self.pools(dx, self.dxs)
        xpools=self.pools(x, self.xstride)
        
        hidden=hidden.reshape(self.hiddenstride, -1, hidden.shape[1], hidden.shape[2])
        hidden=hidden.permute(1,2,3,0).reshape(hidden.shape[1], hidden.shape[2], -1)
        inputx=torch.cat([dxpools, xpools, hidden], dim=2)
        inputx=self.bnlayer(inputx.reshape(-1, self.inputxlen)).reshape(inputx.shape)
        x=self.csm1(inputx)   
            
        #emptygpumem()
        return x

def logop(data, fieldnames, allfields, scales):
    idxs=[]
    for field in fieldnames:
        idxs.append(rawdataday.getfieldidx(allfields, field))
    idxs=torch.tensor(idxs).to(data.device)
    fieldscale=torch.tensor(scales).to(data.device)[idxs]
    data=torch.index_select(data, -1, idxs)
    datalog=(data/fieldscale).log()
    return datalog
def stocktimenormop(data, fieldnames, allfields, scales):
    idxs=[]
    for field in fieldnames:
        idxs.append(rawdataday.getfieldidx(allfields, field))
    idxs=torch.tensor(idxs).to(data.device)
    fieldscale=torch.tensor(scales).to(data.device)[idxs]
    data=torch.index_select(data, -1, idxs)
    # datamean=data.nanmean(dim=0)
    # datastd=data.std(dim=0).clamp(gv["eps"])
    datanorm=tools.tensornorm(data, gv["eps"], 0)
    return datanorm
def stockgradop(data, fieldnames, allfields, scales, graddeltas=None):
    idxs=[]
    for field in fieldnames:
        idxs.append(rawdataday.getfieldidx(allfields, field))
    idxs=torch.tensor(idxs).to(data.device)
    fieldscale=torch.tensor(scales).to(data.device)[idxs]
    data=torch.index_select(data, -1, idxs)
    grads=[]
    if graddeltas is None:
        graddeltas=gv["graddeltas"]
    for delta in graddeltas:    
        grad=data[delta:]-data[:-delta]
        grad=torch.cat([torch.zeros(delta, grad.shape[1], grad.shape[2]).to(grad.device), grad], dim=0)
        grads.append(grad)
    grads=torch.cat(grads, dim=-1)
    return grads


class FeatureModel(nn.Module):
    def __init__(self, randomins=0, lrr=1.0, deviceid=0, saveresdict=False, section=False, lastday=True, sampleinsnum=None):
        super(FeatureModel, self).__init__()
        torch.cuda.set_device(deviceid)
        self.randomins=randomins
        self.sampleinsnum=sampleinsnum
        self.name="feamodel_"+str(randomins)
        self.movingmean, self.movingstd, self.daymovingmean, self.daymovingstd=None, None, None, None
        self.tvridxs=[0]
        self.deviceid=deviceid
        self.device=torch.device('cuda:'+str(self.deviceid))
        self.tvrratio=torch.tensor([0.0]).to(self.device)
        self.resdict={}
        self.reservedm=1
        self.saveresdict=saveresdict
        self.tmpdict={}
        self.lrr=lrr
        self.lastday=lastday
        torch.cuda.set_device(0)
        
    
    def hsize(self):
        return self.hiddensize
    
    def calcloss(self, tidx, x, y, vf):
        x, y=x[vf], y[vf]
        cosloss=-1.0*nn.CosineSimilarity(dim=0)(x, y)
        lossres=dict(loss=cosloss.mean(), cos=cosloss.mean()*-1.0)
        return lossres
    
    def calcfee(self, tidx, x, y, vf, lastq, lastvf, calctop):
        return self.mincalcfee(tidx, x, y, vf, lastq, lastvf, calctop)
    
    def mincalcfee(self, tidx, x, y, vf, lastq, lastvf, calctop):
        return None
    
    def ydays(self):
        return gv["tmdelta"]

    def getyy(self, tidx):
        return self.mingetyy(tidx)
    
    def getyy2(self, tidx):
        return self.mingetyy(tidx)
    
    def mingetyy(self, tidx):
        return None

    
    def norm(self, x, mean, std):
        if not mean is None:
            x-=mean
        x[~x.isfinite()]=0
        if not std is None:
            x=x/std.clamp(1e-8)
        return x
    
    def updatenorm(self, dx, x):
        decay=0.999
        mean=x.reshape(-1,x.shape[-1]).nanmean(dim=0)
        x=x-mean
        x[~x.isfinite()]=0
        std=x.reshape(-1,x.shape[-1]).std(dim=0)
        dmean=dx.reshape(-1,dx.shape[-1]).nanmean(dim=0)
        dx=dx-dmean
        dx[~dx.isfinite()]=0
        dstd=dx.reshape(-1,dx.shape[-1]).std(dim=0)
        if self.movingmean is None:
            self.movingmean, self.movingstd = mean, std
            self.daymovingmean, self.daymovingstd = dmean, dstd
        else:
            self.movingmean = self.movingmean*decay+(1-decay)*mean
            self.movingstd = self.movingstd*decay+(1-decay)*std
            self.daymovingmean = self.daymovingmean*decay+(1-decay)*dmean
            self.daymovingstd = self.daymovingstd*decay+(1-decay)*dstd
         
    def inputprocess(self, tidx):
        de=int((tidx+1)/gv["insperday"])+1
        ds=de-self.seqlen
        s=(ds-1)*gv["insperday"]
        e=tidx+1
        dx, x=gv["oriins"].getx(s,e)
        x=x[-self.headdays*gv["insperday"]:]
        dx,x = dx.to(self.device).detach(), x.to(self.device).detach()
        ids=dx[:,:,:3].long()
        timeid=x[:,:,0].long()
        vf=x[-1,:,1] > 0
        if gv["next5minhighlimit"]:
            limitidx=rawdataday.getfieldidx(rawdataday.g_minfields, 'min_g_highlimit')
            vf=(x[-1,:,1] > 0) & (x[-1,:,limitidx] < 1)

        vf=vf.flatten()
        dcv=dx[:,:,:].float()
        cv=x[:,:,:].float()
        if len(gv["daylog"]) > 0:
            daylog=logop(dcv, gv["daylog"], rawdataday.g_dayfields, gv["dayscale"])
            dcv=torch.cat([dcv, daylog], dim=-1)
        if len(gv["minlog"]) > 0:
            minlog=logop(cv, gv["minlog"], rawdataday.g_minfields, gv["minscale"])
            cv=torch.cat([cv, minlog], dim=-1)
        if len(gv["daytimenorm"]) > 0:
            daytimenorm=stocktimenormop(dcv, gv["daytimenorm"], rawdataday.g_dayfields, gv["dayscale"])
            dcv=torch.cat([dcv, daytimenorm], dim=-1)
        if len(gv["mintimenorm"]) > 0:
            mintimenorm=stocktimenormop(cv, gv["mintimenorm"], rawdataday.g_minfields, gv["minscale"])
            cv=torch.cat([cv, mintimenorm], dim=-1)
        if len(gv["daygrad"]) > 0:
            daygrad=stockgradop(dcv, gv["daygrad"], rawdataday.g_dayfields, gv["dayscale"])
            dcv=torch.cat([dcv, daygrad], dim=-1)
        if len(gv["mingrad"]) > 0:
            mingrad=stockgradop(cv, gv["mingrad"], rawdataday.g_minfields, gv["minscale"])
            cv=torch.cat([cv, mingrad], dim=-1)
        if self.training:
            self.updatenorm(dcv.detach(), cv.detach())
        dcv=self.norm(dcv, self.daymovingmean, self.daymovingstd)
        cv=self.norm(cv, self.movingmean, self.movingstd)
        
        return dcv.detach(), cv.detach(), vf.detach(), ids, timeid, None

    def trans15input(self, tidx):
        de=int(tidx/gv["min2t15"])+1
        ds=de-gv["trans15seqlen"]

        trans15x=gv["oriins"].gettrans15x(ds, de)
        trans15x= trans15x.to(self.device).detach().float()

        if len(gv["trans15log"]) > 0:
            trans15log=logop(trans15x, gv["trans15log"], rawdataday.g_trans15fields, gv["trans15scale"])
            trans15x=torch.cat([trans15x, trans15log], dim=-1)
            del trans15log
        if len(gv["trans15timenorm"]) > 0:
            trans15timenorm=stocktimenormop(trans15x, gv["trans15timenorm"], rawdataday.g_trans15fields, gv["trans15scale"])
            trans15x=torch.cat([trans15x, trans15timenorm], dim=-1)
            del trans15timenorm
        if len(gv["trans15grad"]) > 0:
            trans15grad=stockgradop(trans15x, gv["trans15grad"], rawdataday.g_trans15fields, gv["trans15scale"])
            trans15x=torch.cat([trans15x, trans15grad], dim=-1)
            del trans15grad

        if self.training:
            self.trans15updatenorm(trans15x.detach())
        trans15x=self.norm(trans15x, self.trans15movingmean, self.trans15movingstd)
        
        return trans15x.detach().half()
    
    def trans15updatenorm(self, x):
        decay=0.999
        mean=x.reshape(-1,x.shape[-1]).nanmean(dim=0)
        # mean=splitmean(x)
        x=x-mean
        x[~x.isfinite()]=0
        std=x.reshape(-1,x.shape[-1]).std(dim=0)
        if self.trans15movingmean is None:
            self.trans15movingmean, self.trans15movingstd = mean, std
        else:
            self.trans15movingmean = self.trans15movingmean*decay+(1-decay)*mean
            self.trans15movingstd = self.trans15movingstd*decay+(1-decay)*std
            
    def trans1input(self, tidx):
        de=int(tidx/gv["min2t1"])+1
        ds=de-gv["trans1seqlen"]

        trans1x=gv["oriins"].gettrans1x(ds, de)
        trans1x= trans1x.to(self.device).detach().float()

        if len(gv["trans1log"]) > 0:
            trans1log=logop(trans1x, gv["trans1log"], rawdataday.g_trans1fields, gv["trans1scale"])
            trans1x=torch.cat([trans1x, trans1log], dim=-1)
            del trans1log
        if len(gv["trans1timenorm"]) > 0:
            trans1timenorm=stocktimenormop(trans1x, gv["trans1timenorm"], rawdataday.g_trans1fields, gv["trans1scale"])
            trans1x=torch.cat([trans1x, trans1timenorm], dim=-1)
            del trans1timenorm
        if len(gv["trans1grad"]) > 0:
            trans1grad=stockgradop(trans1x, gv["trans1grad"], rawdataday.g_trans1fields, gv["trans1scale"])
            trans1x=torch.cat([trans1x, trans1grad], dim=-1)
            del trans1grad

        if self.training:
            self.trans1updatenorm(trans1x.detach())
        trans1x=self.norm(trans1x, self.trans1movingmean, self.trans1movingstd)
        
        return trans1x.detach().half()
    
    def trans1updatenorm(self, x):
        decay=0.999
        mean=x.reshape(-1,x.shape[-1]).nanmean(dim=0)
        # mean=splitmean(x)
        x=x-mean
        x[~x.isfinite()]=0
        std=x.reshape(-1,x.shape[-1]).std(dim=0)
        if self.trans1movingmean is None:
            self.trans1movingmean, self.trans1movingstd = mean, std
        else:
            self.trans1movingmean = self.trans1movingmean*decay+(1-decay)*mean
            self.trans1movingstd = self.trans1movingstd*decay+(1-decay)*std
    
            
    def forward(self, tidx, return_y=False):
        #dlimodel
        pass
    
    def pred(self, tidx):
        with torch.no_grad():
            self.eval()
            x, vf, xetr = self.forward(tidx)
            self.train()
        return x.detach(), vf.detach(), xetr
        
    def get_traindays(self, traintidx):
        setup_seed(traintidx)
        traintidx-=self.ydays()
        ip=gv["insperday"]
        if self.randomins > 0:
            randomstart=max(gv["oriins"].x.s+(gv["seqlen"]+16)*ip, traintidx-(self.randomins-128)*ip )
            days=list(range(randomstart, traintidx-ip))
            traindays = random.sample(days, self.sampleinsnum)
            if self.lastday:
                traindays+=[traintidx]
            if gv["sametimeid"]:
                traindays=(np.array(traindays)/ip).astype(int)*ip
            traindays=(np.array(traindays)-traintidx%gv["step"])/gv["step"]
            traindays=traindays.astype(int)*gv["step"]+traintidx%gv["step"]
        else:
            traindays=[traintidx-(gv["tmdelta"]-gv["step"])*gv["trainbackdelta"]*i for i in range(self.sampleinsnum)]
            # traindays=np.array([traintidx-gv["trainbackdelta"]*ip,traintidx])
            traindays.sort()
            traindays=np.array(traindays)
            traindays=traindays[traindays>gv["oriins"].x.s+(gv["seqlen"]+1)*ip]
        traindays=traindays.tolist()
        return traindays
       
    def start_asyncupdatewm(self, cur_tidx):
        th1 = threading.Thread(target=self.updatewm, args=(cur_tidx,))
        th1.start()
        return th1
    def wait_asyncupdatewm(self, th1):
        th1.join()
        
    def updatewm(self, cur_tidx):
        tpf=TimeProfilingLoop(self.name+"train")
        traindays=self.get_traindays(cur_tidx)
        for tidx,curtimeidx in enumerate(traindays):
            disw=max(0.5, 1-(cur_tidx-curtimeidx)/gv["insperday"]*0.002) if gv["disw"] else 1.0

            if curtimeidx == traindays[-1]:
                lrbak=self.opt.param_groups[0]['lr']
                self.opt.param_groups[0]['lr']=lrbak*gv["cvlr"]
            frowardpf=tpf.add(self.name+"feedforward")
            output, vf, xetr, y, loss = self.forward(curtimeidx, return_y=True)

            frowardpf.end()
            losssum=loss["loss"]*disw*gv["dynamiclr"]
            backwardpf=tpf.add(self.name+"backward")
            self.opt.zero_grad()
            losssum.backward()
            
            if self.tvrratio.sum()>0.0  and (cur_tidx-gv["step"]) in self.oli.xx:
                lastq, lastvf=self.oli.xx[cur_tidx-gv["step"]], self.oli.vfs[cur_tidx-gv["step"]]
                output, vf, xetr=self.forward(cur_tidx, return_y=False)

                lastw=getsimratio(lastq[:, self.tvridxs].to(self.device), lastvf.to(self.device))
                w=getsimratio(output[:, self.tvridxs], vf)
                lossregw=-(nn.CosineSimilarity(dim=0)(lastw.detach(), w)*self.tvrratio).mean()
                oripredtvr=-(nn.CosineSimilarity(dim=0)(lastq.to(self.device).detach(), output)*self.tvrratio*gv["oripredtvrratio"]).mean()
                lossregw=lossregw*disw+oripredtvr*disw
                lossregw.backward()
                self.opt.step()
            else:
                lossregw=0
                self.opt.step()
            if curtimeidx == traindays[-1]:
                self.opt.param_groups[0]['lr']=lrbak
            backwardpf.end()

        tpf.end()
        timestr=tpf.to_string()
        flog("tm:", cur_tidx, timestr, traindays, "losssum:", losssum, "std:", output.std(dim=0))
    
    def start_asynctestday(self, tidx):
        th1 = threading.Thread(target=self.testday, args=(tidx,))
        th1.start()
        return th1
    
    def testday(self, tidx):
        latestyidx=tidx-self.ydays()*gv["traindelay"]
        self.olitrain.infferandstats(self, latestyidx-self.ydays()*gv["insperday"]*7)
        x, vf, xetr = self.oli.addx(self, tidx)
        y_latest=self.oli.addy(self, latestyidx)
        df=self.oli.calcstats(self, latestyidx)
        extradata={}
        extradata["yy2"]=self.getyy2(tidx-gv["tmdelta"])
        if self.saveresdict:
            if not y_latest is None and gv["ts"][latestyidx] in self.resdict:
                dd=self.resdict[gv["ts"][latestyidx]]
                dd["y"]=y_latest
                dd["df"]=df
                # self.resdict[gv["ts"][latestyidx]]=self.resdict[gv["ts"][latestyidx]][:2]+(y_latest, df)
            self.resdict[gv["ts"][tidx]]=dict(x=x, vf=vf, extradata=extradata, xetr=xetr)
        
        return x, vf, xetr
