#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 09:40:58 2023

@author: dli
"""
import torch
import torch.nn as nn

from cryptoqt.smlp.alpha.tools import flog as flog
from cryptoqt.smlp.alpha.tools import CssModel as CssModel
from cryptoqt.smlp.alpha.tools import SeqCssModel as SeqCssModel

import cryptoqt.smlp.alpha.tools as tools
import cryptoqt.smlp.alpha.seqfea as seqfea
import cryptoqt.smlp.alpha.common as common
import cryptoqt.smlp.alpha.mergemodel as mmodel
import cryptoqt.smlp.alpha.yamlcfg as yamlcfg
from cryptoqt.smlp.alpha.yamlcfg import gv
import numpy as np
from audtorch.metrics.functional import pearsonr
import random
import datetime
import time
import os
import io
import threading
import cryptoqt.smlp.alpha.data2ins as d2i
import cryptoqt.data.constants as conts
from cryptoqt.smlp.alpha.tools import *

class PredM:
    def __init__(self):
        self.obj=None
        # self.reskeys=["seqm1"] 
    def gettrainins(self, traintidx):
        if gv["randomins"]:
            days=list(range(self.oriins.x.s+1024*48, traintidx))
            traindays = random.sample(days,gv["traindeltanum"])
        else:
            traindays=[traintidx-(gv["reduce"]-gv["step"])*gv["trainbackdelta"]*i for i in range(gv["traindeltanum"])]
            traindays=np.array(traindays)
            traindays.sort()
            traindays=traindays[traindays>self.oriins.x.s+1024*48]
        return traindays
    
    def pred(self, xhis,yhis,tshis, predcfg={"checkflag":False}):
        return 
            
    def pred(self, testtidx):
        tfp=tools.TimeProfilingLoop("pred")
        model=self.model
        x, vf, hidden= self.model.testday(testtidx)
        tfp.end()
        flog("test tm:", testtidx, "tmcose:",tfp.to_string(), time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        # torch.cuda.empty_cache()
        return x,vf, hidden 
    
    def tune(self, traintidx):
        tfp=tools.TimeProfilingLoop("tune")
        tools.setup_seed(traintidx)
        model=self.model
        self.model.updatewm(traintidx)
        
        self.traineddate=gv["ts"][traintidx]
        tfp.end()
        flog("tune day:", traintidx, "tmcose:",tfp.to_string(), time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    
    def checkpoint(self, path):
        gvbak= self.gv
        resd=[self.model.resdict]
        self.model.resdict={}
        torch.save(resd, gv["model_path"]+".res")
        self.gv = None
        buffer = io.BytesIO()
        torch.save(self, buffer)
        buffer.seek(0)
        self.gv = gvbak 
        self.model.resdict = resd[0]
        off=1
        flog("save info:", gv["model_path"], len(buffer.getvalue()))
        dd=dict(obj=self.obj, pred=buffer)
        torch.save(dd, path)
        
    def hotstart(self, path, flag=False):
        global gv
        dd=torch.load(path)
        prefix=path.split('_date_')[0]
        modeldate=path.split('_date_')[1]
        dd["pred"].seek(0)
        pred=torch.load(dd["pred"])
        self.model, self.gv=pred.model, gv
        self.lastdt=pred.lastdt
        if os.path.exists(prefix+".res"):
            resd=torch.load(prefix+".res")
            self.model.resdict = resd[0]
        else:
            self.model.resdict={}
        off=1

        return modeldate
   
    
    def coldstart(self):
        self.gv=gv
        if gv["mergemodelname"] == "base":
            self.model=MLPmodel().cuda()
        elif gv["mergemodelname"] == "combo":
            self.model=combomodel.ComboModel().cuda()
        else:
            self.model=MLPmodel().cuda()
        gv["tmfea"]={}     
    

class AttCssModel(nn.Module):
    def __init__(self, inputsize, outsize):
        super(AttCssModel, self).__init__()
        self.do=nn.Dropout(gv["dropout"])
        self.secm1 = CssModel(inputsize, outsize, bnlen=1)
        self.attm = tools.AttentionM(inputsize, outsize, bn=True)
    
    def forward(self, x):
        x1=self.secm1(x)
        x2=self.attm(x)
        x3=torch.cat([x1, x2], dim=-1)
        return x3

class SectionCss(nn.Module):
    def __init__(self, inputsize, crosssize, sectionsize=0, dropout=0.1):
        super(SectionCss, self).__init__()
        sectionsize = len(gv["baseuids"]) if sectionsize==0 else sectionsize
        self.css1=CssModel(inputsize, crosssize, dropout=dropout)
        self.css2=CssModel(sectionsize, crosssize, dropout=dropout)

    def forward(self, x):
        x=self.css1(x)
        x=self.css2(x.permute(1,0))
        return x.flatten()
    

class MmModel(nn.Module):
    def __init__(self, inputsize, outsize):
        super(MmModel, self).__init__()
        self.do=nn.Dropout(gv["dropout"])
        cssize1=256
        cssize2=128
        scsize=16
        self.css1 = CssModel(inputsize, cssize1, bnlen=1, dropout=gv["dropout"])
        self.css4 = CssModel(cssize1, 1)

    def forward(self, x):
        x=self.css1(x)
        x=self.css4(x)
        return x
    
def seqparseparam(name):
    ss=name.split("_")
    params={}
    params["seqlen"]=int(ss[1])
    params["ld"]=False if ss[2][2:]=="False" else True
    params["sn"]=int(ss[3][2:])
    return params
    
    
class MLPmodel(seqfea.FeatureModel):
    def __init__(self):
        super(MLPmodel, self).__init__()
        self.name="mergemodel"
        inputsize=0
        oriinput=19
        # oriinput=1
        self.daym=seqfea.DayModel(crosssize=16, mseqlen=8)
        self.dayembedlen=128
        self.h1m=seqfea.H1Model(crosssize=16, mseqlen=8)
        self.h1embedlen=128
        self.min15m=seqfea.Min15Model(crosssize=16, mseqlen=8)
        self.min15embedlen=128
        crosssize=gv["min15hiddensize"]
        self.min1m=seqfea.Min1Model(crosssize=crosssize, mseqlen=gv["mseqlen"])
        self.min1embedlen=int(crosssize*(gv["reserved"]*2+gv["seqlayer3"]*4))
        self.min5m=seqfea.Min5Model(crosssize=crosssize, mseqlen=gv["mseqlen"])
        self.min5embedlen=int(crosssize*(gv["reserved"]*2+gv["seqlayer3"]*4))
        self.sidembed=nn.Embedding(10000, gv["sidembedsize"])
        self.sidembed.weight.data.fill_(0.0)
        hiddenfeasize=0
        if "hiddenfealist" in gv and len(gv["hiddenfealist"])>0:
            for item in gv["hiddenfealist"]:
                hiddenfeasize+=item[1]
        inputsize=gv["sidembedsize"]+gv["orim"]*oriinput+gv["daym"]*self.dayembedlen+\
            gv["h1m"]*self.h1embedlen+gv["min15m"]*self.min15embedlen+hiddenfeasize+gv["min1m"]*self.min1embedlen\
                +gv["min5m"]*self.min5embedlen
        self.do=nn.Dropout(gv["dropout"])
        cssize1=512
        cssize2=256
        self.css1 = CssModel(inputsize, cssize1, bnlen=1, dropout=gv["dropout"])
        self.css2 = CssModel(cssize1, cssize1, bnlen=1, dropout=gv["dropout"])
        self.css3 = CssModel(cssize1, cssize2, bnlen=1, dropout=gv["dropout"])
        self.secms=nn.ModuleList()
        for key in gv["outvalue"]:
            mm = MmModel(cssize2, 1)
            self.secms.append(mm)
             
        ###################opt#############################
        self.opt=tools.AdamWReg(self.named_parameters(), lr=gv["learning_rate"]*gv["mergelr"],
                            betas=(gv["beta1"], gv["beta2"]), weight_decay1=0.0, weight_decay2=0.1)
        self.oli=tools.OlInffer(self.name+"_test", calctop=True)
        self.olitrain=tools.OlInffer(self.name+"_train", calctop=True)
        self.saveresdict=True
        
        
    def pred(self, tidx):
        self.eval()
        x, vf, hidden = self.forward(tidx)
        self.train()
        return x.detach(), vf.detach(), hidden.detach()
    
    def forward(self, tidx, return_y=False):
        valid=d2i.getvalid(gv["savedr"], tidx)
        valid=torch.from_numpy(valid).cuda()
        sidembed=self.sidembed(torch.arange(valid.shape[0]).cuda().long())*(not gv["delmergesid"])
        oo=torch.zeros((valid.shape[0],len(gv["outvalue"]))).cuda()
            
        if (valid.isfinite() & valid).sum() < 2:
            tools.flog(gv["ts"][tidx], "valid sum:", valid.sum(), tidx, flush=True)
            return oo, valid, oo
        assert (valid.isfinite() & valid).sum()> 1, str(valid.sum())+"_"+str(gv["ts"][tidx])+"_"+str((valid.isfinite() & valid).sum())
        
        inputs=[sidembed[valid]]

        if gv["orim"]:
            x=d2i.getx(gv["savedr"], tidx)
            xshape=x.shape
            x=torch.from_numpy(x).cuda()
            x=x[valid]
            x=x-x.nanmean(dim=0)
            x[~x.isfinite()]=0
            inputs.append(x)
        if gv["daym"]:
            self.dayembed=self.daym(tidx, valid)
            inputs+=self.dayembed
        if gv["h1m"]:
            self.h1embed=self.h1m(tidx, valid)
            inputs+=self.h1embed
        if gv["min15m"]:
            self.min15embed=self.min15m(tidx, valid)
            inputs+=self.min15embed
        if gv["min5m"]:
            self.min5embed=self.min5m(tidx, valid)
            inputs+=self.min5embed
        if gv["min1m"]:
            self.min1embed=self.min1m(tidx, valid)
            inputs+=self.min1embed
        if "hiddenfealist" in gv and len(gv["hiddenfealist"])>0:
            for item in gv["hiddenfealist"]:
                hid=d2i.gethiddenfea(gv["data_dict"], item[0], tidx)
                hid=hid.reshape(len(gv["baseuids"]), -1)
                inputs.append(torch.from_numpy(hid).cuda()[valid])
        inp=torch.cat(inputs, dim=1)
        inp=self.css1(inp)
        inp=self.css2(inp)
        inp=self.css3(inp)
        hidden=torch.zeros((valid.shape[0], inp.shape[1])).cuda()
        hidden[valid]=inp.detach()
        xx=[]
        for idx in range(len(self.secms)):
            mm=self.secms[idx]
            tmpx=mm(inp)
            xx.append(tmpx)
        out=torch.cat(xx, dim=1)
        out=self.do(out)
        
        oo[valid]=out
        return oo, valid, hidden
    
    def get_traindays(self, tidx):
        delay=gv["traindelay"]
        tdi=int(tidx/gv["tmdelta"])
        days=list(range(max(int(tdi-365*1440/gv["tmdelta"]), 960), tdi-delay))
        self.sampleinsnum=gv["sampleinsnum"]
        tdays = random.sample(days, self.sampleinsnum)
        tdays=sorted(tdays)
        assigndays=gv["traindays"][:]+[delay]
        assigndays=tdi-np.array(assigndays)
        tdays+=assigndays.tolist()
        tdays=[item*gv["tmdelta"] for item in tdays]
        return tdays
    
    def ydays(self):
        days=gv["tmdelta"]
        return days 
    def getyy(self, tidx):
        y=d2i.gety(gv["savedr"], tidx, gv["tmdelta"])
        # y=d2i.gety(gv["savedr"], tidx, 120)
        valid=(d2i.getvalid(gv["savedr"], tidx) & np.isfinite(y))
        y=y-y[valid].mean()
        y=torch.from_numpy(y).cuda()
        y[~valid]=0
        y=y.reshape(-1,1).repeat(1,len(gv["outvalue"]))
        return y
        
    def rankfunc(self, y):
        y=(y*100).int()
        yv, yr=y.unique(sorted=True,return_inverse=True)
        yr=yr.float()
        yr=(yr-yr.mean())/yr.std().clamp(gv["eps"])
        return yr
            
    def calcloss(self, tidx, x, y, valid):
        oy=y[valid]
        y=oy[:,0].flatten()
        x=x[valid]
        
        ymean, ystd=y.mean(),y.std()
        ynorm=(y-ymean)/y.std(dim=0).clamp(gv["eps"])
        ynorm=ynorm.clamp(-10,10)
        
        
        yrank=self.rankfunc(y)
        #####
        longret=torch.from_numpy(d2i.getyl(gv["savedr"], tidx, int(gv["tmdelta"]*gv["longterm"]))).cuda()[valid]
        longret[~longret.isfinite()]=0.0
        longretnorm=(longret-longret.nanmean(dim=0))/longret.std(dim=0).clamp(gv["eps"])
        ylongrank = self.rankfunc(longret)
        #####


        cosloss=-1.0*nn.CosineSimilarity(dim=0)(x, oy) 
        ics=tools.pearsonr(x, oy, batch_first=False)[0]
        oyd=yrank.reshape(-1,1).repeat(1,len(gv["outvalue"]))
        icsd=tools.pearsonr(x, oyd, batch_first=False)[0]
        
        targetY = ylongrank if gv["longtermTarget"] else yrank
        
        mseloss=nn.MSELoss(reduction='none')(x[:,0], targetY*10).mean(dim=0)
        wsharp=(ics[1]*-1.0).exp().detach()
        mseloss1=(nn.MSELoss(reduction='none')(x[:,1], targetY*10)).mean(dim=0)*wsharp
        wsharp=(ics[2]*-0.0).exp().detach()
        mseloss2=(nn.MSELoss(reduction='none')(x[:,2], (ynorm)*10)).mean(dim=0)*wsharp
        wsharp=(ics[3]*-1.0).exp().detach()
        mseloss3=(nn.MSELoss(reduction='none')(x[:,3], ynorm*10)).mean(dim=0)*wsharp

        lossres=dict(loss=mseloss+mseloss1+mseloss2+mseloss3, 
                     ic=ics[0], valid=valid.sum(),
                     xstd=x.detach().std(),
                     ymean=ymean, ystd=ystd
                     )
        for i in range(len(gv["outvalue"])):
            lossres["ic"+str(i)]=ics[i]
        for i in range(len(gv["outvalue"])):
            lossres["icd"+str(i)]=icsd[i]
        return lossres
    
    def calcfee(self, tidx, x, y, vf, lastq, lastvf, calctop):
        wratio=getsimratio(x, vf)
        ret=(wratio*y)[vf].sum(dim=0)
        wsratio=getsimratio(-1.0*x, vf)
        rets=(wratio*y)[vf].sum(dim=0)
        lastw=getsimratio(lastq, lastvf)
        diffw=wratio-lastw
        tvr=diffw.abs().sum(dim=0)
        
        infos={}
        infos["ret"]=ret
        infos["rets"]=rets
        infos["tvr"]=tvr
        if calctop:
            for topk in [5, 10, 20, 50]:
                wtop=getwratiotop(x, vf, topk)
                rettop=(wtop*y)[vf].sum(dim=0)

                swtop=getwratiotop(-1.0*x, vf, topk)
                retstop=(swtop*y)[vf].sum(dim=0)

                lastwtop=getwratiotop(lastq, lastvf, topk)
                tvrpos=(lastwtop-wtop).abs().sum(dim=0)

                wtopavg=getwratiotopavg(x[vf], topk)
                rettopavg=(wtopavg*y[vf]).sum(dim=0)
                
                wtopavgs=getwratiotopavg(-x[vf], topk)
                rettopavgs=(wtopavgs*y[vf]).sum(dim=0)
            
                infos["ret"+str(topk)]=rettop
                infos["rets"+str(topk)]=retstop
                infos["retavg"+str(topk)]=rettopavg
                infos["retavgs"+str(topk)]=rettopavgs
                infos["tvr"+str(topk)]=tvrpos

        return infos
        
    def updatewm(self, cur_tidx):
        if gv["longterm"]>1:
            tools.setup_seed(cur_tidx)
            randmax=100000
            v=random.randint(0, randmax)/randmax 
            if (v > (1 / gv["longterm"])):
                # tools.flog("skip train:", cur_tidx)
                return
            
        tpf=tools.TimeProfilingLoop(self.name+"train")
        traindays=self.get_traindays(cur_tidx)
        frowardpf=tpf.add(self.name+"feedforward")
        for tday in traindays:
            output, vf, hidden= self.forward(tday, return_y=True)
            if vf.sum()<2:
                tools.flog("skip train valid zero:", gv["ts"][tday], gv["ts"][cur_tidx],
                           cur_tidx, vf.sum())
                frowardpf.end()
                continue
            y=self.getyy(tday)
            loss=self.calcloss(tday, output, y, vf)
    
            frowardpf.end()
            losssum=(loss["loss"])*gv["dynamiclr"]
            backwardpf=tpf.add(self.name+"backward")
            self.opt.zero_grad()
            losssum.backward()
            self.opt.step()
            backwardpf.end()

        tpf.end()
        timestr=tpf.to_string()
        tools.flog("tm:", cur_tidx, timestr, traindays, 
                   # "losssum:", losssum,
                   "std:", output.std(dim=0), 
             "allocated:", torch.cuda.memory_allocated(), "maxallocated:", torch.cuda.max_memory_allocated(), 
             "reserverd:", torch.cuda.memory_reserved())

