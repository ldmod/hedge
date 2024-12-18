#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 09:40:58 2023

@author: dli
"""
import torch
import torch.nn as nn

from cryptoqt.prm.alpha.tools import flog as flog
from cryptoqt.prm.alpha.tools import CssModel as CssModel
from cryptoqt.prm.alpha.tools import SeqCssModel as SeqCssModel
import cryptoqt.data.sec_klines.sec_klines as sk
import cryptoqt.prm.alpha.tools as tools
import cryptoqt.prm.alpha.seqfea as seqfea
import cryptoqt.prm.alpha.common as common
import cryptoqt.prm.alpha.mergemodel as mmodel
import cryptoqt.prm.alpha.yamlcfg as yamlcfg
from cryptoqt.prm.alpha.yamlcfg import gv
import numpy as np
from audtorch.metrics.functional import pearsonr
import random
import datetime
import time
import os
import io
import threading
import cryptoqt.prm.alpha.data2ins as d2i
import cryptoqt.data.constants as conts
from cryptoqt.prm.alpha.tools import *
import cryptoqt.data.sec_klines.sec_klines as sk
class PredM:
    def __init__(self):
        self.obj=None

    def pred(self, xhis,yhis,tshis, predcfg={"checkflag":False}):
        return 
            
    def pred(self, testtidx):
        tfp=tools.TimeProfilingLoop("pred")
        model=self.model
        x, vf, xetr= self.model.testday(testtidx)
        tfp.end()
        flog("test tm:", testtidx, "tmcose:",tfp.to_string(), time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        # torch.cuda.empty_cache()
        return x,vf, xetr 
    
    def tune(self, traintidx):
        tfp=tools.TimeProfilingLoop("tune")
        tools.setup_seed(traintidx)
        model=self.model
        self.model.updatewm(traintidx)
        
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
        resd=torch.load(prefix+".res")
        self.model.resdict = resd[0]
        off=1

        return modeldate
   
    
    def coldstart(self):
        self.gv=gv
        self.model=MLPmodel().cuda()    
    

class AttCssModel(nn.Module):
    def __init__(self, inputsize, outsize):
        super(AttCssModel, self).__init__()
        self.do=nn.Dropout(gv["dropout"])
        self.secm1 = CssModel(inputsize, outsize, bnlen=0)
        self.attm = tools.AttentionM(inputsize, outsize, bn=True)
    
    def forward(self, x):
        x1=self.secm1(x)
        x2=self.attm(x)
        x3=torch.cat([x1, x2], dim=-1)
        return x3

class SectionCss(nn.Module):
    def __init__(self, inputsize, crosssize, sectionsize=0, dropout=0.1):
        super(SectionCss, self).__init__()
        sectionsize = len(gv["sids"]) if sectionsize==0 else sectionsize
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
        self.css1 = CssModel(inputsize, cssize1, bnlen=0, dropout=gv["dropout"])
        self.css2 = CssModel(cssize1, cssize1, bnlen=0, dropout=gv["dropout"])
        self.css3 = CssModel(cssize1, cssize1, bnlen=0, dropout=gv["dropout"])
        self.css4 = CssModel(cssize1, 1)

    def forward(self, x):
        x=self.css1(x)
        x=self.css2(x)
        x=self.css3(x)
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

        crosssize=gv["s15corrsize"]
        self.s1m=seqfea.S1Model(crosssize=crosssize, mseqlen=8)
        self.s1embedlen=int(crosssize*(8+gv["seqlayer3"]*4))
        self.s5m=seqfea.S5Model(crosssize=crosssize, mseqlen=8)
        self.s5embedlen=int(crosssize*(8+gv["seqlayer3"]*4))
        self.sidembed=nn.Embedding(10000, gv["sidembedsize"])
        self.sidembed.weight.data.fill_(0.0)

        inputsize=gv["sidembedsize"]+gv["s1m"]*self.s1embedlen\
            +gv["s5m"]*self.s5embedlen
        self.do=nn.Dropout(gv["dropout"])
        cssize1=512
        cssize2=256
        self.css1 = CssModel(inputsize, cssize1, bnlen=0, dropout=gv["dropout"])
        self.css2 = CssModel(cssize1, cssize1, bnlen=0, dropout=gv["dropout"])
        # self.css3 = CssModel(cssize1, cssize2, bnlen=0, dropout=gv["dropout"])
        self.secms=nn.ModuleList()
        for key in gv["outvalue"]:
            mm = MmModel(cssize1, 1)
            self.secms.append(mm)
        
             
        ###################opt#############################
        self.opt=tools.AdamWReg(self.named_parameters(), lr=gv["learning_rate"]*gv["mergelr"],
                            betas=(gv["beta1"], gv["beta2"]), weight_decay1=0.0, weight_decay2=gv["weight_decay2"])
        self.oli=tools.OlInffer(self.name+"_test", calctop=True)
        self.olitrain=tools.OlInffer(self.name+"_train", calctop=True)
        self.saveresdict=True
        
        
    def pred(self, tidx):
        self.eval()
        x, vf, xetr = self.forward(tidx)
        self.train()
        return x.detach(), vf.detach(), xetr
    
    def forward(self, tidx, return_y=False):
        valid=d2i.getvalid(gv["insdict"].dd, tidx)
        sidembed=self.sidembed(torch.arange(valid.shape[0]).cuda().long())*(not gv["delmergesid"])
        oo=torch.zeros((valid.shape[0],len(gv["outvalue"]))).cuda()
            
        if (valid.isfinite() & valid).sum() < 2:
            tools.flog("valid sum:", valid.sum(), flush=True)
            return oo, valid, oo
        assert (valid.isfinite() & valid).sum()> 1, str(valid.sum())+"_"+str(gv["ts"][tidx])+"_"+str((valid.isfinite() & valid).sum())

        inputs=[sidembed[valid]]
        if gv["s5m"]:
            self.s5embed=self.s5m(tidx, valid)
            inputs+=self.s5embed
        if gv["s1m"]:
            self.s1embed=self.s1m(tidx, valid)
            inputs+=self.s1embed

        inp=torch.cat(inputs, dim=1)
        inp=self.css1(inp)
        inp=self.css2(inp)
        # inp=self.css3(inp)
        xetr={}
        xx=[]
        for idx in range(len(self.secms)):
            mm=self.secms[idx]
            tmpx=mm(inp)
            xx.append(tmpx)
        out=torch.cat(xx, dim=1)
        # out=self.do(out)
        
        oo[valid]=out
        return oo, valid, xetr
    
    def get_traindays(self, tidx):
        scnt=gv["scnt"]
        delay=gv["traindelay"]*scnt
        tdi=int(tidx/gv["tmdelta"]*scnt)
        days=list(range(max(int(tdi-365*1440/(gv["tmdelta"]/scnt)), 192), tdi-delay))
        self.sampleinsnum=gv["sampleinsnum"]
        tdays = random.sample(days, self.sampleinsnum)
        tdays=sorted(tdays)
        assigndays=gv["traindays"][:]+[delay]
        assigndays=tdi-np.array(assigndays)
        tdays+=assigndays.tolist()
        tdays=[int(item*(gv["tmdelta"]/scnt)-(gv["ltdelta"]-gv["tmdelta"])) for item in tdays]
        
        return tdays
    def ydays(self):
        days=gv["tmdelta"]
        return days 
    
    def rankfunc(self, y):
        y=(y*10).int()
        yv, yr=y.unique(sorted=True,return_inverse=True)
        yr=yr.float()
        yr=(yr-yr.mean())/yr.std().clamp(gv["eps"])
        return yr
            
    def calcloss(self, tidx, x, valid, xetr=None, calcstats=False):
        dd=gv["insdict"].dd
        ysvwap, yslow, ysl, vf_ys=d2i.getys(dd, tidx, gv["tmdelta"], gv["ltdelta"])
        ybvwap, ybhigh, ybl, vf_yb=d2i.getyb(dd, tidx, gv["tmdelta"], gv["ltdelta"])
        yvwap, yl, vf_y = d2i.getyvwap(dd, tidx, gv["tmdelta"], gv["ltdelta"])
        
        svf=valid&vf_ys
        bvf=valid&vf_yb
        ysvwap, yslow = ysvwap[svf], yslow[svf]
        xsvwap, xslow = x[:,0][svf], x[:,1][svf]
        ybvwap, ybhigh = ybvwap[bvf], ybhigh[bvf]
        xbvwap, xbhigh = x[:,2][bvf], x[:,3][bvf]
        xsl, xbl = x[:,4][svf], x[:,5][bvf]
        
        mse_ysvwap=nn.MSELoss(reduction='none')(xsvwap, ysvwap).mean(dim=0)
        mse_yslow=nn.MSELoss(reduction='none')(xslow, yslow).mean(dim=0)
        mse_ybvwap=nn.MSELoss(reduction='none')(xbvwap, ybvwap).mean(dim=0)
        mse_ybhigh=nn.MSELoss(reduction='none')(xbhigh, ybhigh).mean(dim=0)
        mse_ysl=nn.MSELoss(reduction='none')(xsl, ysl[svf]).mean(dim=0)
        mse_ybl=nn.MSELoss(reduction='none')(xbl, ybl[bvf]).mean(dim=0)

        loss=mse_ysvwap*gv["lossratio"][0]+mse_yslow*gv["lossratio"][1]+\
            mse_ybvwap*gv["lossratio"][2]+mse_ybhigh*gv["lossratio"][3]+\
                mse_ysl*gv["lossratio"][4]+mse_ybl*gv["lossratio"][5]

        vwapsd=(ysvwap-yvwap[svf]).mean()
        vwapbd=(ybvwap-yvwap[bvf]).mean()
        br, br2=(xsvwap>=yslow).sum()/svf.sum(), (xsvwap>=yslow).sum()/valid.sum()
        be, be2=-1.0*((xsvwap-ysvwap)*(xsvwap>=yslow)).sum()/svf.sum(), -1.0*((xsvwap-ysvwap)*(xsvwap>=yslow)).sum()/valid.sum()
        sr, sr2=(xbvwap<=ybhigh).sum()/bvf.sum(), (xbvwap<=ybhigh).sum()/valid.sum()
        se, se2=((xbvwap-ybvwap)*(xbvwap<=ybhigh)).sum()/bvf.sum(), ((xbvwap-ybvwap)*(xbvwap<=ybhigh)).sum()/valid.sum()
        lossres=dict(loss=loss, mse_ysvwap=mse_ysvwap, mse_yslow=mse_yslow,mse_ybvwap=mse_ybvwap,mse_ybhigh=mse_ybhigh,
                     mse_ysl=mse_ysl, mse_ybl=mse_ybl, 
                     valid=valid.sum(),svf=svf.sum(), bvf=bvf.sum(),
                     br=br, br2=br2, sr=sr, sr2=sr2,
                     be=be, be2=be2, se=se, se2=se2,
                     vwapsd=vwapsd, vwapbd=vwapbd,
                     )
        if calcstats:
            # statsmat=torch.full((gv["sids"].shape[0], len(gv["outvalue"])), torch.nan)
            diffsvwap=xsvwap-ysvwap
            # statsmat[svf,0]=diffsvwap.cpu()
            diffslow=xslow-yslow
            # statsmat[svf,1]=diffslow.cpu()
            diffbvwap=xbvwap-ybvwap
            # statsmat[bvf,2]=diffbvwap.cpu()
            diffbhigh=xbhigh-ybhigh
            # statsmat[bvf,3]=diffbvwap.cpu()
            
            icsvwap=tools.pearsonr(xsvwap, ysvwap)[0]
            icsslow=tools.pearsonr(xslow, yslow)[0]
            icbvwap=tools.pearsonr(xbvwap, ybvwap)[0]
            icsbhigh=tools.pearsonr(xbhigh, ybhigh)[0]
            icsl=tools.pearsonr(xsl, ysl[svf])[0]
            icbl=tools.pearsonr(xbl, ybl[bvf])[0]
            diffsl=xsl - ysl[svf]
            diffbl=xbl - ybl[bvf]
            
            tmpd=dict(diffsvwap=diffsvwap.abs().mean(), diffslow=diffslow.abs().mean(), 
                      diffbvwap=diffbvwap.abs().mean(), diffbhigh=diffbhigh.abs().mean(),
                      #
                      dsvwapmean=xsvwap.mean()-ysvwap.mean(), dslowmean=xslow.mean()-yslow.mean(), 
                      dbvwapmean=xbvwap.mean()-ybvwap.mean(), dbhighmean=xbhigh.mean()-ybhigh.mean(),
                      icsvwap=icsvwap, icsslow=icsslow, icbvwap=icbvwap, icsbhigh=icsbhigh,
                      icsl=icsl, icbl=icbl, 
                      dslmean=diffsl.abs().mean(), dblmean=diffbl.abs().mean(),
                      # statsmat=statsmat,
                      )
            lossres.update(tmpd)
            

        return lossres
    
        
    def updatewm(self, cur_tidx):
        if gv["dropins"]:
            tools.setup_seed(cur_tidx)
            randmax=100000
            v=random.randint(0, randmax)/randmax 
            if (v > (gv["tmdelta"] / gv["ltdelta"])):
                # tools.flog("skip train:", cur_tidx)
                return
        
        tpf=tools.TimeProfilingLoop(self.name+"train")
        traindays=self.get_traindays(cur_tidx)
        frowardpf=tpf.add(self.name+"feedforward")
        for tday in traindays:
            output, vf, xetr= self.forward(tday, return_y=True)
            if vf.sum()<2:
                tools.flog("skip train valid zero:", cur_tidx, vf.sum(), sk.gtm_i(cur_tidx), sk.gtm_i(tday))
                continue
            loss=self.calcloss(tday, output, vf)
    
            frowardpf.end()
            losssum=(loss["loss"])*gv["dynamiclr"]
            backwardpf=tpf.add(self.name+"backward")
            self.opt.zero_grad()
            losssum.backward()
            self.opt.step()
            backwardpf.end()

        tpf.end()
        timestr=tpf.to_string()
        tools.flog("train tm:", cur_tidx, timestr, traindays, "losssum:", losssum, "std:", output.std(dim=0), 
             "allocated:", torch.cuda.memory_allocated(), "maxallocated:", torch.cuda.max_memory_allocated(), 
             "reserverd:", torch.cuda.memory_reserved())



