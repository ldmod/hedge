import os
import sys
import h5py
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import matplotlib.pyplot as plt
from alpha import catransinit as cinit
import torch
from audtorch.metrics.functional import pearsonr
from scipy import stats
from alpha import cononrawdataday as rawdataday
from alpha import catransinit as cinit
from alpha import cononrawdataday as rawdataday
from alpha import seqfea as seqfea
from alpha import querydata as qd
from datacode import jqdownloader as jqdl
import datetime
from datetime import datetime
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 150)

class ReadRes():

    def __init__(self, cfg):
        self.dr=cfg["dr"]
        self.yamlpath=cfg.get("yamlcfgpath", "")
        cinit.gv["lasteddownload"]=cfg.get("lasteddownload")
        cinit.gv["basedates"]=self.dr["alldates"]
        cinit.gv["rdateoffset"]=0
        cinit.gv["baseuids"]=self.dr["sids"]
        cinit.initcfg(self.yamlpath)
        cinit.gv["realtime"]=int(cfg.get("realtime", 0))
        cinit.gv["savedelta"]=int(cfg.get("savedelta", 32))
        cinit.gv["data_dict"]=self.dr
        self.targettidx=int(cfg.get("targettidx", 44))
        cinit.pconfig()
        # self.modelpath=cfg.get("modelpath", None)
        lastedmodelpath=None
        if os.path.exists(cinit.gv["modelpath"]):
            ms=os.listdir(cinit.gv["modelpath"])
            modelpaths=[]
            for path in ms:
                if "_date_" in path:
                    modelpaths.append((path, path.split("_date_")[1]))
            if len(modelpaths) > 0:
                lastedmodelpath=cinit.gv["modelpath"]+sorted(modelpaths, key=lambda x:x[1])[-1][0]  
        # initdatalen=5
        # cinit.initdata(self.dr, initdatalen, self.mininsstart+initdatalen)
        # enddi=(self.startdi-1)*cinit.gv["insperday"]+self.targettidx+1
        self.pred=cinit.initmodel(self.dr, lastedmodelpath)
        
        stats=[]
        dates=[]
        baseret=[0]
        queryd=qd.QueryData(self.dr, cinit.gv["basedates"], cinit.gv["rdateoffset"])
        cinit.gv["queryd"]=queryd
        self.selectq=10
        for di in range(2000, cinit.gv["lasteddownload"]-2):
            df=self.parseperf(di)
            ret=df["ret"][df["argx"][-20:]].mean()
            date=cinit.gv["basedates"][di]
            dates.append(date)
            stats.append(ret)
            baseret.append(baseret[-1]+ret)
            sortdf=df.iloc[df["argx"][::-1]]
            sortdf=sortdf.set_index("uid").join(cinit.gv["data_dict"]["stocksinfo"])
            sortdf.to_csv(cinit.gv["modelpath"]+"./res/"+cinit.gv["basedates"][di]+".csv")
            
            
        plt.plot(dates, baseret[1:])
    
    def parseperf(self, di):
        queryd=cinit.gv["queryd"]
        pred, argx, y=self.getpredandres(di)
        df0=queryd.getdata(di)
        df=queryd.getdata(di+1)

        df["close0"]=df0["close"]
        df["ret"]=df["close"]/df0["close"]-1.0
        df["pred"]=pred
        if y is not None:
            df["y"]=y
        df["argx"]=argx
        df["date"].iloc[:]=cinit.gv["basedates"][di]
        return df
            
    def modelpath_parser(self, path):
        prefix=path.split('_date_')[0]
        modeldate=path.split('_date_')[1]
        modelday=modeldate.split('T')[0]
        modelmin=modeldate.split('T')[1].replace("-", ":")
        return prefix, modelday, modelmin
    def getpredandres(self, di):
        date=str(cinit.gv["basedates"][di])
        timekey=date+"T"+rawdataday.times[self.targettidx]
        res=self.pred.model.resdict[timekey]
        x,vf,y=res[0].numpy(), res[1].numpy(), res[2].numpy() if len(res)>2 else None
        x=x[:, self.selectq].flatten().copy()
        y=y[:, self.selectq].flatten().copy() if y is not None else None
        x[~vf]=0
        x[~np.isfinite(x)]=0
        argx=np.argsort(x)
        return x, argx, y
    def getptopred(self, alpha, vf):
        x=alpha.flatten().copy()
        x[~vf]=0
        x[~np.isfinite(x)]=0
        argx=np.argsort(x)
        argx=argx[::-1]
        df=pd.DataFrame({})
        df["uid"]=cinit.gv["baseuids"][argx]
        df["pred_score"]=x[argx]
        return df
def getwratiotop(output,vf, hlnum=1000):
    eps=0.000001
    ratio=output.clone().detach()
    ratio=ratio.reshape(vf.shape[0],-1)
    ratio=ratio/ratio.std(dim=0).clamp(eps)
    ratio[~vf,:]=-10000.0
    offnum=min(hlnum, ratio.shape[0])
    minw=ratio.topk(offnum, dim=0).values[-1]
    ratio=ratio-minw
    ratio[ratio<0]=0
    ratio=ratio/ratio.sum(dim=0).clamp(eps)
    return ratio
def readres(path="/home/prod/newversion/tnow/dli/model_states/infmodel/tnowtest/res"):
    fns=os.listdir(path)
    fns=sorted(fns)
    predfns=[item for item in fns if item[:4]=="pred" ]
    statfns=[item for item in fns if item[:4]!="pred" ]
    statdf=pd.DataFrame()
    for fname in statfns:
        df=pd.read_csv(path+"/"+fname)
        date=df["date"].iloc[0]
        w=df["pred"].iloc[:20]/df["pred"].iloc[:20].sum()
        w50=df["pred"].iloc[:50]/df["pred"].iloc[:50].sum()
        w100=df["pred"].iloc[:100]/df["pred"].iloc[:100].sum()
        retc2c=df["close"]/df["close0"]-1.0
        ret10=(retc2c.iloc[:10]).sum()/10
        ret20=(retc2c.iloc[:20]).sum()/20
        ret50=(retc2c.iloc[:50]).sum()/50
        ratio=getwratiotop(torch.from_numpy(df["pred"].to_numpy().reshape(-1,1)), torch.ones(df["pred"].shape[0]).bool(), 20)
        retsim20=(ratio.numpy().flatten()*(df["close"]/df["close0"]-1.0)).sum()
        retw10=(w*retc2c.iloc[:10]).sum()
        retw20=(w*retc2c.iloc[:20]).sum()
        retw50=(w50*retc2c.iloc[:50]).sum()
        retw100=(w100*retc2c.iloc[:100]).sum()
        # y20=(w*df["y"].iloc[:20]).sum()

        statdf=statdf._append(dict(date=date, ret10=ret10, ret20=ret20, ret50=ret50,retsim20=retsim20,
                                   retw10=retw10,
                                   retw20=retw20, retw50=retw50, retw100=retw100), ignore_index=True)
        
    plt.plot([datetime.strptime(d, '%Y-%m-%d').date() for d in statdf["date"]], 
             np.cumsum(statdf["ret10"]), 'o-',label='pnl')
    plt.tick_params(axis='both',which='both',labelsize=10)
    plt.gcf().autofmt_xdate()
    plt.show()
    return
        
    
if __name__ == "__main__":
    did="0"
    if len(sys.argv)>1:
        did=sys.argv[1]
    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2"
    # readres()

    # lasteddownload=jqdl.loadalldict(filddataflag=False)
    lasteddownload=jqdl.readh5new()
    jqdl.filterdata()
    dr=jqdl.g_data
    dr["uid"]=dr["sids"]

    cfg=dict(universeId="TopAllD1", dr=dr, online="0", 
             # modelpath=modelpath,
             m4m="1", pricetarget="tick5min_intra_adjhigh",
             seqatt="0", sd="2014-11-18", misd="2011-01-04", delay5min="3",
             targettidx="44",savedelta="1", 
             realtime="1", step="48", traindeltanum="6", learning_rate="0.0003", tvrdeltas="1",
             yamlcfgpath="../econf/cfg.yaml",
             lasteddownload=lasteddownload
             )
    conon=ReadRes(cfg)
