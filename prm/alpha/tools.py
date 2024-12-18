# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 21:15:11 2021

@author: ld
"""
"""
??¨è????￥?￥????????????¨GPU??????è?-???
"""
# Import libraries
import time
import datetime
import torch
import torch.nn as nn # ??-??o?￥??????????
from torch.optim import SGD # ????????¨SGD
import torch.utils.data as Data # ??°???é￠??¤????
#from audtorch.metrics.functional import pearsonr
import math
import pandas as pd
import numpy as np
from scipy import stats
import numba
from numba import jit
import matplotlib.pyplot as plt
import os
from functools import partial
import sys
import copy
from threading import Thread, Lock
import einops
from scipy.stats import rankdata
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from functools import reduce
from torch.autograd import Function
from torch.autograd import gradcheck
from typing import List, Optional
from scipy import spatial
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
import pandas as pd
#from functorch import vmap
from functools import reduce
import importlib
import importlib.util
from cryptoqt.prm.alpha.yamlcfg import gv
import threading
import shutil
import gc
import cryptoqt.data.sec_klines.sec_klines as sk

class TimeProfiling(object):
    def __init__(self, name):
        self._s=datetime.datetime.now()
        self._subt={}
        self._name=name
    def end(self):
        self._e=datetime.datetime.now()
    def add(self, name):
        assert not name in self._subt
        self._subt[name]=TimeProfiling(name)
        return self._subt[name]
    def to_string(self):
        # assert (self._e != None) << self._name+"timeend"
        s='{'+self._name+':'+str((self._e-self._s).seconds*1000.0+(self._e-self._s).microseconds/1000.0)
        for k,v in self._subt.items():
            s+=v.to_string()+'--'
        s+='}'
        return s
    
class TimeProfilingLoop(object):
    def __init__(self, name):
        self.deltas=[]
        self._s=datetime.datetime.now()
        self._subt={}
        self._name=name
    def end(self):
        self._e=datetime.datetime.now()
        delta=(self._e-self._s).seconds*1000.0+(self._e-self._s).microseconds/1000.0
        self.deltas.append(delta)
    def restart(self):
        self._s=datetime.datetime.now()
    def add(self, name):
        if not name in self._subt:
            self._subt[name]=TimeProfilingLoop(name)  
        else:
            self._subt[name].restart()
        return self._subt[name]
    def to_string(self):
        # assert (self._e != None) << self._name+"timeend"
        deltastr=str(sum(self.deltas))+":"+str(sum(self.deltas)/len(self.deltas))+":"+str(len(self.deltas))
        s='{'+self._name+':'+deltastr
        for k,v in self._subt.items():
            s+=v.to_string()+'--'
        s+='}'
        return s
    
def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # 
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def ranknorm(xx):
    x=xx.copy().astype(float)
    nz=(x!=0) & np.isfinite(x)
    a=x[nz].copy()
    a=rankdata(a)
    a-=np.mean(a)
    if a.std()!=0:
        a/=np.std(a)
    x[nz]=a
    return x
def ranknorm1(xx):
    x=xx.copy().astype(float)
    nz=(x!=0)& np.isfinite(x)
    if nz.sum() <1:
        return x
    a=x[nz].copy()
    a=rankdata(a)
    a-=np.mean(a)
    if a.std()!=0:
        a/=np.std(a)
        a=(a-a.min())/(a.max()-a.min())
    x[nz]=a
    return x

def appenddict(a, b):
    for key in b.keys():
        if not key in a:
            a[key]=[]
        a[key]+=b[key]
    return a
        
def clampgrad(a, minv, maxv):
    b=torch.zeros(a.shape).cuda()
    b[a>maxv]=a[a>maxv]-(a[a>maxv].detach()-maxv)
    b[a<minv]=a[a<minv]-(a[a<minv].detach()-minv)
    b[(a<=maxv)&(a>=minv)]=a[(a<=maxv)&(a>=minv)]
    return b
def normclamp(a, maxv):
    a=a.flatten()
    l=a.shape[0]
    res=[0 for x in range(l)]
    arg=a.sort().indices
    argrank=arg.sort().indices
    maskv=torch.arange(arg.shape[0]).cuda().reshape(-1,1)
    w=1.0
    ra=a.reshape(1,-1).repeat([a.shape[0],1])
    rarg=argrank.reshape(1,-1).repeat([a.shape[0],1])
    mask=rarg<=maskv
    ss=torch.sum(ra*mask,dim=1)+0.000001
    rv=a[arg]
    rw=rv/ss
    rwc=rw.flip(dims=[0])
    rwcd=rwc.detach()
    maxratio=0.02
    maxv=[maxratio/max((1-i*maxratio),maxratio) for i in range(a.shape[0])]
    maxv=torch.tensor(maxv).cuda()
    rwclamp=clampgrad(rwc, 0.0, maxratio)
    rwclampd=rwclamp.detach()
    wt=torch.zeros(a.shape[0]).cuda()
    for i in range(l):
        wt[i]=w
        w-=rwclampd[i]*w
    res=rwclamp*wt

    return res


def moving_average(x, w):
    if torch.is_tensor(x):
        x=x.cpu().detach().numpy()
    return np.convolve(x, np.ones(w), 'same') / w

def mova(x, w):
    if torch.is_tensor(x):
        x=x.cpu().detach().numpy()
    return np.convolve(x, np.ones(w), 'same') / w
def nt(a):
    return a.cpu().detach().numpy().flatten()
def pltg(a, b=20,range=None,clip=False):
    if clip:
        range=(np.percentile(a,1),np.percentile(a,99))
    if torch.is_tensor(a):
        return plt.hist(a.flatten().cpu().detach().numpy(),bins=b,range=range)
    else:
        return plt.hist(a.flatten(),bins=b,range=range)
def plotg(a, clip=False):
    if clip:
        range=(np.percentile(a,1),np.percentile(a,99))
    if torch.is_tensor(a):
        return plt.plot(a.flatten().cpu().detach().numpy())
    else:
        return plt.plot(a.flatten())
def isfinite(a):
    if torch.is_tensor(a):
        return a[torch.isfinite(a)]
    else:
        return a[np.isfinite(a)]

def pgp(a, b=50,name=""):
    if torch.is_tensor(a):
        a=a.flatten().cpu().detach().numpy()
    pavg=[np.percentile(a, i) for i in range(b)]
    plt.title("percentile:"+name)
    plt.plot(pavg[1:])
    plt.show()
    return np.array(pavg)
def readw(m):
    a=list(m.parameters())
    w=[]
    wg=[]
    for i in range(len(a)):
        print(i, a[i].shape)
        w+=[a[i].flatten()]
        # w+=[a[i]]
        if a[i].grad != None:
            wg+=[a[i].grad.flatten()]
    return w,wg
def readwflatten(m):
    a=list(m.parameters())
    w=[]
    wg=[]
    for i in range(len(a)):
        print(i, a[i].shape)
        w+=[a[i].flatten()]
        # w+=[a[i]]
        if a[i].grad != None:
            wg+=[a[i].grad.flatten()]
    return w,wg
def argdiff(outv,argd):
    oarg=outv.argsort()
    ll=outv.shape[0]
    dd=20
    parg=[]
    for i in range(20):
        parg.append(argd[oarg[int(i*ll/dd):int(i*ll/dd+ll/dd)]].mean().item())
    return np.array(parg)
    # plt.plot(list(range(dd)), parg)

def calc_loss(a,b,loss_function=nn.MSELoss()):
    return loss_function(torch.from_numpy(a), torch.from_numpy(b))**0.5
def calc_ret(a,b,num):
    arg=a.argsort()
    a=a[arg[-num:]]
    b=b[arg[-num:]]
    a=a/a.sum()
    return (a*b).sum()

class rloss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return -1.0*x*y
class smloss(nn.Module):
    def __init__(self, b1,b2):
        super().__init__()
        self._b1=b1
        self._b2=b2

    def forward(self, x, y):
        flag1=((x-y)>self._b1)&((x-y)<0.0)
        flag2=((x-y)<self._b2)&((x-y)>0.0)
        loss=torch.abs(x-y)
        loss[flag1]=-0.5*(x-y)*(x-y)/self._b1
        loss[flag2]=0.5*(x-y)*(x-y)/self._b2
        return -1.0*x*y

class RegSGD(torch.optim.Optimizer):
    def __init__(self, named_params, lr=0.01, betas=(0.9,0.99), eps=1e-8, theta=5e-4,                 weight_decay1=0, weight_decay2=0):
        params=[]
        names=[]
        for name,param in named_params:
            names.append(name)
            params.append(param)
        theta=lr*0.2
        defaults = dict(lr=lr, names=names, betas=betas,eps=eps,
                        weight_decay1=weight_decay1,theta=theta, weight_decay2=weight_decay2)

        super(RegSGD, self).__init__(iter(params), defaults)

    def __setstate__(self, state):
        super(RegSGD, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step. Arguments: closure (callable, optional): A closure that reevaluates the model and returns the loss. """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay1 = group['weight_decay1']
            weight_decay2 = group['weight_decay2']
            eps=group['eps']
            beta1,beta2=group['betas']
            lr=group['lr']
            theta=group['theta']
            names=group['names']
            params=group['params']

            for idx,p in enumerate(params):
                name=names[idx]
                # p=params[idx]
                # if p.grad is None or (not "weight" in name):
                if p.grad is None :
                    continue
                param_state = self.state[p]
                if len(param_state) == 0:
                    param_state['step'] = 0
                    param_state['exp_avg'] = torch.zeros_like(p.data)
                    param_state['exp_avg_sq'] = torch.zeros_like(p.data)
                    param_state['exp_var'] = torch.zeros_like(p.data)

                step=param_state['step']
                step+=1

                p.data.mul_(1 - lr * weight_decay2)
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                step_size = lr / bias_correction1
                bias_correction2_sqrt = math.sqrt(bias_correction2)

                grad = p.grad.data
                if beta1 != 0:
                    exp_avg = param_state['exp_avg']
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)


                    exp_avg_sq = param_state['exp_avg_sq']
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                    exp_var = param_state['exp_var']
                    exp_d=grad-exp_avg
                    exp_var.mul_(beta2).addcmul_(exp_d, exp_d, value=1.0 - beta2)

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                if weight_decay1 != 0:
                    mask1=(p.data >= 0) & (p.data < theta)
                    mask2=(p.data < 0) & (p.data > -theta)
                    p.data[mask1]=torch.clamp(p.data[mask1]-lr*weight_decay1,-10000,0)
                    p.data[mask2]=torch.clamp(p.data[mask2]+lr*weight_decay1,0,100000)


                # if weight_decay1 != 0:
                #     p.data.add_(-lr*weight_decay1, torch.sign(p.data))
                # if weight_decay2 != 0:
                #     p.data.add_(-lr*weight_decay2, p.data)

        return loss
 
def convw(x, weight):
    return F.conv1d(x.unsqueeze(0).unsqueeze(0), weight.reshape(1,1,-1).cuda())

def tensormem(x):
    memused=x.element_size()*x.nelement()/1024/1024/1024
    return memused

class Net(torch.nn.Module):
    def __init__(self,lag_list):
        super(Net,self).__init__()
        
        self.lag_list=lag_list
        #??????é??
 
        self.l=len(self.lag_list)
        #??????é???¤§?°?
 
        self.param=torch.nn.Parameter(torch.zeros(self.l,2))
        #è?a???????3???°????????atime lag ???è??è?a???????3???°
        #è?a???????3???°?????￥Parameteré??é?￠???é?￡?1??1????é??????￥???????è·ˉ?￠ˉ?o|???é????′??°
        
    #????1????????????-??1???
    def forward(self,x):
        max_len=max(self.lag_list)
        #??????é?????????¤§????′???????è????a????′??1????????????a????′?????§????è??è??è?a??????é￠??μ???????èo?????·?è·?
 
        l=x.shape[0]
        if(type(l)==type(1)):
            pass
        else:
            l=int(l)
        #è??é????o????1?è|?è???1?????±?????????￠?????????è???1???ˉ???????????a???é?????
        '''
        ??′??￥??¨pytorch???èˉ????l??ˉ?????aint?±?????????°?-?
        ?????ˉ?|?????????¨?o?tensorboardX?1???????l??ˉ?????aTensor???é?￡?1??°±?????¨?±????è????￠???int
        '''
 
        tensor_len=l-max_len
        #è?¨?¤oé??è|?è??è?????è?a??????é￠??μ?-??°??????èo?????ˉ?è?????é???o|
        #?1??°±??ˉmax_len?????????????§???°?????????è?????é?¨??????é???o|
        
        lst_lag=[]
        #????????????é???ˉ1?o????è????￥??°??????é?¨???
        for i in self.lag_list:
            lst_lag.append(x[max_len-i:max_len-i+tensor_len].clone())
        #è??é????o????1?è|?é??è??è???§?clone+?????￥???è?¨?????1?????￠???
        #?????o?|?????????′??￥??¨????§???°??????è??è??????????????????ˉ??|?x[i]=x[i-1]+x[i-2]???è???§?
        #???????????o?????????inplace???é??é￠????è??????3????é???￠ˉ?o|??????????????￥é??
 
        ret_tmp_origin=x[max_len:max_len+tensor_len].clone()
        #????§???°???é??è|??ˉ?è????????é?¨???
        
        ret_var=self.param[0]*lst_lag[0]
        for i in range(1,self.l):
            ret_var=ret_var+self.param[i]*lst_lag[i]
        #è?a??????é￠??μ????é?¨???
 
        return(ret_var-ret_tmp_origin)
        #è?a??????é￠??μ????é?¨??????????§?é?¨???????·?è·?
        
def active_bytes():
    stats = torch.cuda.memory_stats()
    current_active_byte =  stats['active_bytes.all.current']
    return current_active_byte

def tensornorm(x, eps=1e-8, dim=0):
    mean=x.nanmean(dim=dim).unsqueeze(dim=dim)
    std=(((x-mean)**2).nanmean(dim=dim)).unsqueeze(dim=dim)**0.5
    x=(x-mean)/std.clamp(eps)
    return x

def splitmean(x):
    halflen=int(x.shape[0]/2)
    mean1=x[:halflen].reshape(-1,x.shape[-1]).nanmean(dim=0)
    mean2=x[halflen:].reshape(-1,x.shape[-1]).nanmean(dim=0)
    mean=(mean1+mean2)/2
    return mean


def cosvalue(outv,tyv, batch_first=False):
    if torch.is_tensor(outv):
        cosvalue=gv["cosloss_function"](outv.detach(),tyv.detach()).item()
    else:
        cosvalue=outv.dot(tyv) / (np.linalg.norm(outv) * np.linalg.norm(tyv))
    return cosvalue

def transposex(x):
    dims=list(range(len(x.shape)))
    xdims=dims[:-2]+[dims[-1],dims[-2]]
    return x.permute(xdims)
class HalfLinearFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        output = input.half().matmul(transposex(weight).half()) 
        if bias is not None:
            output += bias.half().unsqueeze(0).expand_as(output) 
        return output.float()
 
    @staticmethod
    def backward(ctx, grad_output): 
        # print("grad_output:", grad_output.shape, grad_output.dtype, grad_output.std())
        scale=grad_output.abs().mean()
        # print("scale:", scale.shape, scale)
        grad_output=(grad_output/scale).half()
        input, weight, bias = ctx.saved_tensors
        # print("ctx:", input.shape, weight.shape, bias.shape)
        input, weight, bias = input.half(), weight.half(), bias.half()
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.matmul(weight) 
            grad_input=grad_input.float()*scale
        if ctx.needs_input_grad[1]:
            grad_weight = transposex(grad_output).matmul(input) 
            grad_weight=grad_weight.float()*scale
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)
            grad_bias=grad_bias.float()*scale
        # print("std:", input.abs().mean(), weight.abs().mean(), bias.abs().mean(),               # (grad_input/scale).abs().mean(), (grad_weight/scale).abs().mean(), (grad_bias/scale).std())
        # print("ctx grad:", grad_input.shape, grad_weight.shape, grad_bias.shape)
        return grad_input, grad_weight, grad_bias

class HalfLinearModel(nn.Module):
    def __init__(self, inputsize, outsize, bnlen=0,dropout=0.0):
        super(HalfLinearModel, self).__init__()
        self.cssfunc=HalfLinearFunction.apply
        setup_seed(0)  
        self.hidden1 = nn.Linear(in_features=inputsize, out_features=outsize, bias=True)

    def forward(self, x):
        cross1=self.cssfunc(x, self.hidden1.weight, self.hidden1.bias)
        return cross1
def cssbackwardhock(m, grad_input, grad_output):
    print("backward m:",m)
    print("grad_input:", grad_input)
    print("grad_output:", grad_output)
    for item in grad_input:
        print("item:", item.dtype, item.shape)
        print(item)
        # item*=0.1
def cssbforwardprehock(m, input):
    print("preforward m:",m)
    print("input:", input)
    for item in input:
        print("item:", item.dtype, item.shape)
        print(item)
        # item*=0.1

class HalfCssFunction(Function):
    @staticmethod
    def forward(ctx, x, w1, bias1, w2, bias2):
        x, w1, bias1, w2, bias2 = x.half(), w1.half(), bias1.half(), w2.half(), bias2.half()
        h1=x.matmul(w1.t())+bias1.unsqueeze(0).unsqueeze(0).expand(x.shape[0], x.shape[1], -1) 
        h1c=x.matmul(w2.t())+bias2.unsqueeze(0).unsqueeze(0).expand(x.shape[0], x.shape[1], -1) 
        h1cs=h1c.sigmoid()
        ctx.save_for_backward(x, w1, bias1, w2, bias2, h1, h1c, h1cs)
        out=h1*h1cs
        return out.float()
    @staticmethod
    def backward(ctx, grad_output): 
        scale=grad_output.std()
        grad_output=(grad_output/scale).half()
        x, w1, bias1, w2, bias2, h1, h1c, h1cs = ctx.saved_tensors
        # h1g=grad_output*h1cs
        # h1cg=grad_output*h1*(1-h1cs)*h1cs
        
        grad_input = grad_weight1 = grad_bias1 = grad_weight2 = grad_bias2 = None
        # print(ctx.needs_input_grad, file=gv["stdoutbak"])
        if ctx.needs_input_grad[0]:
            grad_input = (grad_output*h1cs).matmul(w1) 
            grad_input=grad_input.float()*scale
        if ctx.needs_input_grad[1]:
            grad_weight1 = (grad_output*h1cs).permute(0,2,1).matmul(x) 
            grad_weight1=grad_weight1.float()*scale
        if ctx.needs_input_grad[2]:
            grad_bias1 = (grad_output*h1cs).sum(0).squeeze(0) 
            grad_bias1=grad_bias1.float()*scale
            
        if ctx.needs_input_grad[0]:
            grad_input += (grad_output*h1*(1-h1cs)*h1cs).matmul(w2) 
            grad_input=grad_input.float()*scale
        if ctx.needs_input_grad[3]:
            grad_weight2 = (grad_output*h1*(1-h1cs)*h1cs).permute(0,2,1).matmul(x) 
            grad_weight2=grad_weight2.float()*scale
        if ctx.needs_input_grad[4]:
            grad_bias2 = (grad_output*h1*(1-h1cs)*h1cs).sum(0).squeeze(0) 
            grad_bias2=grad_bias2.float()*scale
        return grad_input, grad_weight1, grad_bias1, grad_weight2, grad_bias2

class HalfCssModel(nn.Module):
    def __init__(self, inputsize, outsize, bnlen=0,dropout=0.0):
        super(HalfCssModel, self).__init__()
        self.cssfunc=HalfCssFunction.apply
        self.hidden1 = HalfLinearModel(inputsize, outsize)
        self.hidden1c = HalfLinearModel(inputsize, outsize)
        self.bnlen=bnlen*outsize
        self.bnlayer=nn.BatchNorm1d(self.bnlen, track_running_stats=False)
        self.do=nn.Dropout(dropout)
        self.dropout=dropout

    def forward(self, x):
        h1=self.hidden1(x)
        h1c=self.hidden1(x).sigmoid()
        cross1=h1*h1c
        if self.bnlen > 0:
            cross1=self.bnlayer(cross1.reshape(-1,self.bnlen)).reshape(cross1.shape)
        if self.dropout > 0:
            cross1 = self.do(cross1)
        return cross1

class BnFunction(Function):
    @staticmethod
    def forward(ctx, gamma, beta, x):
        assert len(x.shape)==2, str(x.shape)
        mean=x.mean(dim=0)
        std=x.std(dim=0)
        eps=(1e-5)
        var_sqrt_i = 1/(std*std+eps)**0.5
        out=(x-mean).mul_(var_sqrt_i*gamma).add_(beta)
        ctx.cache=(x, mean, var_sqrt_i, gamma)
        return out
    @staticmethod
    def backward(ctx, grad_output): 
        dout=grad_output
        N, D = dout.shape
        x, mean, var_sqrt_i, gamma = ctx.cache
        xstd=(x-mean).mul_(var_sqrt_i)
        dbeta = torch.sum(dout, axis=0)
        dgamma = torch.sum(xstd*dout, axis=0)
        dxstd = dout * gamma
        dx = 1.0/N * var_sqrt_i * (N*dxstd - torch.sum(dxstd, axis=0) - xstd*torch.sum(dxstd*xstd, axis=0))
        del xstd, ctx.cache
        return dgamma, dbeta, dx

class OptBnModel(nn.Module):
    def __init__(self, inputsize):
        super(OptBnModel, self).__init__()
        self.bnfunc=BnFunction.apply
        self.bnlayer=nn.BatchNorm1d(inputsize, track_running_stats=False)

    def forward(self, x):
        out=self.bnfunc(self.bnlayer.weight, self.bnlayer.bias, x)
        return out

class OptLinearFunction(Function):
    @staticmethod
    def forward(ctx, w1, bias1,  *x):
        with torch.no_grad():
            if len(x)>1:
                catx=torch.cat(x, dim=-1)
            else:
                catx=x[0]

            w1reshape=w1.t()
            if len(catx.shape) > len(w1reshape.shape):
                expandshape=catx.shape[:len(catx.shape) - len(w1reshape.shape)]+w1reshape.shape
                w1reshape=w1reshape.unsqueeze(0).expand(expandshape)
                h1=catx.matmul(w1reshape).add_(bias1)
            else:
                h1=catx.mm(w1reshape).add_(bias1)
    
            ctx.savexlen=len(x)
            ctx.savevalues=x+(w1, bias1)
            # ctx.save_for_backward(*x, w1, bias1, w2, bias2, yp, h1cs, gamma, beta, ypmean, ypstd)
            return h1
    @staticmethod
    def backward(ctx, grad_output): 
        # import pdb; pdb.set_trace()
        with torch.no_grad():
            xlen=ctx.savexlen
            x=ctx.savevalues[:xlen]
            w1, bias1 = ctx.savevalues[xlen:]
            if len(x)>1:
                catx=torch.cat(x, dim=-1)
            else:
                catx=x[0]
            del ctx.savevalues
            grad_input = grad_weight1 = grad_bias1 = None
            grad_h1=grad_output

            if len(grad_h1.shape) > len(w1.shape):
                expandshape=grad_h1.shape[:len(grad_h1.shape) - len(w1.shape)]+w1.shape
                w1reshape=w1.unsqueeze(0).expand(expandshape)
                grad_input = (grad_h1).matmul(w1reshape)
            else:
                w1reshape=w1
                grad_input = (grad_h1).matmul(w1reshape)

            if w1.requires_grad:
                grad_weight1 = (grad_h1).transpose(-2,-1).matmul(catx) 
                if len(grad_h1.shape)-len(w1.shape)>0:
                    grad_weight1=grad_weight1.sum(dim=list(range(len(grad_h1.shape)-len(w1.shape))))
            if bias1.requires_grad:
                if len(grad_h1.shape)-len(bias1.shape) > 0:
                    grad_bias1 = (grad_h1).sum(dim=list(range(len(grad_h1.shape)-len(bias1.shape))))
                grad_bias1=grad_bias1.float()

            
            splitsize=[item.shape[-1] for item in x]
            grad_inputs=torch.split(grad_input, splitsize, dim=-1)
    
            del grad_input
            # gc.collect()
            return grad_weight1, grad_bias1, *grad_inputs
    
class OptLinearModel(nn.Module):
    def __init__(self, inputsize, outsize):
        super(OptLinearModel, self).__init__()
        self.hidden1 = nn.Linear(in_features=inputsize, out_features=outsize, bias=True)
        self.linearfunc=OptLinearFunction.apply

    def forward(self, x):
        if isinstance(x, list):
            cross1=self.linearfunc(self.hidden1.weight, self.hidden1.bias, *tuple(x))
        else:
            cross1=self.linearfunc(self.hidden1.weight, self.hidden1.bias, x)
        return cross1
    
    
class CssFunction(Function):
    @staticmethod
    def forward(ctx, w1, bias1, w2, bias2, gamma, beta, *x):
        with torch.no_grad():
            if len(x)>1 or (not gamma is None):
                catx=torch.cat(x, dim=-1)
            else:
                catx=x[0]

            if not gamma is None:
                oldshape=catx.shape
                catx=catx.reshape(-1, gamma.shape[0])
                mean=catx.mean(dim=0)
                std=catx.std(dim=0)
                eps=(1e-8)
                var_sqrt_i = 1/((std*std+eps)**0.5)
                catx=catx.add_(mean, alpha=-1.0).mul_(var_sqrt_i).mul_(gamma).add_(beta)
                catx=catx.reshape(oldshape)
                del mean, std, var_sqrt_i

            w1reshape=w1.t()
            if len(catx.shape) > len(w1reshape.shape):
                expandshape=catx.shape[:len(catx.shape) - len(w1reshape.shape)]+w1reshape.shape
                w1reshape=w1reshape.unsqueeze(0).expand(expandshape)
                h1=catx.matmul(w1reshape).add_(bias1)

            else:
                h1=catx.mm(w1reshape).add_(bias1)
            w2reshape=w2.t()
            if len(catx.shape) > len(w2reshape.shape):
                expandshape=catx.shape[:len(catx.shape) - len(w2reshape.shape)]+w2reshape.shape
                w2reshape=w2reshape.unsqueeze(0).expand(expandshape)
                h1c=catx.matmul(w2reshape).add_(bias2)
            else:
                h1c=catx.mm(w2reshape).add_(bias2)

            h1cs=h1c.sigmoid_()
            eps=1e-12
            h1[(h1>=0)&(h1<eps)]=eps
            h1[(h1<0)&(h1>-eps)]=-eps
            h1cs[(h1cs>=0)&(h1cs<eps)]=eps
            h1cs[(h1cs<0)&(h1cs>-eps)]=-eps
            yp=h1*h1cs
            out=yp
            ypmean, ypstd = None, None

    
            ctx.savexlen=len(x)
            ctx.savevalues=x+(w1, bias1, w2, bias2, yp,h1cs, gamma, beta, ypmean, ypstd)
            # ctx.save_for_backward(*x, w1, bias1, w2, bias2, yp, h1cs, gamma, beta, ypmean, ypstd)
            del h1, catx
            return out
    @staticmethod
    def backward(ctx, grad_output): 
        # import pdb; pdb.set_trace()
        with torch.no_grad():
            xlen=ctx.savexlen
            x=ctx.savevalues[:xlen]
            w1, bias1, w2, bias2, yp, h1cs, gamma, beta, ypmean, ypstd = ctx.savevalues[xlen:]
            
            if len(x)>1 or (not gamma is None):
                catx=torch.cat(x, dim=-1)
            else:
                catx=x[0]
            inputneedgrad=True
            del ctx.savevalues
            if not gamma is None:
                oldshape=catx.shape
                catx=catx.reshape(-1, gamma.shape[0])
                mean=catx.mean(dim=0)
                std=catx.std(dim=0)
                eps=(1e-8)
                var_sqrt_i = 1/((std*std+eps)**0.5)
                catx=catx.add_(mean, alpha=-1.0).mul_(var_sqrt_i).mul_(gamma).add_(beta)
                catx=catx.reshape(oldshape)
                
            maxlimit=1e8
            h1=yp.div_(h1cs).clamp_(-maxlimit, maxlimit)
            del yp
            grad_before_bn=grad_output

            grad_input = grad_weight1 = grad_bias1 = grad_weight2 = grad_bias2 = None
            grad_h1=grad_before_bn*h1cs
            grad_h1cs=grad_before_bn.mul_(h1).mul_(h1cs)
            grad_h1cs=grad_h1cs.mul_(h1cs.mul_(-1.0).add_(1))
            del h1, h1cs, grad_before_bn
            if inputneedgrad:
                if len(grad_h1.shape) > len(w1.shape):
                    expandshape=grad_h1.shape[:len(grad_h1.shape) - len(w1.shape)]+w1.shape
                    w1reshape=w1.unsqueeze(0).expand(expandshape)
                    grad_input = (grad_h1).matmul(w1reshape)
                    # oldxshape=grad_h1.shape
                    # grad_h1_reshape=grad_h1.reshape(-1, grad_h1.shape[-2], grad_h1.shape[-1])
                    # w1reshape=w1reshape.reshape(-1, w1reshape.shape[-2], w1reshape.shape[-1])
                    # grad_input=grad_h1_reshape.bmm(w1reshape)
                    # grad_input=grad_input.reshape(oldxshape[:-1]+(grad_input.shape[-1],))
                else:
                    w1reshape=w1
                    grad_input = (grad_h1).matmul(w1reshape)
                if len(grad_h1cs.shape) > len(w2.shape):
                    expandshape=grad_h1cs.shape[:len(grad_h1cs.shape) - len(w2.shape)]+w2.shape
                    w2reshape=w2.unsqueeze(0).expand(expandshape)
                    try:
                        grad_input=grad_input.add_((grad_h1cs).matmul(w2reshape)) 
                    except Exception as e:
                        print(e)
                        print("grad_input shape:", grad_input.shape, "w2reshape:", w2reshape.shape, "grad_h1cs:", grad_h1cs.shape)
                        exit()
                    # oldxshape=grad_h1cs.shape
                    # grad_h1cs_reshape=grad_h1cs.reshape(-1, grad_h1cs.shape[-2], grad_h1cs.shape[-1])
                    # w2reshape=w2reshape.reshape(-1, w2reshape.shape[-2], w2reshape.shape[-1])
                    # grad_input=grad_input.add_(grad_h1cs_reshape.bmm(w2reshape).reshape(oldxshape[:-1]+(grad_input.shape[-1],)))
    
                else:
                    w2reshape=w2
                    grad_input=grad_input.add_((grad_h1cs).matmul(w2reshape)) 

            if w1.requires_grad:
                grad_weight1 = (grad_h1).transpose(-2,-1).matmul(catx) 
                if len(grad_h1.shape)-len(w1.shape)>0:
                    grad_weight1=grad_weight1.sum(dim=list(range(len(grad_h1.shape)-len(w1.shape))))
            if bias1.requires_grad:
                if len(grad_h1.shape)-len(bias1.shape) > 0:
                    grad_bias1 = (grad_h1).sum(dim=list(range(len(grad_h1.shape)-len(bias1.shape))))
                grad_bias1=grad_bias1.float()
            del grad_h1
                
            if w2.requires_grad:
                grad_weight2 = (grad_h1cs).transpose(-2,-1).matmul(catx) 
                if len(grad_weight2.shape)-len(w2.shape) > 0:
                    grad_weight2=grad_weight2.sum(dim=list(range(len(grad_weight2.shape)-len(w2.shape))))
            if bias2.requires_grad:
                if len(grad_h1cs.shape)-len(bias2.shape) > 0:
                    grad_bias2 = (grad_h1cs).sum(dim=list(range(len(grad_h1cs.shape)-len(bias2.shape))))
                grad_bias2=grad_bias2.float()
            del grad_h1cs, catx
            
            dgamma, dbeta = None, None
            if not gamma is None:     
                grad_input=grad_input.reshape(-1, gamma.shape[0])
                dout=grad_input
                N, D = dout.shape
                bninputx=torch.cat(x, dim=-1).reshape(-1, gamma.shape[0])
                xstd=bninputx.add(mean, alpha=-1.0).mul_(var_sqrt_i)
                dbeta = torch.sum(dout, axis=0)
                dgamma = torch.sum(xstd*dout, axis=0)
                dxstd = dout * gamma
                grad_input = 1.0/N * var_sqrt_i * (N*dxstd - torch.sum(dxstd, axis=0) - xstd*torch.sum(dxstd*xstd, axis=0))
                del xstd, dxstd, mean, std, var_sqrt_i 
                grad_input=grad_input.reshape(oldshape)
            
            splitsize=[item.shape[-1] for item in x]
            grad_inputs=torch.split(grad_input, splitsize, dim=-1)
    
            del grad_input
            # gc.collect()
            return grad_weight1, grad_bias1, grad_weight2, grad_bias2, dgamma, dbeta, *grad_inputs
    
class OptCssModel(nn.Module):
    def __init__(self, inputsize, outsize, bnlen=0,dropout=0.0):
        super(OptCssModel, self).__init__()
        self.hidden1 = nn.Linear(in_features=inputsize, out_features=outsize, bias=True)
        self.hidden1c = nn.Linear(in_features=inputsize, out_features=outsize, bias=True)
        self.cssfunc=CssFunction.apply
        if gv["optbn"]:
            self.bnlen=bnlen*inputsize
        else:
            self.bnlen=bnlen*outsize
        self.bnlayer=nn.BatchNorm1d(self.bnlen, track_running_stats=False)
        self.do=nn.Dropout(dropout)
        self.dropout=dropout

    def forward(self, x):
        if gv["optbn"]:
            if self.bnlen > 0:
                gamma, beta = self.bnlayer.weight, self.bnlayer.bias
            else:
                gamma, beta = None, None
        else:
            gamma, beta = None, None
        if isinstance(x, list):
            cross1=self.cssfunc(self.hidden1.weight, self.hidden1.bias, self.hidden1c.weight, self.hidden1c.bias, gamma, beta , *tuple(x))
        else:
            cross1=self.cssfunc(self.hidden1.weight, self.hidden1.bias, self.hidden1c.weight, self.hidden1c.bias, gamma, beta , x)
        if not gv["optbn"]:
            if self.bnlen > 0:
                cross1=self.bnlayer(cross1.reshape(-1,self.bnlen)).reshape(cross1.shape)
        if self.dropout > 0:
            cross1 = self.do(cross1)
        return cross1
    
class CssModel(nn.Module):
    def __init__(self, inputsize, outsize, bnlen=0, ln=False, act='sigmoid', dropout=0.0, res=False):
        super(CssModel, self).__init__()
        self.inputs=inputsize
        self.os=outsize
        # setup_seed(0)  
        self.hidden1 = nn.Sequential(
            nn.Linear(in_features=inputsize, out_features=outsize, bias=True),
            # torch.nn.Dropout(gv["dropout_rate"]),
            )
        # setup_seed(0)  
        self.hidden1c = nn.Sequential(
            nn.Linear(in_features=inputsize, out_features=outsize, bias=True),
            # torch.nn.Dropout(gv["dropout_rate"]),
            nn.Sigmoid() if act=='sigmoid' else nn.ReLU(),
            )
        self.bnlen=bnlen*outsize
        self.bnlayer=nn.BatchNorm1d(self.bnlen, track_running_stats=False)
        self.layernorm=nn.LayerNorm(outsize)
        self.lnflag=ln
        self.do=nn.Dropout(dropout)
        self.dropout=dropout
        self.resflag=res

    def forward(self, x):
        # if self.bnlen > 0:
        #     x=self.bnlayer(x.reshape(-1,self.bnlen)).reshape(x.shape)
        #     # self.bnx=x
        #     # self.bnx.retain_grad()
        h1 = self.hidden1(x)
        h1c = self.hidden1c(x)
        cross1 = h1*h1c
        if self.resflag:
            cross1=cross1+x
        if self.bnlen > 0:
            cross1=self.bnlayer(cross1.reshape(-1,self.bnlen)).reshape(cross1.shape)
        if self.lnflag:
            cross1=self.layernorm(cross1)
        if self.dropout > 0:
            cross1 = self.do(cross1)
        return cross1
CssModel=CssModel
class SeqCssModel(nn.Module):
    def __init__(self, in_channels, out_channels, groups, kernel_size=1, stride=1, padding=0, dilation=1,
                 bnlen=0, ln=False, act='sigmoid', dropout=0.0, res=False):
        super(SeqCssModel, self).__init__()
        self.hidden1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, groups=groups,
                      kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation),
            nn.Sigmoid() if act=='sigmoid' else nn.ReLU(),
            )
        self.hidden1c = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, groups=groups,
                    kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.do=nn.Dropout(dropout)
        self.dropout=dropout
        self.bnlen=bnlen*out_channels
        self.bnlayer=nn.BatchNorm1d(self.bnlen, track_running_stats=False)
        
    def forward(self, x):
        h1 = self.hidden1(x.unsqueeze(-1)).squeeze()
        h1c = self.hidden1c(x.unsqueeze(-1)).squeeze()
        cross1 = h1*h1c
        if self.dropout > 0:
            cross1 = self.do(cross1)
        if self.bnlen > 0:
            cross1=self.bnlayer(cross1.reshape(-1,self.bnlen)).reshape(cross1.shape)

        return cross1


class LiteCssModel(nn.Module):
    def __init__(self, inputsize, hidddensize, bnlen=0, ln=False, act='sigmoid', dropout=0.0, res=False):
        super(LiteCssModel, self).__init__()
        self.inputs=inputsize
        self.hidden1 = nn.Sequential(
            nn.Linear(in_features=inputsize, out_features=hidddensize, bias=True),
            nn.Sigmoid() if act=='sigmoid' else nn.ReLU(),
            )
        
        self.hidden2 = nn.Sequential(
            nn.Linear(in_features=hidddensize, out_features=inputsize, bias=True),
            nn.Sigmoid() if act=='sigmoid' else nn.ReLU(),
            )

    def forward(self, x):
        h1c = self.hidden1(x)
        self.h2c = self.hidden2(h1c)
        cross1 = self.h2c*x
        return cross1

class TransCssModel(nn.Module):
    def __init__(self, inputsize, inputsize2, outsize,hidddensize, bnlen=0, ln=False, act='sigmoid', dropout=0.0, res=False):
        super(TransCssModel, self).__init__()
        self.inputs=inputsize
        self.os=outsize
        self.hidden1 = nn.Sequential(
            nn.Linear(in_features=inputsize, out_features=hidddensize, bias=True),
            # torch.nn.Dropout(gv["dropout_rate"]),
            # nn.Sigmoid() if act=='sigmoid' else nn.ReLU(),
            )
        self.hidden1c = nn.Sequential(
            nn.Linear(in_features=inputsize, out_features=hidddensize, bias=True),
            # torch.nn.Dropout(gv["dropout_rate"]),
            )
        
        self.hidden2 = nn.Sequential(
            nn.Linear(in_features=hidddensize, out_features=inputsize, bias=True),
            # torch.nn.Dropout(gv["dropout_rate"]),
            nn.Sigmoid() if act=='sigmoid' else nn.ReLU(),
            )
        self.hidden2c = nn.Sequential(
            nn.Linear(in_features=hidddensize, out_features=inputsize, bias=True),
            # torch.nn.Dropout(gv["dropout_rate"]),
            )
        self.css=CssModel(inputsize2, outsize, bnlen=1)
        self.bnlen=bnlen*outsize
        self.bnlayer=nn.BatchNorm1d(self.bnlen, track_running_stats=False)
        self.layernorm=nn.LayerNorm(outsize)
        self.lnflag=ln
        self.do=nn.Dropout(dropout)
        self.dropout=dropout
        self.resflag=res

    def forward(self, x):
        x=self.css(x)
        x=x.permute(0,2,1)
        h1 = self.hidden1(x)
        self.h2 = self.hidden2(h1)
        h1c = self.hidden1c(x)
        self.h2c = self.hidden2c(h1c)
        
        cross1 = self.h2c*self.h2
        if self.resflag:
            cross1=cross1+x
        if self.bnlen > 0:
            cross1=self.bnlayer(cross1.reshape(-1,self.bnlen)).reshape(cross1.shape)
        if self.lnflag:
            cross1=self.layernorm(cross1)
        if self.dropout > 0:
            cross1 = self.do(cross1)
        cross1=cross1.permute(0,2,1)
        return cross1

class CtxNet(nn.Module):
    def __init__(self, insize,ctxlen,tickern):
        super(CtxNet, self).__init__()
        self.insize=insize
        self.tn=tickern
        self.ctxlen=ctxlen
        self.ctxhid1=CssModel(insize, self.ctxlen, bnlen=1)
        self.ctxhid2=CssModel(tickern, self.ctxlen, bnlen=1)

    def forward(self, x):
        batchsize=x.shape[0]
        ctx1=self.ctxhid1(x)
        ctx2=self.ctxhid2(ctx1.transpose(0, 1))
        return ctx2.flatten().reshape([1,-1]).repeat([batchsize, 1])

class FeedForwardNet(nn.Module):
    def __init__(self, inputsize, expandsionsize,bn=False):
        super(FeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(inputsize, expandsionsize, bias=False),
            nn.ReLU(),
            nn.Linear(expandsionsize, inputsize, bias=False)
        )
        self.bnlayer=nn.BatchNorm1d(inputsize, track_running_stats=False)
        self.bnflag=bn

    def forward(self, inputs):
        """
        inputs: [batch_size, seq_len, d_model]
        """
        residual = inputs
        output = self.fc(inputs)
        if self.bnflag:
            output=self.bnlayer(output)
        return output
    def cleardata(self):
        pass

class AttentionM(nn.Module):
    def __init__(self, hidden_size, outsize=0, num_attention_heads=8,dropout_prob=0.15, 
                 vdrop=-1.0, bn=False,ffflag=False, addflag=False):
        super(AttentionM, self).__init__()
        if outsize==0:
            outsize=hidden_size
        if vdrop<0:
            vdrop=dropout_prob
        if outsize % num_attention_heads != 0:   # ??′é?¤
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads, outsize))
        self.num_attention_heads = num_attention_heads    # 8

        self.attention_head_size = int(hidden_size / num_attention_heads)  # 16  ?ˉ???a?3¨???????¤′?????′?o|
        self.qlen=16
        self.vlen=int(outsize / num_attention_heads)
        self.all_head_size = int(self.num_attention_heads * (self.vlen))

        self.query = CssModel(hidden_size, num_attention_heads*self.qlen, dropout=0.0)    # 128, 128
        self.key = CssModel(hidden_size, num_attention_heads*self.qlen, dropout=0.0)
        self.value = CssModel(hidden_size, num_attention_heads*self.vlen, dropout=vdrop)
        self.bnlayer=nn.BatchNorm1d(self.all_head_size, track_running_stats=False)
        self.bnflag=bn
        # dropout
        self.dropout = nn.Dropout(dropout_prob)
        self.ff=FeedForwardNet(hidden_size,hidden_size,bn=False)
        self.ffflag=ffflag
        self.addflag=addflag
        self.attention_scores,self.attention_probs,self.context_layer=None,None,None

    def transpose_for_scores(self, x, elen):
        # INPUT:  x'shape = [bs, seqlen, hid_size]  ???è??hid_size=128
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, elen) # [bs, seqlen, 8, 16]
        x = x.view(*new_x_shape)   #
        return x.permute(0, 2, 1, 3)   # [bs, 8, seqlen, 16]
    def forward(self, x):
        xdim=len(x.shape)
        if xdim==2:
            x=x.unsqueeze(0)
        padding=16-x.shape[1]%16
        if padding > 0:
            paddingx=torch.zeros(x.shape[0], padding, x.shape[2]).to(x.device)
            x=torch.cat([x,paddingx], dim=1)
            
        batchsize=x.shape[0]
        # x=x.reshape(1,x.shape[0],x.shape[1])

        # ?o???§?????￠
        mixed_query_layer = self.query(x)   # [bs, seqlen, hid_size]
        mixed_key_layer = self.key(x)       # [bs, seqlen, hid_size]
        mixed_value_layer = self.value(x)   # [bs, seqlen, hid_size]

        query_layer = self.transpose_for_scores(mixed_query_layer,self.qlen)    # [bs, 8, seqlen, 16]
        key_layer = self.transpose_for_scores(mixed_key_layer,self.qlen)
        value_layer = self.transpose_for_scores(mixed_value_layer,self.vlen)   # [bs, 8, seqlen, 16]

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # è?????query???title?1?é?′?????1?§ˉ?3¨???????????°???è???????ˉ???é???????a?ooè?¤??o???é???o?èˉ￥??ˉ?????o1????|?????????????
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # [bs, 8, seqlen, 16]*[bs, 8, 16, seqlen]  ==> [bs, 8, seqlen, seqlen]
        attention_scores = attention_scores / math.sqrt(self.attention_head_size) / 1.0  # [bs, 8, seqlen, seqlen]

        attention_probs = nn.Softmax(dim=-1)(attention_scores)    # [bs, 8, seqlen, seqlen]
        attention_probs = self.dropout(attention_probs)
        # ???é?μ????1????[bs, 8, seqlen, seqlen]*[bs, 8, seqlen, 16] = [bs, 8, seqlen, 16]
        context_layer = torch.matmul(attention_probs, value_layer)   # [bs, 8, seqlen, 16]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()   # [bs, seqlen, 8, 16]
        new_context_layer_shape = context_layer.size()[0:-2] + (self.num_attention_heads*self.vlen,)   # [seqlen, 128]
        context_layer = context_layer.view(*new_context_layer_shape)
        if self.ffflag:
            context_layer=self.ff(context_layer)
        if self.addflag:
            context_layer+=x
        if self.bnflag:
            cshape=context_layer.shape
            context_layer=self.bnlayer(context_layer.reshape(cshape[0]*cshape[1],-1))
            context_layer=context_layer.reshape(cshape)
        if padding > 0:
            context_layer=context_layer[:,:-padding,:]
        if xdim==2:
            context_layer=context_layer.squeeze(0)
        return context_layer

class MixChannle(nn.Module):
    def __init__(self, chin, cho, feanum,bn=False,act='sigmoid'):
        super(MixChannle, self).__init__()
        self.layer1=nn.Conv1d(chin*feanum,cho*feanum,1, groups=feanum, bias=False)
        self.layer2=nn.Conv1d(chin*feanum,cho*feanum,1, groups=feanum, bias=False)
        self.bnlayer=nn.BatchNorm1d(cho*feanum, track_running_stats=False)
        self.bnflag=bn
        self.actfun=nn.Sigmoid() if act=='sigmoid' else nn.ReLU()
        self.hd,self.hdc,self.hd1=None,None,None

    def forward(self, x):
        self.hd=self.layer1(x)
        self.hdc=self.actfun(self.layer2(x))
        self.hd1=self.hd*self.hdc
        # if self.bnflag:
        #     self.hd1=self.bnlayer(self.hd1)
        return self.hd1

class MixNet(nn.Module):
    def __init__(self, pl):
        super(MixNet, self).__init__()
        self.mixs=nn.ModuleList([])
        self.chs=nn.ModuleList([])
        self.pl=pl
        for p in pl:
            chin, cho, feanum, cvoutnum=p[0],p[1],p[2],p[3]
            mix=MixChannle(chin, cho, feanum)
            self.mixs.append(mix)
            ch=CssModel(feanum,cvoutnum)
            self.chs.append(ch)


    def forward(self, x):
        batchsize=x.shape[0]
        tmp=x
        for i in range(len(self.pl)):
            p=self.pl[i]
            chin, cho, feanum, cvoutnum=p[0],p[1],p[2],p[3]
            mix=self.mixs[i]
            ch=self.chs[i]
            tmp=tmp.reshape(batchsize, chin, feanum).transpose(1,2).reshape(batchsize,-1,1)
            tmp=mix(tmp)
            tmp=tmp.reshape(batchsize,feanum,-1).transpose(1,2)
            tmp=ch(tmp).reshape(batchsize,-1)
        return tmp
    def cleardata(self):
        pass

def generate_sent_masks(batch_size, max_seq_length, source_lengths):
    enc_masks = torch.zeros(batch_size, max_seq_length, dtype=torch.float).cuda()
    for e_id, src_len in enumerate(source_lengths):
        enc_masks[e_id, :src_len] = 1
    return enc_masks

class AttSeqM(nn.Module):
    def __init__(self, sl,inqlen, invlen, hidden_size,mask=None,naheads=8,dropout_prob=0, bn=False,ffflag=False):
        super(AttSeqM, self).__init__()
        self.invlen=invlen
        self.seqlen=sl
        self.ylen=4
        self.posembedsize=8
        self.posembed=nn.Embedding(100, self.posembedsize,padding_idx=0)
        self.posembed.weight.data.fill_(0.0)
        if hidden_size % naheads != 0:   # ??′é?¤
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, naheads))
        self.naheads = naheads    # 8
        self.qlen=16
        self.vlen=int(hidden_size / naheads)
        self.inqlen=inqlen
        self.query = CssModel(self.inqlen+self.posembedsize, naheads*self.qlene, dropout=0.0)    # 128, 128
        self.key = CssModel(self.inqlen+self.posembedsize, naheads*self.qlen, dropout=0.0)
        self.value = CssModel(self.invlen+self.posembedsize, naheads*self.vlen, ln=True, dropout=dropout_prob)
        self.bnlayer=nn.BatchNorm1d(hidden_size, track_running_stats=False)
        self.bnflag=bn
        self.layernorm=nn.LayerNorm(self.inqlen)
        self.vlayernorm=nn.LayerNorm(self.invlen)
        # dropout
        self.dropout = nn.Dropout(dropout_prob)
        self.ff=FeedForwardNet(hidden_size,hidden_size,bn=False)
        self.ffflag=ffflag
        self.mask=mask
        self.attention_scores,self.attention_probs,self.context_layer=None,None,None

    def transpose_for_scores(self, x, elen):
        # INPUT:  x'shape = [bs, seqlen, hid_size]  ???è??hid_size=128
        new_x_shape = x.size()[:-1] + (self.naheads, elen) # [bs, seqlen, 8, 16]
        x = x.view(*new_x_shape)   #
        return x.permute(0, 2, 1, 3)   # [bs, 8, seqlen, 16]
    def forward(self, posid,qcv,y=None):
        # x=x.detach()
        batchsize=qcv.shape[0]

        attention_mask = self.mask.repeat([batchsize,1]).unsqueeze(1).unsqueeze(2)   # [bs, 1, 1, seqlen] ?￠??????′?o|
        attention_mask = (1.0 - attention_mask) * -10000.0   # padding???token?????o-10000???exp(-1w)=0


        posembed=self.posembed(posid.long())
        qcv=torch.cat([qcv,posembed],dim=2)
        if y == None:
            vcv=qcv
        else:
            vcv=torch.cat([qcv,y],dim=2)
        # qcv=self.layernorm(qcv)
        # vcv=self.vlayernorm(vcv)
        mixed_query_layer = self.query(qcv[:,0:1])   # [bs, 1, qlen*naheads]
        mixed_key_layer = self.key(qcv)       # [bs, seqlen, qlen*naheads]
        mixed_value_layer = self.value(vcv)   # [bs, seqlen, vlen*naheads]

        query_layer = self.transpose_for_scores(mixed_query_layer,self.qlen)   # [bs, 8, 1, qlen]
        key_layer = self.transpose_for_scores(mixed_key_layer,self.qlen) # [bs, 8, seqlen, qlen]
        value_layer = self.transpose_for_scores(mixed_value_layer,self.vlen)   # [bs, 8, seqlen, vlen]

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) #[bs, 8, 1,seqlen]
        attention_scores = attention_scores + attention_mask
        # [bs, 8, seqlen, 16]*[bs, 8, 16, seqlen]  ==> [bs, 8, seqlen, seqlen]
        self.attention_scores = attention_scores / math.sqrt(self.qlen) / 1.0  # [bs, 8, 1,seqle]

        # attention_scores=clampgrad(attention_scores,-9.0,9.0)
        self.attention_probs = nn.Softmax(dim=-1)(self.attention_scores)    # [bs, 8, 1,seqlen]
        self.attention_probs = self.dropout(self.attention_probs)
        # ???é?μ????1????[bs, 8, 1, seqlen]*[bs, 8, seqlen, vlen] = [bs, 8, 1, vlen]
        context_layer = torch.matmul(self.attention_probs, value_layer)   # [bs, 8, seqlen, vlen]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()   # [bs, 1, 8, vlen]
        new_context_layer_shape = context_layer.size()[0:-2] + (self.naheads*self.vlen,)   # [seqlen, 128]
        self.context_layer = context_layer.view(*new_context_layer_shape)
        if self.ffflag:
            self.context_layer=self.ff(self.context_layer)
        if self.bnflag:
            cshape=self.context_layer.shape
            self.context_layer=self.bnlayer(self.context_layer.reshape(cshape[0],-1))
            self.context_layer=self.context_layer.reshape(cshape)
        return self.context_layer

class AttSeqMall(nn.Module):
    def __init__(self, sl, inqlen, invlen, hidden_size,mask=None,naheads=8,dropout_prob=0, bn=False,ffflag=False):
        super(AttSeqMall, self).__init__()
        self.invlen=invlen
        self.seqlen=sl
        self.ylen=4
        self.posembedsize=16
        self.posembed=nn.Embedding(100, self.posembedsize,padding_idx=0)
        self.posembed.weight.data.fill_(0.0)
        if hidden_size % naheads != 0:   # ??′é?¤
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, naheads))
        self.naheads = naheads    # 8
        self.qlen=16
        self.vlen=int(hidden_size / naheads)
        self.inqlen=inqlen
        self.query = CssModel(self.inqlen, naheads*self.qlen, dropout=0.0)    # 128, 128
        self.key = CssModel(self.inqlen, naheads*self.qlen, dropout=0.0)
        self.value = CssModel(self.invlen, naheads*self.vlen, ln=True, dropout=dropout_prob)
        self.bnlayer=nn.BatchNorm1d(hidden_size*self.seqlen, track_running_stats=False)
        self.bnflag=bn
        self.qbn=nn.BatchNorm1d(self.inqlen*self.seqlen, track_running_stats=False)
        self.vbn=nn.BatchNorm1d(self.invlen*self.seqlen, track_running_stats=False)
        # dropout
        self.dropout = nn.Dropout(dropout_prob)
        self.ff=FeedForwardNet(hidden_size,hidden_size,bn=False)
        self.ffflag=ffflag
        self.mask=mask
        self.attention_scores,self.attention_probs,self.context_layer=None,None,None

    def transpose_for_scores(self, x, elen):
        # INPUT:  x'shape = [bs, seqlen, hid_size]  ???è??hid_size=128
        new_x_shape = x.size()[:-1] + (self.naheads, elen) # [bs, seqlen, 8, 16]
        x = x.view(*new_x_shape)   #
        return x.permute(0, 2, 1, 3)   # [bs, 8, seqlen, 16]
    def forward(self, posid, qcv,y=None):
        # x=x.detach()
        batchsize=qcv.shape[0]

        attention_mask = self.mask.repeat([batchsize,self.naheads,1,1])   # [bs, naheads, seqlen, seqlen] ?￠??????′?o|
        attention_mask = (1.0 - attention_mask) * -10000.0   # padding???token?????o-10000???exp(-1w)=0

        posembed=self.posembed(posid.long())

        if y == None:
            vcv=qcv
        else:
            vcv=torch.cat([qcv,y],dim=2)
        # qcv=self.qbn(qcv.reshape(batchsize,-1)).reshape(batchsize,self.seqlen,-1)
        # vcv=self.vbn(vcv.reshape(batchsize,-1)).reshape(batchsize,self.seqlen,-1)
        mixed_query_layer = self.query(qcv)   # [bs, seqlen, qlen*naheads]
        mixed_key_layer = self.key(qcv)       # [bs, seqlen, qlen*naheads]
        mixed_value_layer = self.value(vcv)   # [bs, seqlen, vlen*naheads]

        query_layer = self.transpose_for_scores(mixed_query_layer,self.qlen)   # [bs, naheads, seqlen, qlen]
        key_layer = self.transpose_for_scores(mixed_key_layer,self.qlen) # [bs, naheads, seqlen, qlen]
        value_layer = self.transpose_for_scores(mixed_value_layer,self.vlen)   # [bs, naheads, seqlen, vlen]

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) #[bs, naheads, seqlen,seqlen]
        attention_scores = attention_scores + attention_mask
        # [bs, 8, seqlen, 16]*[bs, 8, 16, seqlen]  ==> [bs, 8, seqlen, seqlen]
        self.attention_scores = attention_scores / math.sqrt(self.qlen) / 1.0  # [bs, naheads, seqlen,seqle]

        # attention_scores=clampgrad(attention_scores,-9.0,9.0)
        self.attention_probs = nn.Softmax(dim=-1)(self.attention_scores)    # [bs, naheads, seqlen,seqlen]
        self.attention_probs = self.dropout(self.attention_probs)
        # ???é?μ????1????[bs, 8, 1, seqlen]*[bs, 8, seqlen, vlen] = [bs, 8, 1, vlen]
        context_layer = torch.matmul(self.attention_probs, value_layer)   # [bs, naheads, seqlen, vlen]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()   # [bs, seqlen, naheads, vlen]
        new_context_layer_shape = context_layer.size()[0:-2] + (self.naheads*self.vlen,)   # [seqlen, 128]
        self.context_layer = context_layer.view(*new_context_layer_shape)
        if self.ffflag:
            self.context_layer=self.ff(self.context_layer)
        if self.bnflag:
            cshape=self.context_layer.shape
            self.context_layer=self.bnlayer(self.context_layer.reshape(cshape[0],-1))
            self.context_layer=self.context_layer.reshape(cshape)
        return self.context_layer
        
class ScalePool(nn.Module):
    def __init__(self,sl,feanum):
        super(ScalePool, self).__init__()
        self.poolings=nn.ModuleList([])
        k=2
        while k < sl:
            pool=nn.AvgPool1d(k,1)
            self.poolings.append(pool)
            k=k*2
        self.sl=sl
        self.feanum=feanum
    def forward(self, x):
        os=[]
        tmppad=x.mean(dim=3).unsqueeze(3).repeat([1,1,x.shape[3]]).detach()
        k=2
        i=0
        for i in range(len(self.poolings)):
            o=self.poolings[i](torch.cat([x,tmppad[:,:,:k]], dim=3))
            os.append(o)
            k=k*2

        os=torch.cat(os,dim=1)
        return os
    def getoutsize(self):
        return len(self.poolings)*self.feanum
        
class MomentumV(nn.Module):
    def __init__(self,sl):
        super(MomentumV, self).__init__()
        self.moms=nn.ModuleList([])
        k=2
        while k <= sl:
            mom_conv = nn.Conv1d(in_channels=1,
                      out_channels=1,
                      kernel_size=k)
            mom_conv.weight.data=torch.zeros([1,1,k])
            mom_conv.weight.data[0]=-1
            mom_conv.weight.data[-1]=1

            self.moms.append(mom_conv)
            k=k*2

        self.sl=sl
    def forward(self, x):
        os=[]
        tmppad=x.mean(dim=3).unsqueeze(3).repeat([1,1,x.shape[3]]).detach()
        k=2
        i=0
        for i in range(len(self.poolings)):
            o=self.moms[i](torch.cat([x,tmppad[:,:,:k]],dim=3))
            os.append(o)
            k=k*2

        os=torch.cat(os,dim=1)
        return os
    def getoutsize(self):
        return len(self.moms)*self.feanum
        
class SeqCurve(nn.Module):
    def __init__(self,sl,feanum):
        super(SeqCurve, self).__init__()
        self.means=ScalePool(sl,feanum)
        self.moms=MomentumV(sl,feanum)

        self.sl=sl
        self.feanum=feanum
    def forward(self, x):
        os=[]
        mcurve=self.means(x)
        momcurve=self.moms(x)

        os=torch.cat([x,mcurve,momcurve],dim=2)
        return os
    def getoutsize(self):
        return self.feanum+self.means.getoutsize()+self.moms.getoutsize()

        
class SeqFea(nn.Module):
    def __init__(self,sl,feanum,odim):
        super(SeqFea, self).__init__()
        self.sc=SeqCurve(sl,feanum)
        self.feamerge=CssModel(self.sc.getoutsize(),odim)

        self.sl=sl
        self.feanum=feanum
    def forward(self, x):
        sco=self.sc(x)
        out=self.feamerge(sco)

        return out

class MultiPool(nn.Module):
    def __init__(self,sl,minsplit=8):
        super(MultiPool, self).__init__()
        self.poolings=nn.ModuleList([])
        outsize=0
        k=2
        while k*minsplit <= sl:
            pool=nn.AdaptiveAvgPool1d(k)
            self.poolings.append(pool)          

            outsize+=k
            for j in range(len(self.poolings)):
                n=torch.tensor([2]).pow(j).item()
                # outsize+=int(k/n)-1
                outsize+=k-n
            k=k*2
                
        self.outsize=outsize
        self.sl=sl
    def forward(self, x):
        os=[]

        i=0
        for i in range(len(self.poolings)):
            o=self.poolings[i](x)
            os.append(o)
            for j in range(i+1):
                n=torch.tensor([2]).pow(j).item()
                # if len(o.shape)==2:
                #     omom=torch.diff(o[:,::n])
                # else:
                #     omom=torch.diff(o[:,:,::n])
                omom=torch.diff(o, n=n)
                
                os.append(omom)
        os=torch.cat(os,dim=-1)
        return os
    def getoutsize(self):
        return self.outsize

class AvgPools(nn.Module):
    def __init__(self,sl,minsplit=8):
        super(AvgPools, self).__init__()
        self.poolings=nn.ModuleList([])
        outsize=0
        k=1
        while k*minsplit <= sl:
            pool=nn.AdaptiveAvgPool1d(k)
            self.poolings.append(pool)          
            outsize+=k
            k=k*2
        self.outsize=outsize
        self.sl=sl
    def forward(self, x):
        os=[]
        i=0
        for i in range(len(self.poolings)):
            o=self.poolings[i](x)
            os.append(o)
        os=torch.cat(os,dim=-1)
        return os
    def getoutsize(self):
        return self.outsize
    
class CosPools(nn.Module):
    def __init__(self,sl,minsplit=64):
        super(CosPools, self).__init__()
        self.poolings=nn.ModuleList([])
        outsize=0
        k=minsplit
        while k <= sl:      
            outsize+=1
            k=k*2
                
        self.outsize=outsize
        self.minsplit=minsplit
        self.sl=sl
    def forward(self, x, y):
        os=[]
        k=self.minsplit
        while k <= self.sl:
            # o=F.cosine_similarity(x[-k:],y[-k:], dim=0).detach()
            o=[]
            for si in range(0,x.shape[1],128):
                s=si
                e=min(si+128, x.shape[1])
                tmpo=F.cosine_similarity(x[-k:, s:e],y[-k:, s:e].repeat([1,1,x.shape[2]]), dim=0).detach()
                o.append(tmpo)
            o=torch.cat(o, dim=0)
            #torch.cuda.empty_cache()
            os.append(o)
            k*=2
        os=torch.stack(os)
        return os.detach()
    def getoutsize(self):
        return self.outsize
 

def flog(*args,**kwargs):
    print(*args,**kwargs)
    # sys.stdout.flush()

def adamwreg(
        names_with_grad,
        params: List[Tensor],
          grads: List[Tensor],
          exp_avgs: List[Tensor],
          exp_avg_sqs: List[Tensor],
          exp_vars: List[Tensor],
          max_exp_avg_sqs: List[Tensor],
          state_steps: List[int],
          *,
          amsgrad: bool,
          varopt: bool,
          beta1: float,
          beta2: float,
          lr: float,
          weight_decay1: float,
          weight_decay2: float,
          eps: float,
          maximize: bool,
          theta: float):
    r"""Functional API that performs AdamW algorithm computation.

    See :class:`~torch.optim.AdamW` for details.
    """
    baselr=lr
    for i, param in enumerate(params):
        name=names_with_grad[i]
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        exp_var=exp_vars[i]
        step = state_steps[i]

        # Perform stepweight decay
        # if "d1m.curmix" in name:
        #     weight_decay2*=2.0
        #embed lr
        lr=baselr
        if "embed" in name:
            lr*=gv["embedlr"]
        if "icmodel" in name:
            lr*=gv["swratio"]
        param.mul_(max(0,1 - lr * weight_decay2))
        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        exp_d=grad-exp_avg
        exp_var.mul_(beta2).addcmul_(exp_d, exp_d, value=1.0 - beta2)
        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
            # Use the max. for normalizing running avg. of gradient
            denom = (max_exp_avg_sqs[i].sqrt() / math.sqrt(bias_correction2)).add_(eps)
        else:
            if varopt:
                denom = (exp_var.sqrt() / math.sqrt(bias_correction2)).add_(eps)
            else:
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

        step_size = lr / bias_correction1

        param.addcdiv_(exp_avg, denom, value=-step_size)

        if weight_decay1 != 0:
            mask1=(param >= 0) & (param < theta)
            mask2=(param < 0) & (param > -theta)
            param[mask1]=torch.clamp(param[mask1]-lr*weight_decay1,-10000,0)
            param[mask2]=torch.clamp(param[mask2]+lr*weight_decay1,0,100000)

class AdamWReg(torch.optim.Optimizer):
    def __init__(self, named_params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay1=0, weight_decay2=0, theta=5e-4,
                amsgrad=False, *, maximize: bool = False, varopt=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay1:
            raise ValueError("Invalid weight_decay1 value: {}".format(weight_decay1))
        if not 0.0 <= weight_decay2:
            raise ValueError("Invalid weight_decay2 value: {}".format(weight_decay2))

        params=[]
        names=[]
        for name,param in named_params:
            names.append(name)
            params.append(param)
        theta=lr*weight_decay1*2
        defaults = dict(lr=lr, betas=betas, eps=eps, amsgrad=amsgrad, maximize=maximize,
                        weight_decay1=weight_decay1,theta=theta, varopt=varopt,weight_decay2=weight_decay2,
                        names=names)
        super(AdamWReg, self).__init__(iter(params), defaults)

    def __setstate__(self, state):
        super(AdamWReg, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
            group.setdefault('maximize', False)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            exp_vars = []
            state_sums = []
            max_exp_avg_sqs = []
            state_steps = []
            amsgrad = group['amsgrad']
            beta1, beta2 = group['betas']
            names=group['names']
            varopt=group['varopt']
            names_with_grad=[]
            for idx,p in enumerate(group['params']):
                name=names[idx]
                if p.grad is None:
                    continue
                if p.grad.abs().sum() == 0.0:  # filter grad zero
                    continue
                if (~p.grad.isfinite()).sum() > 0:
                    print("grad nan")
                    continue
                params_with_grad.append(p)
                names_with_grad.append(name)
                if p.grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')
                grads.append(p.grad)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    state['exp_var'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])
                exp_vars.append(state['exp_var'])

                if amsgrad:
                    max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                # update the steps for each param group update
                state['step'] += 1
                # record the step after step update
                state_steps.append(state['step'])

            adamwreg(names_with_grad,
                     params_with_grad,
                    grads,
                    exp_avgs,
                    exp_avg_sqs,
                    exp_vars,
                    max_exp_avg_sqs,
                    state_steps,
                    amsgrad=amsgrad,
                    varopt=varopt,
                    beta1=beta1,
                    beta2=beta2,
                    lr=group['lr'],
                    weight_decay1=group['weight_decay1'],
                    weight_decay2=group['weight_decay2'],
                    eps=group['eps'],
                    maximize=group['maximize'],
                    theta=group['theta'])

        return loss

def validf(idsfeas):
    vf=idsfeas[:,gv["valitidx"]]>gv["validvalue"]
    return vf

def pccfeaidx():
    return gv["sfpcc"].abs()
def pccfeasign():
    return (gv["sfpcc"]<0)
def torchrank(x):
    rankx=x.clone().detach()
    for i in range(rankx.shape[1]):
        _,rankx[:,i]=torch.unique(rankx[:,i], sorted=True, return_inverse=True)
    return rankx

def get_keys_from_dict(keys,feadict):
    # need make sure keys in dict_key
    out = itemgetter(*keys)(feadict)
    return np.array(list(out))

def getlswratio(outv,lswratio):
    sigmoidfun=lambda x:(1/(1+np.exp(-x)))
    if torch.is_tensor(outv):
        lsw=(outv).sigmoid().detach()
    else:
        lsw=sigmoidfun(-outv)
    lsw=lsw*lswratio+1.0
    # lsw[outv<0]=1.0
    return lsw

def getshortw(tyv,shortwratio):
    sigmoidfun=lambda x:(1/(1+np.exp(-x)))
    if torch.is_tensor(tyv):
        shortw=(-tyv/tyv.std(dim=0).clamp(gv["eps"])).sigmoid()
    else:
        shortw=sigmoidfun(-tyv/tyv.std())
    shortw=shortw*shortwratio+1.0
    return shortw

def pearsonrsl(outv,tyv, batch_first,dim,lswratio,shortwratio):
    lsw=getlswratio(outv,lswratio)
    shortw=getshortw(tyv,shortwratio)
    if torch.is_tensor(outv):
        cosvalue=torch.nn.functional.cosine_similarity(outv.detach(),tyv.detach()*lsw*shortw,dim=dim)
    else:
        cosvalue=outv.dot(tyv*lsw*shortw) / (np.linalg.norm(outv) * np.linalg.norm(tyv*lsw*shortw))
    return cosvalue

def pearsonrslic(outv,tyv, batch_first,dim,lswratio,shortwratio):
    normoutv=(outv-outv.mean(dim=0))/outv.std(dim=0).clamp(gv["eps"])
    ic=pearsonrsl(normoutv.detach(),tyv, batch_first,dim,lswratio,shortwratio)
    return ic

def dldetach(dd):
    for key in dd.keys():
        if torch.is_tensor(dd[key]):
            dd[key]=dd[key].detach()
def dltotensor(dl,keylist=None):
    ll={}
    for i,dd in enumerate(dl):
        for key in dd.keys():
            if (not keylist is None) and (key not in keylist):
                continue
            if not key in ll:
                ll[key] = []
            data=dd[key]
            ll[key].append(data)
    for key in ll.keys():
        ll[key]=torch.stack(ll[key])
    return ll
def dltonp(dl,keylist=None):
    ll={}
    for i,dd in enumerate(dl):
        for key in dd.keys():
            if (not keylist is None) and (key not in keylist):
                continue
            if not key in ll:
                ll[key] = []
            ll[key].append(dd[key].cpu().detach().numpy().reshape(1,-1))
    for key in ll.keys():
        ll[key]=np.vstack(ll[key])
    return ll
def statstrmean(dl,keylist=None,ra=None):
    if keylist is None:
        keylist = dl.keys()
    resstr=""
    for key in keylist:
        if not key in dl:
            continue
        resstr+=key+":"
        if type(dl[key]) is torch.Tensor:
            if ra is None:
                data=dl[key]
            else:
                data=dl[key][[x for x in ra if abs(x)<dl[key].shape[0]-1]]
            if key =="coslist":
                cosm=calc_cosmean(data*-1.0, gv["zzyidx"])
                resstr+=str(torch.squeeze(cosm[0]).tolist())+str(torch.squeeze(cosm[1]).tolist())+str(torch.squeeze(cosm[2]).tolist())
                resstr+=" "
            else:
                resstr+=str(torch.squeeze(data.mean(dim=0)).cpu().numpy())
                resstr+="_"+str(torch.squeeze(data.std(dim=0)).cpu().numpy())
                resstr+=" "
        else:
            resstr+=str(dl[key])+" "
    return resstr
            
def statstr(dl,idx=None,keylist=None,array=True):
    if keylist is None:
        keylist = dl.keys()
    resstr=""
    for key in keylist:
        resstr+=" "+key+":"
        if array:
            data=dl[key][idx]
        else:
            data=dl[key]
        if type(data) is torch.Tensor:
            if key =="coslist":
                cosm=calc_cosmean(data.unsqueeze(0)*-1.0, gv["zzyidx"])
                resstr+=str(torch.squeeze(cosm[0]).tolist())+str(torch.squeeze(cosm[1]).tolist())+str(torch.squeeze(cosm[2]).tolist())
                resstr+=" "
            else:
                resstr+=str(torch.squeeze(data).cpu().numpy())
                resstr+=" "
        else:
            resstr+="error"
    return resstr
    
def getpartialdict(dd,keylist):
    res={}
    for key in keylist:
        res[key]=dd[key]
    return res
def dttodict(dd,keylist=None):
    res={}
    if keylist is None:
        keylist=dd.keys()
    for key in keylist:
        res[key]=dd[key].cpu().tolist()
    return res

def calc_cosmean(cosvl,daynum):
    cosvlm=cosvl.mean(dim=0)
    cosvlmm=cosvlm.mean(dim=0).mean(dim=0)
    wcosvlmm=weightedavgday(cosvlm,0.7,daynum).mean()
    return wcosvlmm,cosvlmm,cosvlm.flatten()

def weightedavgday(data,decay,targetdaynum):
    assert targetdaynum<=data.shape[0]
    # assert len(data.shape)==3
    dayweights=torch.tensor([0.7**x for x in range(targetdaynum)]).cuda()
    res=(data[:targetdaynum]*dayweights.view(targetdaynum,1,1)).sum(dim=0)/dayweights.sum(dim=0)
    return res
    
        
def getsimratio(output,vf):
    output=output.reshape(vf.shape[0],-1)
    ratio=torch.zeros(output.shape).to(output.device)
    ratio[~vf,:]=-1000.0
    ratio[vf]=output[vf]/output[vf].std(dim=0)*gv["offscale"]
    
    offnum=min(gv["hlnum"],vf.sum())
    offv=ratio.topk(offnum, dim=0).values[-1].detach()
    ratio=ratio-offv.detach()-gv["offt"]
    # ratio=nn.Softmax(dim=0)(ratio)
    ratio=ratio.sigmoid()
    ratio=ratio/ratio.sum(dim=0).clamp(gv["eps"])
    return ratio
      
def getwratiotop(output,vf, hlnum=1000):
    ratio=output.clone().detach()
    ratio=ratio.reshape(vf.shape[0],-1)
    ratio=ratio/ratio.std(dim=0).clamp(gv["eps"])*gv["offscale"]
    ratio[~vf,:]=-10000.0
    offnum=min(hlnum, ratio.shape[0])
    minw=ratio.topk(offnum, dim=0).values[-1]
    ratio=ratio-minw
    ratio[ratio<0]=0
    ratio=ratio/ratio.sum(dim=0).clamp(gv["eps"])
    return ratio

def getwratiotopavg(output,hlnum):
    arg=output.argsort(dim=0).argsort(dim=0)
    ratio=torch.zeros(arg.shape).to(output.device)
    ratio[arg>=arg.shape[0]-hlnum]=1
    ratio=ratio/ratio.sum(dim=0)
    return ratio

eps=1e-8

class CircleTensor:
    def __init__(self, maxlen=1280, dtype=torch.float, insstart=0, expandlen=16*16, bufdict=None, cuda=True):
        if not bufdict is None:
            self.load(bufdict)
            return
        # assert maxlen % expandlen ==0 and maxlen / expandlen > 0 , "circletensor error:"+str((maxlen, s, e, expandlen))
        self.s, self.e = int(insstart), int(insstart)
        self.expandlen=int(expandlen)
        self.maxlen=int(maxlen)
        self.data=None
        self.dtype=dtype
        self.cuda=cuda
    def save(self):
        bufdict=dict(s=self.s, e=self.e, maxlen=self.maxlen, data=self.data, dtype=self.dtype, expandlen=self.expandlen)
        return bufdict
    def load(self, bufdict):
        self.s=bufdict["s"]
        self.e=bufdict["e"]
        self.maxlen=bufdict["maxlen"]
        self.data=bufdict["data"]
        self.dtype=bufdict["dtype"]
        self.expandlen=bufdict["expandlen"]
    def inner_idx(self, s):
        return s-self.s
    def setdata(self, xdata, s, e):
        self.data=xdata[:].clone()
        self.s, self.e = s, e
            
    def append(self, xdata, s):
        if self.data is None:
            assert xdata.shape[0] <= self.maxlen
            self.data=torch.zeros((self.maxlen,)+xdata.shape[1:], dtype=self.dtype)
            if self.cuda:
                self.data=self.data.cuda()
            self.data[:xdata.shape[0]]=xdata[:]
            self.s, self.e=s,s+xdata.shape[0]
            self.shape=self.data.shape
            return
        assert s == self.e and xdata.shape[0] <= self.maxlen, "append error:"+str((s,self.s, self.e, xdata.shape, self.maxlen))
        #
        curtail=self.e-self.s
        appendlen=min(self.maxlen-curtail, xdata.shape[0])
        self.data[curtail:curtail+appendlen]=xdata[:appendlen]
        self.e+=appendlen
        xdataidx=appendlen
        #
        if xdataidx<xdata.shape[0]:
            poplen=max(xdata.shape[0]-xdataidx, self.expandlen)
            self.data=self.data.roll(shifts=-poplen, dims=0)
            self.s+=poplen
            curtail=self.e-self.s
            self.data[curtail:curtail+xdata[xdataidx:].shape[0]]=xdata[xdataidx:]
            self.e+=xdata[xdataidx:].shape[0]
        return
            
    def getdata(self, ss, ee=0):
        flag=(ee!=0)
        ee=ss+1 if ee==0 else ee
        assert ss >= self.s and ss < self.e and ee<= self.e and ee >self.s, "getdata error:"+str((ss, ee, self.s, self.e))
        data=self.data[ss-self.s:ee-self.s] if flag else self.data[ss-self.s]
        return data
    def __getitem__(self, slice_idx):
        if isinstance(slice_idx, slice):
            return self.getdata(slice_idx.start, slice_idx.stop)
        else:
            return self.getdata(slice_idx)

class CircleNp:
    def __init__(self, maxlen=1280, insstart=0, expandlen=16*16, bufdict=None):
        if not bufdict is None:
            self.load(bufdict)
            return
        # assert maxlen % expandlen ==0 and maxlen / expandlen > 0 , "circletensor error:"+str((s, e, expandlen))
        self.maxlen=maxlen
        self.data=None
        self.s, self.e = insstart, insstart
    def save(self):
        bufdict=dict(s=self.s, e=self.e, maxlen=self.maxlen, data=self.data)
        return bufdict
    def load(self, bufdict):
        self.s=bufdict["s"]
        self.e=bufdict["e"]
        self.maxlen=bufdict["maxlen"]
        self.data=bufdict["data"]
    def savepath(self, path):
        np.save(path+".npy", self.data)
        torch.save((self.s, self.e, self.maxlen, self.data.shape), path+".shape")
    def loadpath(self, path):
        self.s, self.e, self.maxlen, xshape=torch.load(path+".shape")
        self.data=np.load(path+".npy")
    def circleidx(self, tidx):
        return tidx % self.data.shape[0]
    def append(self, xdata, s):
        if self.data is None:
            self.data=self.data=np.zeros((self.maxlen,)+xdata.shape[1:])
        assert s == self.e and xdata.shape[0] < self.data.shape[0], "append error:"+str((s, self.s, self.e, xdata.shape))
        for i in range(xdata.shape[0]):
            self.data[self.circleidx(self.e)]=xdata[i]
            self.e+=1
            if self.e -self.s >= self.data.shape[0]:
                self.s+=1
    def update(self, xdata, s):
        assert s < self.e and s+xdata.shape[0] == self.e, "update error:"+str((s,self.s, self.e, xdata.shape))
        ss=self.circleidx(s)
        self.data[ss]=xdata
        
    def getdata(self, ss, ee=0):
        ee=ss+1 if ee==0 else ee
        assert ss >= self.s and ss < self.e and ee<= self.e and ee >self.s, "getdata error:"+str((ss, ee, self.s, self.e))
        css=self.circleidx(ss)
        cee=self.circleidx(ee)
        if css < cee:
            return self.data[css:cee]
        else:
            return np.concatenate([self.data[css:], self.data[:cee]], axis=0)
        
class RollingNorm:
    def __init__(self, momentum=0.95, eps=1e-8, backdaylen=1280, bufdict=None, rncp=None):
        if not bufdict is None:
            self.load(bufdict)
            return
        if rncp is None:
            self.mean, self.std = None, None
        else:
            self.mean, self.std = rncp[0], rncp[1]
        self.cp=[(self.mean, self.std)]
        self.momentum=momentum
        self.eps=eps
        self.backdaylen=backdaylen
    def save(self):
        bufdict=dict(mean=self.mean, std=self.std, momentum=self.momentum, eps=self.eps, cp=self.cp, backdaylen=self.backdaylen)
        return bufdict
    def load(self, bufdict):
        self.mean=bufdict["mean"]
        self.std=bufdict["std"]
        self.momentum=bufdict["momentum"]
        self.eps=bufdict["eps"]
        self.cp=bufdict["cp"]
        self.backdaylen=bufdict["backdaylen"]
    def getcp(self):
        off=len(self.cp)-self.backdaylen-1
        assert off >= 0
        return self.cp[off]
    def norm(self, x, upnorm=False):
        x=x.astype(np.float64)
        x[np.isinf(x)]=np.nan
        validfeanum=np.isfinite(x).sum(axis=0)
        invalidfea=validfeanum < x.shape[0]*0.01
        tmpmean,tmpstd=np.nanmean(x, axis=0), np.nanstd(x,axis=0)
        tmpmean[invalidfea]=0
        tmpstd[invalidfea]=0
        if upnorm:
            if self.mean is None:
                self.mean, self.std = tmpmean, tmpstd
            else:
                self.mean = self.mean*self.momentum+tmpmean*(1.0-self.momentum)
                self.std = self.std*self.momentum+tmpstd*(1.0-self.momentum)
            self.cp.append((self.mean, self.std))
        x = (x-self.mean)/np.clip(self.std, self.eps, self.std.max())
        x[~np.isfinite(x)]=0.0
        x[:,invalidfea]=0
        x=np.clip(x, -100, 100)
        return x
 
        
class COriIns:
    def __init__(self, insstart=1280, buffersize=1280, bufdict=None, expandlen=16*16):
        if not bufdict is None:
            self.load(bufdict)
            return
        self.buffersize=buffersize
        self.insstart=insstart
        self.tmseq=np.array([])
        lessbuffsize=(buffersize-(gv["seqlen"]-128)*gv["insperday"])
        # lessbuffsize=buffersize
        self.x=CircleTensor(buffersize,dtype=torch.float, insstart=insstart, expandlen=expandlen, cuda=gv["inscuda"])
        self.dayx=CircleTensor(buffersize/gv["insperday"],dtype=torch.float, insstart=insstart/gv["insperday"], expandlen=expandlen/gv["insperday"], cuda=gv["inscuda"])
        self.trans15x=CircleTensor(buffersize/gv["min2t15"],
                                   dtype=torch.float, insstart=insstart/gv["min2t15"], expandlen=expandlen/gv["min2t15"], cuda=gv["inscuda"])
        self.trans1x=CircleTensor(int(lessbuffsize/gv["min2t1"]),
                                  dtype=torch.float, 
                                  insstart=int(insstart/gv["min2t1"]), 
                                  expandlen=int(expandlen/gv["min2t1"]), cuda=gv["inscuda"])
        self.rdatas=CircleTensor(int(buffersize/gv["insperday"]),dtype=torch.float32, 
                               insstart=insstart/gv["insperday"], expandlen=expandlen/gv["insperday"], cuda=gv["inscuda"])
        
        self.upnum=16
    def save(self, path):
        bufdict=dict(buffersize=self.buffersize, insstart=self.insstart, 
                     tmseq=self.tmseq, x=self.x.save(), dayx=self.dayx.save(), trans15x=self.trans15x.save(),
                     trans1x=self.trans1x.save(),
                     rdatas=self.rdatas.save(),
                     upnum=self.upnum)
        torch.save(bufdict, path)
    def load(self, path):
        bufdict=torch.load(path)
        self.buffersize=bufdict["buffersize"]
        self.insstart=bufdict["insstart"]
        self.tmseq=bufdict["tmseq"]
        self.upnum=bufdict["upnum"]
        self.x=CircleTensor(bufdict=bufdict["x"])
        self.dayx=CircleTensor(bufdict=bufdict["dayx"])
        if gv["trans15"]:
            self.trans15x=CircleTensor(bufdict=bufdict["trans15x"])
        if gv["trans1data"]:
            self.trans1x=CircleTensor(bufdict=bufdict["trans1x"])
        if gv["reportdata"]:
            self.rdatas=CircleTensor(bufdict=bufdict["rdatas"])

    def updateins(self, dayx, x,instm, timeidx):
        dayxx, xx = self.getcudadata(dayx, x)
        self.x.update(xx, timeidx)
        
    def appendins(self, dd, s, e):
        if self.x.e >= e:
            return
        dayx, x, instm=dd["dayxx"], dd["minxx"], dd["tshis"]

        x=torch.from_numpy(x.copy())
        dayx=torch.from_numpy(dayx.copy())
        
        cv=x[:,:,gv["minoff"]:].float()
        x=cv
        x[x.isinf()]=torch.nan
        x=x.clamp(-gv["fp16clamp"], gv["fp16clamp"])
        
        self.tmseq = np.append(self.tmseq, instm)
        self.tmmap={}
        for i in range(self.tmseq.shape[0]):
            self.tmmap[self.tmseq[i]]=i
        self.x.append(x, s)
        if dayx.shape[0]>0:
            dcv=dayx[:,:,gv["dayoff"]:].float()
            dayx=dcv
            dayx[dayx.isinf()]=torch.nan
            dayx=dayx.clamp(-gv["fp16clamp"], gv["fp16clamp"])
            self.dayx.append(dayx, int(s/gv["insperday"]))
        if gv["trans15"]:
            trans15x=dd["trans15xx"]
            trans15x=torch.from_numpy(trans15x.copy())
            if trans15x.shape[0]>0:
                trans15x=trans15x[:,:,:].float()
                trans15x[trans15x.isinf()]=torch.nan
                trans15x=trans15x.clamp(-gv["fp16clamp"], gv["fp16clamp"])
                self.trans15x.append(trans15x, int(s/gv["min2t15"]))
        if gv["trans1data"]:
            trans1x=dd["trans1xx"]
            trans1x=torch.from_numpy(trans1x.copy())
            if trans1x.shape[0]>0:
                trans1x=trans1x[:,:,:].float()
                trans1x[trans1x.isinf()]=torch.nan
                trans1x=trans1x.clamp(-gv["fp16clamp"], gv["fp16clamp"])
                self.trans1x.append(trans1x, int(s/gv["min2t1"]))
        if gv["reportdata"]:
            rdatas=dd["rdatas"]
            rdatas=torch.from_numpy(rdatas.copy())
            if rdatas.shape[0]>0:
                rdatas[rdatas.isinf()]=torch.nan
                self.rdatas.append(rdatas, int(s/gv["insperday"]))
        
    def getins(self, s,e=0,delay=1):
        if e==0:
            e=s+1
        x=self.x.getdata(s, e)
        y=self.x.getdata(s+delay, e)
        return x,y
    def getx(self, s,e=0,delay=1):
        if e==0:
            e=s+1
        ds=int(s/gv["insperday"])
        de=int(e/gv["insperday"])
        ds=ds if de>ds else de-1
        # reduce mem trans
        starts=max(e-gv["insperday"]*(gv["headdays"]+1), s)
        x=self.x.getdata(starts, e)
        # x=self.x.getdata(s, e)
        dayx=self.dayx.getdata(ds,de)
        if gv["usesmallstocks"]:
            dayx, x = dayx[:,:200],x[:,:200]
        return dayx,x

    def getdayx(self, ds,de=0,delay=1):
        if de==0:
            de=ds+1
        ds=ds if de>ds else de-1
        dayx=self.dayx.getdata(ds,de)
        if gv["usesmallstocks"]:
            dayx=dayx[:,:200]
        return dayx

    def getminx(self, s,e=0,delay=1):
        if e==0:
            e=s+1
        x=self.x.getdata(s, e)
        if gv["usesmallstocks"]:
            x=x[:,:200]
        return x

    def gettrans15x(self, s,e=0,delay=1):
        if e==0:
            e=s+1
        trans15x=self.trans15x.getdata(s,e)
        if gv["usesmallstocks"]:
            trans15x=trans15x[:,:200]
        return trans15x
    def gettrans1x(self, s,e=0,delay=1):
        if e==0:
            e=s+1
        trans1x=self.trans1x.getdata(s,e)
        if gv["usesmallstocks"]:
            trans1x=trans1x[:,:200]
        return trans1x
    def getreportdata(self, s,e=0,delay=1):
        if e==0:
            e=s+1
        rdata=self.rdatas.getdata(s,e)
        if gv["usesmallstocks"]:
            rdata=rdata[:,:200]
        return rdata
    
    def gety(self, s,e=0,delay=1,reduce=1):
        if e==0:
            e=s+reduce
        y=self.x.getdata(s+delay, e+delay)
        if reduce > 1:
            y=y.mean(dim=0).unsqueeze(0)
        if gv["usesmallstocks"]:
            y=y[:,:200]
        return y
    def getdaycount(self):
        return self.datax.shape[0]
    
tm=0  
class CIcIns:
    #be careful time data
    def __init__(self, x, hislen, targetdaynum, offset=0, name=""):
        self.datax=x
        self.s=offset
        self.e=self.datax.shape[2]+self.s
        self.hislen=hislen
        self.targetdaynum=targetdaynum
        self.std=self.datax.flatten().std().clamp(gv["eps"])
        self.name=name
        
    def getins(self, idx, detaily=False):
        assert idx >= self.s+self.hislen and idx < self.e, self.name+" ins idx error:"+str(idx)+","+str(self.s)+","+str(self.e)
        x=self.datax[:,:,(idx-self.s-self.hislen):(idx-self.s)]
        y=self.datax[:,:,(idx-self.s):(idx-self.s+self.targetdaynum)].mean(dim=-1).mean(dim=-1)/self.std
        if detaily:
            return x,y,self.datax[:,:,(idx-self.s):(idx-self.s+self.targetdaynum)].mean(dim=1)/self.std
        return x,y
    
    def getco(self, tidx):
        if tidx < self.e and tidx >= self.s:
            return self.co[:,tidx-self.s-1:tidx-self.s]
        return None
    
    def validtimeidx(self):
        return list(range(self.s+self.hislen, self.e-self.targetdaynum))
    
    def getdaycount(self):
        return self.e
    
    def getcomputese(self, ns, ne):
        if self.s<=ns and self.e > ns:
            ns=self.e
        if self.s<ne and self.e >= ne:
            ne=self.s
        return ns, ne
    
    def mergeins(self, newdata, ns, ne):
        s, e = self.s, self.e
        if ns <= e and ne > e:
            self.datax=torch.cat([self.datax, newdata[:,:,e-ns:]], dim=2)
            self.e=ne
        elif ns<s and ne >=s:
            self.datax=torch.cat([newdata[:,:,:s-ns], self.datax], dim=2)
            self.s=ns
        else:
            assert s==0 and e==0, "mergeins error"
class StatIc:
    def __init__(self, name=None, bufdict=None):
        if not bufdict is None:
            self.lastyearidx=bufdict["lastyearidx"]
            self.lastmonthidx=bufdict["lastmonthidx"]
            self.last3monthidx=bufdict["last3monthidx"]
            self.metrics=bufdict["metrics"]
            self.name=bufdict["name"]
            return 
        self.lastyearidx=None
        self.lastmonthidx=None
        self.last3monthidx=None
        self.metrics=None
        self.name=name
    def save(self):
        bufdict={}
        bufdict["lastyearidx"]=self.lastyearidx
        bufdict["lastmonthidx"]=self.lastmonthidx
        bufdict["last3monthidx"]=self.last3monthidx
        bufdict["metrics"]=self.metrics
        bufdict["name"]=self.name
        return bufdict
    def select_dtypes(self, df, types):
        selectclos=[]
        for colname in df.columns:
             if str(df[colname].dtype) in types:
                 selectclos.append(colname)
        return df[selectclos]
    def printstat(self, dur, begin, end, metric):
        # pstr=metric.select_dtypes(include=[int, float]).mean(0).to_dict()
        pstr=self.select_dtypes(metric, ["int", "float", "int32", "float32", "int64", "float64"]).mean(0).to_dict()
        flog("stats ic info",self.name,dur,"begin:", sk.gtm_i(begin), "end:", sk.gtm_i(end), pstr)
        # pstr=metric.select_dtypes(include=[int, float]).groupby('tiddur').agg("mean").to_dict()
        pstr=self.select_dtypes(metric, ["int", "float", "int32", "float32", "int64", "float64"]).groupby('tiddur').agg("mean").to_dict()
        if gv["printstd"]:
            stdpstr=metric[["tiddur", "ic"]].groupby('tiddur').agg("std").to_dict()
            stdd={}
            for key in stdpstr:
                stdd[key+"_std"]=stdpstr[key]
            pstr.update(stdd)
        flog("stats ic info tidx ",self.name,dur,"begin:", sk.gtm_i(begin), "end:", sk.gtm_i(end), pstr)
    def getlastinfo(self, tidx, num):
        if self.metrics is None or len(self.metrics) < 20:
            return None
        idxlist=self.metrics[self.metrics.tidx==tidx].index.tolist()
        if len(idxlist) <=0:
            return None
        off=idxlist[0]
        num=max(off-num, 0)
        metric=self.metrics[num:off+1]
        return self.select_dtypes(metric, ["int", "float", "int32", "float32", "int64", "float64"]).mean(0).to_dict()
    def update(self, metrics, nowidx):
        if self.lastyearidx is None:
            self.lastyearidx=nowidx
            self.lastmonthidx=nowidx
            self.last3monthidx=nowidx
            self.lastdayidx=nowidx
            self.lasthouridx=nowidx
            self.metrics=metrics
            return
        self.metrics=self.metrics._append(metrics, ignore_index = True)
        lastyeardate=inttime2date(sk.gtm_i(self.lastyearidx),"%Y-%m-%dT%H:%M:%S")
        lastmonthdate=inttime2date(sk.gtm_i(self.lastmonthidx),"%Y-%m-%dT%H:%M:%S")
        last3monthdate=inttime2date(sk.gtm_i(self.last3monthidx),"%Y-%m-%dT%H:%M:%S")
        lastdaydate=inttime2date(sk.gtm_i(self.lastdayidx),"%Y-%m-%dT%H:%M:%S")
        if "lasthouridx" not in dir(self):
            self.lasthouridx=self.lastdayidx
        lasthourdate=inttime2date(sk.gtm_i(self.lasthouridx),"%Y-%m-%dT%H:%M:%S")
        
        nowdt=inttime2date(sk.gtm_i(nowidx),"%Y-%m-%dT%H:%M:%S")
        
        if nowdt.hour != lasthourdate.hour:
            self.printstat("hour", self.lasthouridx, nowidx, 
                           self.metrics[(self.metrics["tidx"]>=self.lasthouridx) & (self.metrics["tidx"]<nowidx)])
            self.lasthouridx=nowidx
        if nowdt.day != lastdaydate.day:
            self.printstat("day", self.lastdayidx, nowidx, self.metrics[(self.metrics["tidx"]>=self.lastdayidx) & (self.metrics["tidx"]<nowidx)])
            self.lastdayidx=nowidx
        if nowdt.year > lastmonthdate.year and nowdt.month == 1 or             nowdt.month > lastmonthdate.month:
            self.printstat("month", self.lastmonthidx, nowidx, self.metrics[(self.metrics["tidx"]>=self.lastmonthidx) & (self.metrics["tidx"]<nowidx)])
            self.lastmonthidx=nowidx
        if nowdt.year > last3monthdate.year and nowdt.month == 1 or             (nowdt.month - last3monthdate.month)==3:
            self.printstat("month3", self.last3monthidx, nowidx, self.metrics[(self.metrics["tidx"]>=self.last3monthidx) & (self.metrics["tidx"]<nowidx)])
            self.last3monthidx=nowidx
        if nowdt.year > lastyeardate.year:
            self.printstat("year", self.lastyearidx, nowidx, self.metrics[(self.metrics["tidx"]>=self.lastyearidx) & (self.metrics["tidx"]<nowidx)])
            self.lastyearidx=nowidx

def inttime2date(inttm, fmt):
    ss=inttime2str(inttm)
    return datetime.datetime.strptime(ss, fmt)

def strtime2int(tm):
    tm=tm.replace(" ","")
    tm=tm.replace("-","")
    tm=tm.replace(":","")
    return int(tm)
def inttime2str(start_date):
    start_date=str(start_date)
    start_date=start_date[:4]+"-"+start_date[4:6]+"-"+start_date[6:8]+"T"+start_date[8:10]+":"+start_date[10:12]+":"+start_date[12:14]
    return start_date

def hashtensor(x):
    if not (type(x) is torch.Tensor):
        x=torch.from_numpy(x).cuda()
    x=x.flatten()  
    delta=max(1,int(x.shape[0]/1000))
    hashcode=str(x.mean().item())+"_"+str(x.std().item())+"_"+str(x.median().item())
    return hashcode
def adddict(d, prefix, xoff, keys, values):
    for i in range(len(keys)):
        key, value = keys[i], values[i]
        d[prefix+key]=[value[xoff].item()]
class OlInffer:   
    def __init__(self, name, testfeaday=-1, bufdict=None, calctop=False):
        if not bufdict is None:
            outsize=1
            # self.lastqs, self.lastws=bufdict["lastqs"], bufdict["lastws"]
            # self.lastwpos=bufdict["lastwpos"]
            # self.tidx=bufdict["tidx"]
            # self.icdiff=bufdict["icdiff"]
            # self.sinfo=StatIc(bufdict=bufdict["sinfo"])
            # self.testfeaday=bufdict["testfeaday"]
            # self.xx=bufdict["xx"]
            # self.vfs=bufdict["vfs"]
            # self.yy=bufdict["yy"]
            # self.xetr=bufdict["xetr"]
            return
        self.name=name
        outsize=1
        self.tidx=0
        self.icdiff=[]
        self.sinfo=StatIc(name)
        self.testfeaday=testfeaday
        self.xx, self.vfs, self.yy={}, {}, {}
        self.xetr={}
        self.calctop=calctop
        
    def inffer(self, tmodel, tidx):
        tmodel.eval()
        output, vf, xetr = tmodel.pred(tidx)
        tmodel.train()
        #torch.cuda.empty_cache()
        return output.cpu(), vf.cpu(), xetr
    def infferandstats(self, tmodel, tidx):
        self.addx(tmodel, tidx)
        self.calcstats(tmodel, tidx)
        #torch.cuda.empty_cache()
    
    def save(self):
        bufdict={}
        bufdict["tidx"]=self.tidx
        bufdict["icdiff"]=self.icdiff
        bufdict["sinfo"]=self.sinfo.save()
        bufdict["testfeaday"]=self.testfeaday
        bufdict["xx"]=self.xx
        bufdict["vfs"]=self.vfs
        bufdict["yy"]=self.yy
        bufdict["xetr"]=self.xetr
        return bufdict
    def addx(self, tmodel, tidx):
        x, vf, xetr = self.inffer(tmodel, tidx)
        self.xx[tidx]=x.cpu()
        self.vfs[tidx]=vf.cpu()
        self.xetr[tidx]=xetr
        #torch.cuda.empty_cache()
        return x.cpu(), vf.cpu(), xetr

    def calcstats(self, tmodel, tidx):
        if not (tidx in self.xx):
            flog("day:",tidx, sk.gtm_i(tidx), "tidx:",tidx,":"+self.name+" x y incomplete")
            return

        lossres=tmodel.calcloss(tidx, self.xx[tidx].to(tmodel.device), self.vfs[tidx].to(tmodel.device), self.xetr[tidx], calcstats=True)
        # statsmat=lossres["statsmat"]
        # del lossres["statsmat"]
        df=dict(date=[sk.gtm_i(tidx)], tidx=[tidx], tiddur=tidx%gv["tiddurcnt"])
        for key in lossres.keys():
            df[key]=[lossres[key].detach().item()]
        
        df=pd.DataFrame(df)
        flog("day:",tidx, sk.gtm_i(tidx), "tidx:",tidx,":"+self.name+" test:", df.to_dict())
        tvroffset=gv["tmdelta"]*gv["tvrdeltas"] 
        if tidx-tvroffset in self.xx:
            del self.xx[tidx-tvroffset], self.vfs[tidx-tvroffset]

        self.sinfo.update(df, tidx)
        return df
    
    def featest(self, imodel,ins,tidx):
        mcflag=False


def meminfo(deviceid):
    sizescale=1024*1024*1024
    print("allocated:", torch.cuda.memory_allocated(deviceid)/sizescale,
          "maxallocated:", torch.cuda.max_memory_allocated(deviceid)/sizescale,
          "reserverd:", torch.cuda.memory_reserved(deviceid)/sizescale)
    
def _is_equal(x: torch.Tensor, y: torch.Tensor, tol=1e-2):
    diff = abs(x - y)
    x_max = torch.max(x)
    y_max = torch.max(y)
    err = torch.max(diff) / torch.max(x_max, y_max)
    return err <= tol

if __name__ == "__main__":
    deviceid=0
    a=torch.randn(10240,1024, device=deviceid)
    cssm1=CssModel(1024,512).to(deviceid)
    cssm2=CssModel(512,512).to(deviceid)
    b=cssm1(a)
    c=cssm2(b)

