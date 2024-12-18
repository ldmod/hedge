#!/usr/bin/env python
# coding=utf-8
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 21:15:11 2021

@author: ld


"""
# Import libraries
import time
import datetime
import torch
import torch.nn as nn # 
from torch.optim import SGD # 
import torch.utils.data as Data # 
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
import pandas as pd
from functools import reduce
import cryptoqt.smlp.alpha.tools as tools
import cryptoqt.smlp.alpha.seqfea as seqfea
import cryptoqt.smlp.alpha.common as common
import cryptoqt.smlp.alpha.mergemodel as mmodel
import cryptoqt.smlp.alpha.yamlcfg as yamlcfg
from cryptoqt.smlp.alpha.yamlcfg import gv
import h5py


def flog(*args,**kwargs):
    print(*args,**kwargs)
    sys.stdout.flush()

def initmodel(data_dict, path=None):
    pred=mmodel.PredM()
    pred.lastdt=None
    step=gv["step"]
    
    if not path is None:
        modeldate=pred.hotstart(path, False)
    else:
        pred.coldstart()
    
    return pred  

def initcfg(yamlcfgpath, rseed=1990):
    # The flag below controls whether to allow TF32 on matmul. This flag defaults to False
    # in PyTorch 1.12 and later.
    torch.backends.cuda.matmul.allow_tf32 = True
    # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
    torch.backends.cudnn.allow_tf32 = True
    
    # setcfg(gv)
    yamlcfg.loadcfg(yamlcfgpath)
    global contents

    # os.environ['CUDA_VISIBLE_DEVICES'] = gv["did"]
    os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    # gv["gpus"]=list(range(len(gv["did"].split(','))))
    # gv["gpus"]=[int(item) for item in gv["did"].split(',')]
    # if len(gv["gpus"])<2:
    #     gv["gpus"].append(0)

    print(torch.cuda.device_count())

    
    torch.set_printoptions(
        precision=5,    #
        threshold=1000,
        edgeitems=3,
        linewidth=150,  #
        profile=None,
        sci_mode=False  #
    )
    np.set_printoptions(
        precision=5,    #
    threshold=1000,
    edgeitems=3,
    linewidth=150,  #
    suppress=False  #
    )
    
    rseed=gv["seed"] if "seed" in gv else(time.monotonic_ns()%100000)
    tools.setup_seed(rseed)
    
    gv["suffix"]=str(gv["did"]).replace(',', '_')+'olldvm_d1'+"_"+str(os.getpid())+"_"+\
        gv["optobj"]+gv['insname']+str(hash(str(gv)))
    if "logname" in gv:
        logfilename=gv["logpath"]+"/"+gv["logname"]
    else:
        logfilename=gv["logpath"]+"/"+gv["suffix"]
    logfile=open(logfilename, 'a',encoding='utf-8')
    # gv["stdoutbak"]=sys.stdout
    sys.stdout=logfile
    with open(yamlcfgpath, 'r', encoding='unicode_escape') as file_obj:
        contents = file_obj.read()
    flog(contents)
    fpath=os.path.dirname(__file__)+"/"+"../alpha/mergemodel.py"
    if os.path.exists(fpath):
        with open(fpath, 'r', encoding='unicode_escape') as file_obj:
            contents = file_obj.read()
        flog(contents)
    if not "online" in gv.keys():
        
        with open(os.path.dirname(__file__)+"/"+"../alpha/canon.py", 'r', encoding='unicode_escape') as file_obj:
            contents = file_obj.read()
        flog(contents)
        
        with open(os.path.dirname(__file__)+"/"+"../alpha/data2ins.py", 'r', encoding='unicode_escape') as file_obj:
            contents = file_obj.read()
        flog(contents)
    
    flog("cfg:", gv)
    flog("seed:", rseed)
    
    
def pconfig():
    modelname=gv["modelname"] if "modelname" in gv else gv["suffix"] 
    gv["modelpath"]=os.path.dirname(__file__)+"/"+gv["modelpath"]+"/"+modelname+"/"
    os.makedirs(gv["modelpath"], exist_ok=True)
    gv["model_path"]=gv["modelpath"]+"/"+"savemodel"
    os.makedirs(gv["modelpath"]+"/res", exist_ok=True)
    
    os.makedirs("./logs/", exist_ok=True)

    # flog("set cfg:", gv)
    for key in gv:
        flog(key, ":", gv[key])


# if __name__ == "__main__":
#     setcfg()
