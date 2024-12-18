#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 13:32:20 2024

@author: ld
"""
import numpy as np
import cryptoqt.data.constants as conts
daymincnt=conts.daymincnt
h4mincnt=conts.h4mincnt
h1mincnt=conts.h1mincnt
min15mincnt=conts.min15mincnt

def nh4r5(dr, min1i):
    di=int(min1i/conts.daymincnt)
    h4i=int(min1i/conts.h4mincnt)
    h1i=int(min1i/conts.h1mincnt)
    min15i=int(min1i/conts.min15mincnt) 
    delay=5
    dclose=dr["h4info_close"]
    ret5day=dclose[h4i-1]/dclose[h4i-delay-1]-1.0
    alpha=ret5day-np.nanmean(ret5day)
    alpha=alpha/np.nanstd(alpha)
    return -1.0*alpha


def ndayr5(dr, min1i):
    di=int(min1i/conts.daymincnt)
    h4i=int(min1i/conts.h4mincnt)
    h1i=int(min1i/conts.h1mincnt)
    min15i=int(min1i/conts.min15mincnt) 
    delay=1
    # di=di+1
    dclose=dr["dayinfo_close"]
    ret5day=dclose[di-1]/dclose[di-delay-1]-1.0
    alpha=ret5day-np.nanmean(ret5day)
    alpha=alpha/np.nanstd(alpha)
    return -1.0*alpha