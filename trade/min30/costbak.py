#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 12:43:31 2024

@author: prod
"""
import numpy as np
np.seterr(invalid='ignore')
from functools import partial
from audtorch.metrics.functional import pearsonr
import torch
import cryptoqt.data.constants as conts
import cryptoqt.data.sec_klines.sec_klines as sk 
import h5py
import cryptoqt.data.updatedata as ud
from binance.um_futures import UMFutures
import pandas as pd
pd.options.mode.chained_assignment = None
import cryptoqt.data.tools as tools
import time
import random
import copy
from multiprocessing import Process,Lock
import os
import cryptoqt.data.data_manager as dm
delaysec=2
global_mutex = Lock()
# 获得最小交易限额(增加了10%的Buffer)
def get_threshold(symbol):
    values = {
        "BTCUSDT": 110,
        "ETHUSDT": 22,
        "BCHUSDT": 22,
        "LTCUSDT": (22),
        "ETCUSDT": (22),
        "LINKUSDT": (22),
        "SOLUSDT": 160,
        "MKRUSDT": 22,
        "AAVEUSDT": 22,
    }
    return max(5, values.get(symbol, (6.5)))+1.0

def load_ppm(path="/home/crypto/proddata/crypto/universe/"):
    global price_precision_map, quantity_precision_map
    price_precision_map=torch.load(path+"/price_precision_map")
    quantity_precision_map=torch.load(path+"/quantity_precision_map")
    
def update_ppm(path="/home/crypto/proddata/crypto/universe/"):
    myproxies = {
            'http': 'http://127.0.0.1:20890',
            'https': 'http://127.0.0.1:20890'
    }

    timeout=60
    um_futures_client = UMFutures(proxies=myproxies, timeout=timeout)
    exchange_info=um_futures_client.exchange_info()
    price_precision_map={}
    quantity_precision_map={}
    for symbolinfo in exchange_info["symbols"]:
        symbol=symbolinfo["symbol"]
        price_precision_map[symbol]=symbolinfo['pricePrecision']
        quantity_precision_map[symbol]=symbolinfo["quantityPrecision"]
    torch.save(price_precision_map, path+"/price_precision_map")
    torch.save(quantity_precision_map, path+"/quantity_precision_map")
    
# load_ppm()   
price_precision_map = {'BTCUSDT': 1, 'ETHUSDT': 2, 'BCHUSDT': 2, 'XRPUSDT': 4, 'EOSUSDT': 3, 'LTCUSDT': 2,
                           'TRXUSDT': 5, 'ETCUSDT': 3, 'LINKUSDT': 3, 'XLMUSDT': 5, 'ADAUSDT': 4, 'XMRUSDT': 2,
                           'DASHUSDT': 2, 'ZECUSDT': 2, 'XTZUSDT': 3, 'BNBUSDT': 2, 'ATOMUSDT': 3, 'ONTUSDT': 4,
                           'IOTAUSDT': 4, 'BATUSDT': 4, 'VETUSDT': 6, 'NEOUSDT': 3, 'QTUMUSDT': 3, 'IOSTUSDT': 6,
                           'THETAUSDT': 4, 'ALGOUSDT': 4, 'ZILUSDT': 5, 'KNCUSDT': 4, 'ZRXUSDT': 4, 'COMPUSDT': 2,
                           'OMGUSDT': 4, 'DOGEUSDT': 5, 'SXPUSDT': 4, 'KAVAUSDT': 4, 'BANDUSDT': 4, 'RLCUSDT': 4,
                           'WAVESUSDT': 4, 'MKRUSDT': 1, 'SNXUSDT': 3, 'DOTUSDT': 3, 'DEFIUSDT': 1, 'YFIUSDT': 0,
                           'BALUSDT': 3, 'CRVUSDT': 3, 'TRBUSDT': 3, 'RUNEUSDT': 3, 'SUSHIUSDT': 4, 'EGLDUSDT': 3,
                           'SOLUSDT': 3, 'ICXUSDT': 4, 'STORJUSDT': 4, 'BLZUSDT': 5, 'UNIUSDT': 3, 'AVAXUSDT': 3,
                           'FTMUSDT': 4, 'ENJUSDT': 5, 'FLMUSDT': 4, 'RENUSDT': 5, 'KSMUSDT': 3, 'NEARUSDT': 3,
                           'AAVEUSDT': 2, 'FILUSDT': 3, 'RSRUSDT': 6, 'LRCUSDT': 5, 'MATICUSDT': 4, 'OCEANUSDT': 4,
                           'CVCUSDT': 5, 'BELUSDT': 4, 'CTKUSDT': 4, 'AXSUSDT': 3, 'ALPHAUSDT': 5, 'ZENUSDT': 3,
                           'SKLUSDT': 5, 'GRTUSDT': 5, '1INCHUSDT': 4, 'CHZUSDT': 5, 'SANDUSDT': 4, 'ANKRUSDT': 5,
                           'LITUSDT': 3, 'UNFIUSDT': 3, 'REEFUSDT': 6, 'RVNUSDT': 5, 'SFPUSDT': 4, 'XEMUSDT': 4,
                           'BTCSTUSDT': 3, 'COTIUSDT': 5, 'CHRUSDT': 4, 'MANAUSDT': 4, 'ALICEUSDT': 3, 'HBARUSDT': 5,
                           'ONEUSDT': 5, 'LINAUSDT': 5, 'STMXUSDT': 5, 'DENTUSDT': 6, 'CELRUSDT': 5, 'HOTUSDT': 6,
                           'MTLUSDT': 4, 'OGNUSDT': 4, 'NKNUSDT': 5, 'SCUSDT': 6, 'DGBUSDT': 5, '1000SHIBUSDT': 6,
                           'BAKEUSDT': 4, 'GTCUSDT': 3, 'BTCDOMUSDT': 1, 'IOTXUSDT': 5, 'RAYUSDT': 3, 'C98USDT': 4,
                           'MASKUSDT': 4, 'ATAUSDT': 4, 'DYDXUSDT': 3, '1000XECUSDT': 5, 'GALAUSDT': 5, 'CELOUSDT': 3,
                           'ARUSDT': 3, 'KLAYUSDT': 4, 'ARPAUSDT': 5, 'CTSIUSDT': 4, 'LPTUSDT': 3, 'ENSUSDT': 3,
                           'PEOPLEUSDT': 5, 'ROSEUSDT': 5, 'DUSKUSDT': 5, 'FLOWUSDT': 3, 'IMXUSDT': 4, 'API3USDT': 4,
                           'GMTUSDT': 5, 'APEUSDT': 4, 'WOOUSDT': 5, 'FTTUSDT': 4, 'JASMYUSDT': 6, 'DARUSDT': 4,
                           'OPUSDT': 4, 'INJUSDT': 3, 'STGUSDT': 4, 'SPELLUSDT': 7, '1000LUNCUSDT': 5, 'LUNA2USDT': 4,
                           'LDOUSDT': 4, 'CVXUSDT': 3, 'ICPUSDT': 3, 'APTUSDT': 3, 'QNTUSDT': 2, 'FETUSDT': 4,
                           'FXSUSDT': 4, 'HOOKUSDT': 4, 'MAGICUSDT': 4, 'TUSDT': 5, 'HIGHUSDT': 4, 'MINAUSDT': 4,
                           'ASTRUSDT': 5, 'AGIXUSDT': 4, 'PHBUSDT': 4, 'GMXUSDT': 3, 'CFXUSDT': 5, 'STXUSDT': 4,
                           'BNXUSDT': 4, 'ACHUSDT': 6, 'SSVUSDT': 3, 'CKBUSDT': 6, 'PERPUSDT': 4, 'TRUUSDT': 5,
                           'LQTYUSDT': 4, 'USDCUSDT': 6, 'IDUSDT': 5, 'ARBUSDT': 4, 'JOEUSDT': 4, 'TLMUSDT': 6,
                           'AMBUSDT': 6, 'LEVERUSDT': 7, 'RDNTUSDT': 5, 'HFTUSDT': 5, 'XVSUSDT': 3, 'BLURUSDT': 4,
                           'EDUUSDT': 4, 'IDEXUSDT': 5, 'SUIUSDT': 4, '1000PEPEUSDT': 7, '1000FLOKIUSDT': 5,
                           'UMAUSDT': 3, 'RADUSDT': 4, 'KEYUSDT': 6, 'COMBOUSDT': 4, 'NMRUSDT': 3, 'MAVUSDT': 5,
                           'MDTUSDT': 5, 'XVGUSDT': 6, 'WLDUSDT': 4, 'PENDLEUSDT': 4, 'ARKMUSDT': 4, 'AGLDUSDT': 4,
                           'YGGUSDT': 4, 'DODOXUSDT': 6, 'BNTUSDT': 5, 'OXTUSDT': 5, 'SEIUSDT': 4, 'CYBERUSDT': 3,
                           'HIFIUSDT': 4, 'ARKUSDT': 4, 'FRONTUSDT': 4, 'GLMRUSDT': 5, 'BICOUSDT': 4, 'STRAXUSDT': 4,
                           'LOOMUSDT': 5, 'BIGTIMEUSDT': 4, 'BONDUSDT': 3, 'ORBSUSDT': 5, 'STPTUSDT': 5, 'WAXPUSDT': 5,
                           'BSVUSDT': 2, 'RIFUSDT': 5, 'POLYXUSDT': 5, 'GASUSDT': 3, 'POWRUSDT': 4, 'SLPUSDT': 6,
                           'TIAUSDT': 4, 'SNTUSDT': 5, 'CAKEUSDT': 4, 'MEMEUSDT': 6, 'TWTUSDT': 4, 'TOKENUSDT': 5,
                           'ORDIUSDT': 3, 'STEEMUSDT': 5, 'BADGERUSDT': 4, 'ILVUSDT': 2, 'NTRNUSDT': 4, 'KASUSDT': 5,
                           'BEAMXUSDT': 6, '1000BONKUSDT': 6, 'PYTHUSDT': 4, 'SUPERUSDT': 4, 'USTCUSDT': 5,
                           'ONGUSDT': 5, 'ETHWUSDT': 4, 'JTOUSDT': 4, '1000SATSUSDT': 7, 'AUCTIONUSDT': 3,
                           '1000RATSUSDT': 5, 'ACEUSDT': 4, 'MOVRUSDT': 3, 'NFPUSDT': 4, 'AIUSDT': 5, 'XAIUSDT': 4,
                           'WIFUSDT': 4, 'MANTAUSDT': 4, 'ONDOUSDT': 4, 'LSKUSDT': 4, 'ALTUSDT': 5, 'JUPUSDT': 4,
                           'ZETAUSDT': 4, 'RONINUSDT': 4, 'DYMUSDT': 4, 'OMUSDT': 5, 'PIXELUSDT': 4, 'STRKUSDT': 4,
                           'MAVIAUSDT': 4, 'GLMUSDT': 4, 'PORTALUSDT': 4, 'TONUSDT': 4, 'AXLUSDT': 4, 'MYROUSDT': 5,
                           'METISUSDT': 2, 'AEVOUSDT': 4, 'VANRYUSDT': 5, 'BOMEUSDT': 6, 'ETHFIUSDT': 3, 'ENAUSDT': 4,
                           'WUSDT': 4, 'TNSRUSDT': 4, 'SAGAUSDT': 4, 'TAOUSDT': 2, 'OMNIUSDT': 3, 'REZUSDT': 5,
                           'BBUSDT': 4, 'NOTUSDT': 6, 'TURBOUSDT': 6, 'IOUSDT': 3, 'ZKUSDT': 5, 'MEWUSDT': 6,
                           'LISTAUSDT': 4, 'ZROUSDT': 3, 'RENDERUSDT': 3} 

quantity_precision_map = {'BTCUSDT': 3, 'ETHUSDT': 3, 'BCHUSDT': 3, 'XRPUSDT': 1, 'EOSUSDT': 1, 'LTCUSDT': 3,
                              'TRXUSDT': 0, 'ETCUSDT': 2, 'LINKUSDT': 2, 'XLMUSDT': 0, 'ADAUSDT': 0, 'XMRUSDT': 3,
                              'DASHUSDT': 3, 'ZECUSDT': 3, 'XTZUSDT': 1, 'BNBUSDT': 2, 'ATOMUSDT': 2, 'ONTUSDT': 1,
                              'IOTAUSDT': 1, 'BATUSDT': 1, 'VETUSDT': 0, 'NEOUSDT': 2, 'QTUMUSDT': 1, 'IOSTUSDT': 0,
                              'THETAUSDT': 1, 'ALGOUSDT': 1, 'ZILUSDT': 0, 'KNCUSDT': 0, 'ZRXUSDT': 1, 'COMPUSDT': 3,
                              'OMGUSDT': 1, 'DOGEUSDT': 0, 'SXPUSDT': 1, 'KAVAUSDT': 1, 'BANDUSDT': 1, 'RLCUSDT': 1,
                              'WAVESUSDT': 1, 'MKRUSDT': 3, 'SNXUSDT': 1, 'DOTUSDT': 1, 'DEFIUSDT': 3, 'YFIUSDT': 3,
                              'BALUSDT': 1, 'CRVUSDT': 1, 'TRBUSDT': 1, 'RUNEUSDT': 0, 'SUSHIUSDT': 0, 'EGLDUSDT': 1,
                              'SOLUSDT': 0, 'ICXUSDT': 0, 'STORJUSDT': 0, 'BLZUSDT': 0, 'UNIUSDT': 0, 'AVAXUSDT': 0,
                              'FTMUSDT': 0, 'ENJUSDT': 0, 'FLMUSDT': 0, 'RENUSDT': 0, 'KSMUSDT': 1, 'NEARUSDT': 0,
                              'AAVEUSDT': 1, 'FILUSDT': 1, 'RSRUSDT': 0, 'LRCUSDT': 0, 'MATICUSDT': 0, 'OCEANUSDT': 0,
                              'CVCUSDT': 0, 'BELUSDT': 0, 'CTKUSDT': 0, 'AXSUSDT': 0, 'ALPHAUSDT': 0, 'ZENUSDT': 1,
                              'SKLUSDT': 0, 'GRTUSDT': 0, '1INCHUSDT': 0, 'CHZUSDT': 0, 'SANDUSDT': 0, 'ANKRUSDT': 0,
                              'LITUSDT': 1, 'UNFIUSDT': 1, 'REEFUSDT': 0, 'RVNUSDT': 0, 'SFPUSDT': 0, 'XEMUSDT': 0,
                              'COTIUSDT': 0, 'CHRUSDT': 0, 'MANAUSDT': 0, 'ALICEUSDT': 1, 'HBARUSDT': 0, 'ONEUSDT': 0,
                              'LINAUSDT': 0, 'STMXUSDT': 0, 'DENTUSDT': 0, 'CELRUSDT': 0, 'HOTUSDT': 0, 'MTLUSDT': 0,
                              'OGNUSDT': 0, 'NKNUSDT': 0, 'SCUSDT': 0, 'DGBUSDT': 0, '1000SHIBUSDT': 0, 'BAKEUSDT': 0,
                              'GTCUSDT': 1, 'BTCDOMUSDT': 3, 'IOTXUSDT': 0, 'RAYUSDT': 1, 'C98USDT': 0, 'MASKUSDT': 0,
                              'ATAUSDT': 0, 'DYDXUSDT': 1, '1000XECUSDT': 0, 'GALAUSDT': 0, 'CELOUSDT': 1, 'ARUSDT': 1,
                              'KLAYUSDT': 1, 'ARPAUSDT': 0, 'CTSIUSDT': 0, 'LPTUSDT': 1, 'ENSUSDT': 1, 'PEOPLEUSDT': 0,
                              'ROSEUSDT': 0, 'DUSKUSDT': 0, 'FLOWUSDT': 1, 'IMXUSDT': 0, 'API3USDT': 1, 'GMTUSDT': 0,
                              'APEUSDT': 0, 'WOOUSDT': 0, 'FTTUSDT': 1, 'JASMYUSDT': 0, 'DARUSDT': 1, 'OPUSDT': 1,
                              'INJUSDT': 1, 'STGUSDT': 0, 'SPELLUSDT': 0, '1000LUNCUSDT': 0, 'LUNA2USDT': 0,
                              'LDOUSDT': 0, 'CVXUSDT': 0, 'ICPUSDT': 0, 'APTUSDT': 1, 'QNTUSDT': 1, 'FETUSDT': 0,
                              'FXSUSDT': 1, 'HOOKUSDT': 1, 'MAGICUSDT': 1, 'TUSDT': 0, 'HIGHUSDT': 1, 'MINAUSDT': 0,
                              'ASTRUSDT': 0, 'AGIXUSDT': 0, 'PHBUSDT': 0, 'GMXUSDT': 2, 'CFXUSDT': 0, 'STXUSDT': 0,
                              'BNXUSDT': 1, 'ACHUSDT': 0, 'SSVUSDT': 2, 'CKBUSDT': 0, 'PERPUSDT': 1, 'TRUUSDT': 0,
                              'LQTYUSDT': 1, 'USDCUSDT': 0, 'IDUSDT': 0, 'ARBUSDT': 1, 'JOEUSDT': 0, 'TLMUSDT': 0,
                              'AMBUSDT': 0, 'LEVERUSDT': 0, 'RDNTUSDT': 0, 'HFTUSDT': 0, 'XVSUSDT': 1, 'BLURUSDT': 0,
                              'EDUUSDT': 0, 'IDEXUSDT': 0, 'SUIUSDT': 1, '1000PEPEUSDT': 0, '1000FLOKIUSDT': 0,
                              'UMAUSDT': 0, 'RADUSDT': 0, 'KEYUSDT': 0, 'COMBOUSDT': 1, 'NMRUSDT': 1, 'MAVUSDT': 0,
                              'MDTUSDT': 0, 'XVGUSDT': 0, 'WLDUSDT': 0, 'PENDLEUSDT': 0, 'ARKMUSDT': 0, 'AGLDUSDT': 0,
                              'YGGUSDT': 0, 'DODOXUSDT': 0, 'BNTUSDT': 0, 'OXTUSDT': 0, 'SEIUSDT': 0, 'CYBERUSDT': 1,
                              'HIFIUSDT': 0, 'ARKUSDT': 0, 'FRONTUSDT': 0, 'GLMRUSDT': 0, 'BICOUSDT': 0, 'STRAXUSDT': 0,
                              'LOOMUSDT': 0, 'BIGTIMEUSDT': 0, 'BONDUSDT': 1, 'ORBSUSDT': 0, 'STPTUSDT': 0,
                              'WAXPUSDT': 0, 'BSVUSDT': 1, 'RIFUSDT': 0, 'POLYXUSDT': 0, 'GASUSDT': 1, 'POWRUSDT': 0,
                              'SLPUSDT': 0, 'TIAUSDT': 0, 'SNTUSDT': 0, 'CAKEUSDT': 0, 'MEMEUSDT': 0, 'TWTUSDT': 0,
                              'TOKENUSDT': 0, 'ORDIUSDT': 1, 'STEEMUSDT': 0, 'BADGERUSDT': 0, 'ILVUSDT': 1,
                              'NTRNUSDT': 0, 'KASUSDT': 0, 'BEAMXUSDT': 0, '1000BONKUSDT': 0, 'PYTHUSDT': 0,
                              'SUPERUSDT': 0, 'USTCUSDT': 0, 'ONGUSDT': 0, 'ETHWUSDT': 0, 'JTOUSDT': 0,
                              '1000SATSUSDT': 0, 'AUCTIONUSDT': 2, '1000RATSUSDT': 0, 'ACEUSDT': 2, 'MOVRUSDT': 2,
                              'NFPUSDT': 1, 'AIUSDT': 0, 'XAIUSDT': 0, 'WIFUSDT': 1, 'MANTAUSDT': 1, 'ONDOUSDT': 1,
                              'LSKUSDT': 0, 'ALTUSDT': 0, 'JUPUSDT': 0, 'ZETAUSDT': 0, 'RONINUSDT': 1, 'DYMUSDT': 1,
                              'OMUSDT': 1, 'PIXELUSDT': 0, 'STRKUSDT': 1, 'MAVIAUSDT': 1, 'GLMUSDT': 0, 'PORTALUSDT': 1,
                              'TONUSDT': 1, 'AXLUSDT': 1, 'MYROUSDT': 0, 'METISUSDT': 2, 'AEVOUSDT': 1, 'VANRYUSDT': 0,
                              'BOMEUSDT': 0, 'ETHFIUSDT': 1, 'ENAUSDT': 0, 'WUSDT': 1, 'TNSRUSDT': 1, 'SAGAUSDT': 1,
                              'TAOUSDT': 3, 'OMNIUSDT': 2, 'REZUSDT': 0, 'BBUSDT': 0, 'NOTUSDT': 0, 'TURBOUSDT': 0,
                              'IOUSDT': 1, 'ZKUSDT': 0, 'MEWUSDT': 0, 'LISTAUSDT': 0, 'ZROUSDT': 1, 'RENDERUSDT': 1,
                              'BANANAUSDT': 1, 'RAREUSDT': 0, 'GUSDT': 0, 'SYNUSDT': 0}   
    
def getyvalues(secdr, s1i, tmperiod, longterm=30, delaysec=2):
    money=secdr["s1info_money"][s1i+delaysec:s1i+tmperiod+delaysec].sum(axis=0)
    volume=secdr["s1info_volume"][s1i+delaysec:s1i+tmperiod+delaysec].sum(axis=0)
    vwap=money/volume
    vf_y=(np.isfinite(vwap))
    
    money=secdr["s1info_smoney"][s1i+delaysec:s1i+tmperiod+delaysec].sum(axis=0)
    volume=secdr["s1info_svolume"][s1i+delaysec:s1i+tmperiod+delaysec].sum(axis=0)
    svwap=money/volume
    shigh=secdr["s1info_shigh"][s1i+delaysec:s1i+tmperiod+delaysec].max(axis=0)
    slow=secdr["s1info_slow"][s1i+delaysec:s1i+tmperiod+delaysec].min(axis=0)
    vf_ys=(np.isfinite(svwap) & np.isfinite(shigh) & np.isfinite(slow))
    shigh[~vf_ys]=np.nan
    slow[~vf_ys]=np.nan
    smoney=money
    
    money=secdr["s1info_money"][s1i+delaysec:s1i+tmperiod+delaysec].sum(axis=0)-\
        secdr["s1info_smoney"][s1i+delaysec:s1i+tmperiod+delaysec].sum(axis=0)
    volume=secdr["s1info_volume"][s1i+delaysec:s1i+tmperiod+delaysec].sum(axis=0)-\
        secdr["s1info_svolume"][s1i+delaysec:s1i+tmperiod+delaysec].sum(axis=0)
    bvwap=money/volume
    bhigh=secdr["s1info_bhigh"][s1i+delaysec:s1i+tmperiod+delaysec].max(axis=0)
    blow=secdr["s1info_blow"][s1i+delaysec:s1i+tmperiod+delaysec].min(axis=0)
    vf_yb=(np.isfinite(bvwap) & np.isfinite(bhigh) & np.isfinite(blow))
    bhigh[~vf_yb]=np.nan
    blow[~vf_yb]=np.nan
    bmoney=money
    
    #
    lsvwap=secdr["s1info_smoney"][s1i+delaysec:s1i+longterm+delaysec].sum(axis=0)/\
        secdr["s1info_svolume"][s1i+delaysec:s1i+longterm+delaysec].sum(axis=0)
    ysl=(lsvwap/svwap-1.0)*10000.0
    #
    money=secdr["s1info_money"][s1i+delaysec:s1i+longterm+delaysec].sum(axis=0)-\
        secdr["s1info_smoney"][s1i+delaysec:s1i+longterm+delaysec].sum(axis=0)
    volume=secdr["s1info_volume"][s1i+delaysec:s1i+longterm+delaysec].sum(axis=0)-\
        secdr["s1info_svolume"][s1i+delaysec:s1i+longterm+delaysec].sum(axis=0)
    lbvwap=money/volume
    ybl=(lbvwap/bvwap-1.0)*10000.0
    #
    
    return vwap, vf_y, svwap, shigh, slow, vf_ys, bvwap, bhigh, blow, vf_yb, smoney, bmoney, ysl, ybl

def get_vwap(secdr, s1i, tmperiod=300):
    money=secdr["s1info_money"][s1i-tmperiod:s1i].sum(axis=0)
    volume=secdr["s1info_volume"][s1i-tmperiod:s1i].sum(axis=0)
    vwap=money/volume
    return vwap

def get_vwap_diff_std(secdr, s1i, tmperiod=300, delta=5):
    vwap_moms=[]
    for tmp_s1i in range(s1i-tmperiod, s1i, delta):
        last_vwap=get_vwap(secdr, tmp_s1i-delta, delta)
        now_vwap=get_vwap(secdr, tmp_s1i, delta)
        vwap_mom=now_vwap/last_vwap-1.0
        vwap_moms.append(vwap_mom)
    vwap_moms=np.stack(vwap_moms)
    return np.nanstd(vwap_moms, axis=0)

def get_svwap(secdr, s1i, tmperiod=60):
    money=secdr["s1info_smoney"][s1i-tmperiod:s1i].sum(axis=0)
    volume=secdr["s1info_svolume"][s1i-tmperiod:s1i].sum(axis=0)
    vwap=money/volume
    return vwap

def get_bvwap(secdr, s1i, tmperiod=60):
    money=secdr["s1info_money"][s1i-tmperiod:s1i].sum(axis=0)-\
        secdr["s1info_smoney"][s1i-tmperiod:s1i].sum(axis=0)
    volume=secdr["s1info_volume"][s1i-tmperiod:s1i].sum(axis=0)-\
        secdr["s1info_svolume"][s1i-tmperiod:s1i].sum(axis=0)
    vwap=money/volume
    return vwap

def get_smom(secdr, s1i, tmperiod=60, delta=5):
    end=get_svwap(secdr, s1i, delta)
    start=get_svwap(secdr, s1i-tmperiod, delta)
    mom=end/start-1.0
    mom = 0 if np.isnan(mom) else mom
    return mom

def get_bmom(secdr, s1i, tmperiod=60, delta=5):
    end=get_bvwap(secdr, s1i, delta)
    start=get_bvwap(secdr, s1i-tmperiod, delta)
    mom=end/start-1.0
    mom = 0 if np.isnan(mom) else mom
    return mom


def readh5res(secdr, tm, cnt=1, path="/home/crypto/smlp_prm/cryptoqt/prm/model_states/infmodel/prmv3/res/res.h5"):
    f=h5py.File(path, "r", swmr=True)
    f["tm"].refresh()
    ridx=np.where(f["tm"][:]==tm)[0]
    if ridx.shape[0] <=0 :
        f.close()
        print("read res fail:", tm, path, flush=True)
        return None
    ridx=ridx[0]
    dd={}
    for key in f.keys():
        if key != "tm":
            f[key].refresh()
            dd[key]=f[key][ridx-cnt+1:ridx+1].astype(np.float64)
    f.close()
    return dd

class GenerateOrders:
    def __init__(self, secdata, cfg):
        self.shift=cfg["shift"]
        self.expand=cfg["expand"]
        self.price_normw=cfg["price_normw"]
        self.money_normw=cfg["money_normw"]
        self.predratio=cfg["predratio"]
        self.path=cfg["path"]
        self.trw=cfg["trw"]
        self.rwp=cfg["rwp"]
        self.simcnt=cfg["simcnt"]
        self.secdata = secdata
        self.secdr=secdata.secdr
        self.symbolidx={}
        self.delta=5
        self.maxr=cfg["maxr"]
        self.ada_shift_range=cfg["ada_shift_range"]
        self.max_shift=cfg["max_shift"]
        self.tcr=cfg["tcr"]
        self.cancel_delay=cfg["cancel_delay"]
        self.save_order_info_path=cfg["save_order_info_path"]
        self.mb_tratio=np.ones((ud.g_data["sids"].shape[0]))*2
        self.ms_tratio=np.ones((ud.g_data["sids"]).shape[0])*2
        self.money_limit_ratio=cfg["money_limit_ratio"]
        self.trade_succ_ratio=cfg["trade_succ_ratio"]
        self.mr_scale=cfg["mr_scale"]
        self.bp_map=cfg["bp_map"]
        self.cfg=cfg
        for idx,sid in enumerate(ud.g_data["sids"]):
            self.symbolidx[sid]=idx
        self.order_infos=[]
        
    def getprice(self, tm, cnt=1):
        maxretry=10
        while (maxretry >0):
            dd=readh5res(self.secdr, tm, cnt=cnt, path=self.path)
            if not dd is None:
                break
            time.sleep(0.2)
            maxretry-=1
        return dd
        
    
    def gen_orders(self, curtm):
        orders=[]
        # adaptive_smpairs=self.calc_adaptive_money(curtm)
        price_his_cnt=60
        dd=self.getprice(curtm, cnt=price_his_cnt)
        
        s1i=int(sk.gtmidx_i(curtm)/self.delta)*self.delta
        leftcnt=int((self.ends1i-s1i)/self.delta)
        if leftcnt<=0:
            return orders
        adaptive_smpairs=[]
        dr=dm.dr
        
        ######
        self.tr_map_flag = False
        llen=self.secdata.read_len()
        tmptm=sk.gtm_i(min(llen-llen%self.delta, s1i))
        self.get_tr(tmptm, self.cfg["simlen"])
        ######
        
        vwap_60s = get_vwap(self.secdr, s1i, 90)
        vwap_5s = get_vwap(self.secdr, s1i, 5)
        vwap_std = get_vwap_diff_std(self.secdr, s1i, 600, delta=5)
        diff_vwap = vwap_5s/vwap_60s - 1.0
        diff_vwap_norm = diff_vwap
        # diff_vwap_norm = diff_vwap_norm / vwap_std
        diff_vwap_norm[~np.isfinite(diff_vwap_norm)]=0
        # diff_vwap_money_ratio = np.clip(diff_vwap_norm/2+1, 0.2, 1.8) 
        
        for sid,item in self.smpairs.items():
            sidx=self.symbolidx[sid]
            symbol=sid
            left_money=item["target_money"]-item["cur_money"]
            cnt_ratio=(item["total_cnt"]-leftcnt)/item["total_cnt"]
            trade_direct=np.sign((item["target_money"]-item["begin_money"]))
            target=self.smpairs[symbol]
            # if ((item["target_money"]-item["begin_money"])*left_money <= 0) \
            exceed_ratio=1.0
            # exceed_ratio =  1- (target["tr_avg"] - self.cfg["mbr_thres"]*0.8)*2
            if ((abs(item["target_money"]-item["begin_money"])*exceed_ratio < abs(item["cur_money"]-item["begin_money"])) 
                # or (abs(item["target_money"])<1.0 and  item["cur_money"]*item["begin_money"]<=0) 
                or (sid not in price_precision_map)):
                # print(sid, "money:", item["target_money"], "left_money", left_money, (item["target_money"]-item["begin_money"]), flush=True)
                continue
            
            last_cnt=1
            pred=pred_mom=0
            if dd is None:
                if np.isfinite(vwap_5s[sidx]):
                    price=vwap_5s[sidx]
                else:
                    continue
            else:
                if trade_direct > 0:
                    price=dd["svwap"][-1, sidx]*self.price_normw+dd["slow"][-1, sidx]*(1-self.price_normw)
                    pred=dd["sl"][-last_cnt:, sidx].mean()
                    pred_mom = dd["sl"][-last_cnt:, sidx].mean() - dd["sl"][-last_cnt-1:-1, sidx].mean()
                    pred /= (dd["sl"][-price_his_cnt:, sidx].std())
                    pred_mom /= (dd["sl"][-price_his_cnt:, sidx].std())
                else:
                    price=dd["bvwap"][-1:, sidx].mean()*self.price_normw+dd["bhigh"][-1, sidx]*(1-self.price_normw)
                    pred=-1.0*dd["bl"][-last_cnt, sidx]
                    pred_mom = -1.0*(dd["bl"][-last_cnt:, sidx].mean() - dd["bl"][-last_cnt-1:-1, sidx].mean())
                    pred /= (dd["bl"][-price_his_cnt:, sidx].std())
                    pred_mom /= (dd["bl"][-price_his_cnt:, sidx].std())
            if ~np.isfinite(price) :
                print("error price nan:", sid, curtm, flush=True)
                if np.isfinite(vwap_60s[sidx]):
                    price=vwap_60s[sidx]
                else:
                    continue

            scale=np.float64(0.1)**price_precision_map[symbol]
            price_min_delta=max(price/10000.0*1.0,  min(scale, price/10000.0))
            precision=scale/price*10000
            #################
            
            left_money=target["target_money"]-target["cur_money"]
            total_money=target["target_money"]-target["begin_money"]
            completed_ratio=(total_money-left_money)/total_money
            cnt_ratio=(target["total_cnt"]-leftcnt)/target["total_cnt"]
            expect_completed_ratio = completed_ratio/(cnt_ratio+0.01)
            decay_ratio=0.1
            target["cr_avg"]=target["cr_avg"]*(1-decay_ratio)+decay_ratio*expect_completed_ratio
                
            expand_ratio=1.0
            expand=self.expand
            if sid in self.symbolidx:
                tratio=self.mb_tratio[self.symbolidx[sid]] if item["avg_money"]>0 else self.ms_tratio[self.symbolidx[sid]]
            else:
                tratio=2.0
            tratio=self.maxr
            high_tratio = get_threshold(sid)/abs(target["avg_money"])
            high_tratio = np.clip(max(high_tratio, tratio), 0, 10)
            
            ori_tratio = tratio
            select_bp=5.0
            ts=self.cfg["ts"]
            # thres_ratio=0.2+np.clip((target["alpha"]*trade_direct*0.5*ts+1.0), 0.0, 2*ts)
            thres_ratio=1.0
            tratio=tratio*thres_ratio
            tr_ratio=0.0
            if self.tr_map_flag : 
                # bps=self.bp_map[-1]
                bps=5
                bp3_mbr = self.tr_map[bps]["mbr"][sidx] 
                bp3_msr = self.tr_map[bps]["msr"][sidx]
            else:
                bp3_mbr=bp3_msr=0
            lsr = (bp3_msr - bp3_mbr)/(max(bp3_mbr, bp3_msr)+0.05)
            ls_avgw=0.8
            target["ls_avg"] = target["ls_avg"] * ls_avgw + lsr * (1-ls_avgw)
            ls_mom = lsr - target["ls_avg"]
            lsrw = 0.99
            ls_merge = lsr * lsrw +  ls_mom * (1-lsrw)
            dv=diff_vwap_norm[sidx]*1000
            pred_w = 0.1
            pred_merge = pred*pred_w+pred_mom*(1-pred_w)
            if self.tr_map_flag:
                for bp in reversed(self.bp_map):
                    mbr=self.tr_map[bp]["mbr"]
                    msr=self.tr_map[bp]["msr"]
                    if (trade_direct > 0 
                        and (
                        (mbr[sidx]>self.cfg["sbp_trhes"] 
                        # and mbr[sidx]*high_tratio*1.0 > self.trw*(thres_ratio**self.cfg["tr_exp"])
                        )
                        or bp == self.bp_map[0])):
                        select_bp=bp
                        tr_ratio=mbr[sidx]
                        break
                    if (trade_direct < 0 
                        and (
                        (msr[sidx]>self.cfg["sbp_trhes"]
                        # and msr[sidx]*high_tratio*1.0 > self.trw*(thres_ratio**self.cfg["tr_exp"])
                        )
                        or bp == self.bp_map[0])):
                        select_bp=bp
                        tr_ratio=msr[sidx]
                        break
            
            # tratio *= (1 + 3*np.clip(tr_ratio*tratio, -0.1, 0.2))
            maxr = self.maxr
            minr = self.cfg["minr"]
            
            limit_ratio=maxr
            tratio = np.clip((self.trw/(tr_ratio+0.001))**self.rwp, 0.5, limit_ratio)
            b_tratio =  tratio
            
            adav = np.clip(trade_direct*ls_merge*0.0+1.0, 0.5, 1.5)
            tratio=np.clip((target["accv"]*0.3+tratio)*adav, minr, maxr)
            
            target["accv"] += (b_tratio -tratio)
            
            bu_w=self.cfg["bu_w"] 
            money_normw=self.money_normw

            tr_avgw=0.1
            target["tr_avg"] = tr_ratio*tr_avgw+target["tr_avg"]*(1-tr_avgw)
            # mbr = bp3_mbr if trade_direct>0 else bp3_msr
            # mbr = tr_ratio
            # mbrw=self.cfg["mbrw"]
            # minr = self.cfg["minr"]
            # tratio = np.clip((1.0/np.clip(mbr*mbrw+(1-self.cfg["mbr_thres"]*mbrw), 0.1, 1.5)), minr, self.maxr)
            
            # slow trade
            if abs(target["target_money"]) < 1.0:
                tratio =np.clip(tratio*0.9, 1.0, self.maxr)
            if target["cr_avg"] > 1.2 and cnt_ratio > 0.15 :
                tratio = tratio / np.clip(target["cr_avg"] - 0.1, 1, 1.5)
                tratio = np.clip(tratio, minr, self.maxr)
            if target["cr_avg"] < 1.0 and cnt_ratio > 0.15 and target["tr_avg"] < self.cfg["mbr_thres"]*0.8:
                tratio = tratio / np.clip(target["cr_avg"]*2-self.cfg["ucw"], 0.3, 1.0)
                tratio = np.clip(tratio, minr, self.maxr*3.0)
                select_bp = select_bp / np.clip((2.0-target["cr_avg"]*1.0), 1.0, 4.0)
            if target["cr_avg"] > 0.8 and cnt_ratio > 0.15 and target["tr_avg"] > self.cfg["mbr_thres"]*0.9:
                tratio = tratio / np.clip(target["cr_avg"] + 0.2, 1, 2.0)
                tratio = np.clip(tratio, minr*0.5, self.maxr)
            
            # if cnt_ratio > 0.85 and (abs(target["target_money"]) > 1.0):
            #     money_normw=1.1-cnt_ratio
            #     select_bp=np.clip(0.9-cnt_ratio, 0, 1.0)*select_bp
            #     tratio=ori_tratio*self.cfg["bu_ratio"]
            #     bu_w=self.cfg["bu_w"] + 0.5*(cnt_ratio-0.7)
            
            if abs(left_money) < get_threshold(sid)*7:
                expand=[0.0]
                # if random.random()<1.0/(min(self.cancel_delay, 10)+0.1):
                if random.random()<(abs(left_money)/(min(self.cancel_delay, 15)*get_threshold(sid))):
                    adaptive_money=trade_direct*min(get_threshold(sid), abs(left_money))
                else:
                    adaptive_money=0
            else:
                adaptive_money=item["avg_money"]*money_normw+\
                    (left_money/leftcnt)*(1-money_normw)

                noise_ratio=0.9+random.random()*0.2  #noise trade money
                adaptive_money*=tratio*noise_ratio
                
                money_ratio=bu_w
                if abs(left_money)*money_ratio<abs(adaptive_money):
                    adaptive_money=left_money*money_ratio
                    
                adaptive_money=np.sign(left_money)*np.clip(abs(adaptive_money), get_threshold(sid), abs(left_money/2))
            
            money = adaptive_money
            
            limit_money = 51
            if target["locnt"] / (target["total_cnt"]-leftcnt+2) > 0.85 and abs(left_money) > limit_money and abs(money) < limit_money:
                money=trade_direct*limit_money
                
            if abs(money) < limit_money:
                target["locnt"]+=1
            
            if abs(money) < get_threshold(sid) :
                continue
            target["o_m"]+=money
            select_bp=float(select_bp)
            adaptive_shift=float(select_bp)
                             
            order_info=dict(curtm=curtm, symbol=symbol, target_money=target["target_money"], begin_money=target["begin_money"],
                            cur_money=target["cur_money"], order_money=money, order_price=0, order_id=0,
                            completed_ratio=completed_ratio, cr_avg=target["cr_avg"], cnt_ratio=cnt_ratio, 
                            expect_completed_ratio=expect_completed_ratio,
                            adaptive_shift=adaptive_shift, leftcnt=leftcnt, precision=precision,
                            tratio=tratio, select_bp=select_bp, alpha=target["alpha"], tr_ratio=tr_ratio, 
                            # bp3_mbr=bp3_mbr, bp3_msr=bp3_msr, 
                            lsr=lsr, ls_mom=ls_mom, ls_merge = ls_merge, accv=target["accv"], 
                            adav=adav, dv=dv, pred=pred, pred_mom=pred_mom, pred_merge=pred_merge,
                            # xswap=dd["svwap"][-1, sidx], xslow=dd["slow"][-1, sidx], xbvwap=dd["bvwap"][-1, sidx], xbhigh=dd["bhigh"][-1, sidx],
                            # pred_sl=dd["sl"][-1:, sidx].mean(), pred_bl=dd["bl"][-1, sidx].mean(),  pmr=pred_money_ratio,
                            # pred_sl=dd["sl"][-1, sidx]-dd["sl"][:, sidx].mean(axis=0), pred_bl=dd["bl"][-1, sidx]-dd["bl"][:, sidx].mean(axis=0),
                            )
            ##############    

            price=price-adaptive_shift*price_min_delta*np.sign(money)

            oldprice=price
            prices=[price-expand_ratio*bp*price_min_delta*np.sign(money) for bp in expand]
            tmporders={}
            

            for price in prices:
                money_expand = money/len(expand)
                up=np.ceil(price/scale)
                down=np.floor(price/scale)
                upmoney=(price/scale-down)*money_expand
                upmoney=upmoney if abs(upmoney) >=get_threshold(sid) else 0
                downmoney=money_expand-upmoney
                downmoney=downmoney if abs(downmoney) >=get_threshold(sid) else 0
                # if (abs(upmoney)>0):
                #     if (up not in tmporders) :
                #         tmporders[up]=upmoney
                #     else:
                #         tmporders[up]+=upmoney
                # if (abs(downmoney)>0):
                #     if (down not in tmporders) :
                #         tmporders[down]=downmoney
                #     else:
                #         tmporders[down]+=downmoney
                
                if random.random() < upmoney/money_expand:
                    if up not in tmporders:
                        tmporders[up]=money_expand
                    else:
                        tmporders[up]+=money_expand
                else:
                    if down not in tmporders:
                        tmporders[down]=money_expand
                    else:
                        tmporders[down]+=money_expand

            keylist=sorted(tmporders.keys()) if  money > 0 else sorted(tmporders.keys(), reverse=True)
            for key in keylist:
                tmp_order_info=copy.deepcopy(order_info)
                tmp_order_info["order_price"]=key*scale
                tmp_order_info["order_money"]=tmporders[key]
                self.order_infos.append(tmp_order_info)
                orders.append((symbol, sidx, key*scale, tmporders[key], tmp_order_info))
        
        return  orders
    
    def restart(self, curtm, endtm, delta, smpairs):
        s1i=int(sk.gtmidx_i(curtm)/delta)*delta
        self.ends1i=int(sk.gtmidx_i(endtm)/delta)*delta
        self.delta=delta
        leftcnt=int((self.ends1i-s1i)/self.delta)
        
        self.status_dict={}
        self.status_dict["t_money"]=np.zeros(self.mb_tratio.shape)
        self.status_dict["cur_money"]=np.zeros(self.mb_tratio.shape)
        self.status_dict["b_money"]=np.zeros(self.mb_tratio.shape)
        self.smpairs={}
        for item in smpairs.items():
            symbol=item[0]
            sidx=self.symbolidx[symbol]
            target_money=item[1][0]
            cur_money=item[1][1]
            if abs(target_money-cur_money)>max(20.0, get_threshold(symbol)):
            # if abs(target_money-cur_money)> 500 or abs(target_money) < 1:
                self.status_dict["t_money"][sidx]=target_money
                self.status_dict["cur_money"][sidx]=cur_money
                self.status_dict["b_money"][sidx]=cur_money
                values=dict(target_money=target_money, begin_money=cur_money, 
                            cur_money=cur_money, total_cnt=leftcnt, avg_money=(target_money-cur_money)/leftcnt,
                            ls_avg = 0, accv=0, locnt=0, o_m=0, tr_avg=self.cfg["mbr_thres"],
                            cr_avg=self.tcr+0.2, alpha=item[1][2])
                self.smpairs[symbol]=values
        return
    
    def update_money(self, smpairs):
        for sid,money in smpairs.items():
            if sid in self.smpairs:
                sidx=self.symbolidx[sid]
                self.smpairs[sid]["cur_money"]=money
                self.status_dict["cur_money"][sidx]=money
        return
    
    def get_completed_symbols(self):
        cr_symbols=[]
        exceed_ratio=0.95
        for sid,item in self.smpairs.items():
            target=item
            # exceed_ratio =  1- (target["tr_avg"] - self.cfg["mbr_thres"]*0.8)*2
            if (abs(item["target_money"]-item["begin_money"])*exceed_ratio <= 
                abs(item["cur_money"]-item["begin_money"])):
                cr_symbols.append(sid)
        return cr_symbols
        
    
    def update_and_gorders(self, curtm, smpairs):
        s1i=int(sk.gtmidx_i(curtm)/self.delta)*self.delta
        curtm=sk.gtm_i(s1i)
        tpl=tools.TimeProfilingLoop("update_and_gorders")
        get_tr_tm=tpl.add("get_tr")

        get_tr_tm.end()
        self.update_money(smpairs)
        orders = self.gen_orders(curtm)
        tpl.end()
        # print("update_and_gorders tm cost:", tpl.to_string(), curtm, tmptm)
        return orders
    
    
    def get_tr(self, curtm, period):
        tradelen=period
        dr=ud.g_data
        dr["uid"]=dr["sids"]
        money=2000.0
        avgmoney=money/(tradelen/self.delta)
        secdr=self.secdr
        
        lend=sk.gtmidx_i(curtm)+1
        end=int(sk.gtmidx_i(curtm)/self.delta)*self.delta
        start=int(sk.gtmidx_i(curtm)/self.delta)*self.delta-period
        self.tr_map={}
        self.tr_map_flag = True
        for bp in self.bp_map:
            #maker buy money
            mb_money=np.zeros(dr["uid"].shape[0])
            mb_volume=np.zeros(dr["uid"].shape[0])
    
            #maker sell money
            ms_money=np.zeros(dr["uid"].shape[0])
            ms_volume=np.zeros(dr["uid"].shape[0])
    
            for s1i in range(start, end, self.delta):
                tm=sk.gtm_i(s1i)
                
                dd=self.getprice(tm)
                if dd is None:
                    break
                    self.tr_map_flag = False
                xsvwap, xslow=dd["svwap"][-1], dd["slow"][-1]
                xbvwap, xbhigh=dd["bvwap"][-1], dd["bhigh"][-1]
                pred_sl, pred_bl = dd["sl"][-1], dd["bl"][-1]
                xsvwap=xsvwap*self.price_normw+xslow*(1-self.price_normw)
                xsvwap=xsvwap-bp*xsvwap/10000.0
                xbvwap=xbvwap*self.price_normw+xbhigh*(1-self.price_normw)
                xbvwap=xbvwap+bp*xsvwap/10000.0
                
                valid=np.isfinite(xsvwap)
                vwap, vf_y, svwap, shigh, slow, vf_ys, bvwap, bhigh, blow, vf_yb, smoney, bmoney, ysl, ybl=getyvalues(
                    self.secdr, s1i, min(self.delta*self.cfg["simdc"], lend-s1i-delaysec))
                # vwap, _, svwap, shigh, slow, _, bvwap, bhigh, blow, _, smoney, bmoney, _, _=getyvalues(
                #     self.secdr, s1i, min(self.delta*self.cancel_delay, end-s1i-delaysec))
    
                svf=valid&vf_ys
                bvf=valid&vf_yb
                money_limit_ratio=self.money_limit_ratio
                trade_succ_ratio=self.trade_succ_ratio
                mb_tradeflag=(svf & (xsvwap>=slow))
                trade_money=(smoney[mb_tradeflag]*money_limit_ratio).clip(0, avgmoney)
                mb_money[mb_tradeflag]+=trade_money
                
                ms_tradeflag=(bvf & (xbvwap<=bhigh))
                trade_money=(bmoney[ms_tradeflag]*money_limit_ratio).clip(0, avgmoney)
                ms_money[ms_tradeflag]+=trade_money

            
            mbr=mb_money/money
            msr=ms_money/money
            self.tr_map[bp]=dict(mbr=mbr, msr=msr)

        return 
    
    def get_tratio(self, curtm):
        tradedelta=15*60
        tradelen=5*60
        dr=ud.g_data
        dr["uid"]=dr["sids"]
        money=10000.0
        avgmoney=money/(tradelen/5)
        secdr=self.secdr
        mbrs=np.zeros(dr["uid"].shape[0])
        mbpds=np.zeros(dr["uid"].shape[0])
        mb_pred_diffs=np.zeros(dr["uid"].shape[0])
        mb_long_pred_diffs=np.zeros(dr["uid"].shape[0])
        msrs=np.zeros(dr["uid"].shape[0])
        mspds=np.zeros(dr["uid"].shape[0])
        ms_pred_diffs=np.zeros(dr["uid"].shape[0])    
        ms_long_pred_diffs=np.zeros(dr["uid"].shape[0]) 
        
        end=int(sk.gtmidx_i(curtm)/self.delta)*self.delta-self.delta*self.cancel_delay
        start=int(sk.gtmidx_i(curtm)/self.delta)*self.delta-self.simcnt-self.delta*self.cancel_delay
        tradecnt=0
        for tidx in range(start, end, tradedelta):
            tstart=tidx
            tend=tstart+tradelen
            #maker buy money
            mb_money=np.zeros(dr["uid"].shape[0])
            mb_volume=np.zeros(dr["uid"].shape[0])
            mb_pred_diff=np.zeros(dr["uid"].shape[0])
            mb_long_pred_diff=np.zeros(dr["uid"].shape[0])
            #maker sell money
            ms_money=np.zeros(dr["uid"].shape[0])
            ms_volume=np.zeros(dr["uid"].shape[0])
            ms_pred_diff=np.zeros(dr["uid"].shape[0])
            ms_long_pred_diff=np.zeros(dr["uid"].shape[0])
            
            bvf_cnt = np.zeros(dr["uid"].shape[0])
            svf_cnt = np.zeros(dr["uid"].shape[0])
            icsls, icbls = [], []
            pred_bl_mat, pred_sl_mat, ysl_mat, ybl_mat = [], [], [], []
            order_cnt=int((tend-tstart)/self.delta)
            for s1i in range(tstart, tend, self.delta):
                tm=sk.gtm_i(s1i)
                
                dd=self.getprice(tm)
                if dd is None:
                    continue
                xsvwap, xslow=dd["svwap"][-1], dd["slow"][-1]
                xbvwap, xbhigh=dd["bvwap"][-1], dd["bhigh"][-1]
                pred_sl, pred_bl = dd["sl"][-1], dd["bl"][-1]
                xsvwap=xsvwap*self.price_normw+xslow*(1-self.price_normw)
                xsvwap=xsvwap-self.shift*xsvwap/10000.0
                xbvwap=xbvwap*self.price_normw+xbhigh*(1-self.price_normw)
                xbvwap=xbvwap+self.shift*xsvwap/10000.0
                
                valid=np.isfinite(xsvwap)
                vwap, vf_y, svwap, shigh, slow, vf_ys, bvwap, bhigh, blow, vf_yb, smoney, bmoney, ysl, ybl=getyvalues(
                    self.secdr, s1i, self.delta)
                vwap, _, svwap, shigh, slow, _, bvwap, bhigh, blow, _, smoney, bmoney, _, _=getyvalues(
                    self.secdr, s1i, self.delta*self.cancel_delay)

                svf=valid&vf_ys
                bvf=valid&vf_yb
                money_limit_ratio=self.money_limit_ratio
                trade_succ_ratio=self.trade_succ_ratio
                # mb_tradeflag=(svf & (xsvwap>=slow) & (xsvwap <= bhigh) & (np.random.random(size=svf.shape) < trade_succ_ratio))
                mb_tradeflag=(svf & (xsvwap>=slow))
                trade_money=(smoney[mb_tradeflag]*money_limit_ratio).clip(0, avgmoney)
                mb_money[mb_tradeflag]+=trade_money
                mb_volume[mb_tradeflag]+=trade_money/xsvwap[mb_tradeflag]
                mb_pred_diff[svf]+=np.abs((xsvwap/svwap-1.0)*10000.0)[svf]
                mb_long_pred_diff[svf]+=np.abs(pred_sl-ysl)[svf]
                svf_cnt[svf]+=1
                
                # ms_tradeflag=(bvf & (xbvwap<=bhigh) & (xbvwap >= slow) & (np.random.random(size=svf.shape) < trade_succ_ratio))
                ms_tradeflag=(bvf & (xbvwap<=bhigh))
                trade_money=(bmoney[ms_tradeflag]*money_limit_ratio).clip(0, avgmoney)
                ms_money[ms_tradeflag]+=trade_money
                ms_volume[ms_tradeflag]+=trade_money/xbvwap[ms_tradeflag]
                ms_pred_diff[bvf]+=np.abs((xbvwap/bvwap-1.0)*10000.0)[bvf]
                ms_long_pred_diff[bvf]+=np.abs(pred_bl-ybl)[bvf]
                bvf_cnt[bvf]+=1
                
                pred_bl_mat.append(pred_bl)
                pred_sl_mat.append(pred_sl)
                ybl_mat.append(ybl)
                ysl_mat.append(ysl)
                icsl=np.corrcoef(pred_sl[svf], ysl[svf])[0, 1]
                icbl=np.corrcoef(pred_bl[bvf], ybl[bvf])[0, 1]
                icsls.append(icsl)
                icbls.append(icbl)
                    
            allvwap=secdr["s1info_money"][tstart:tend].sum(axis=0)/secdr["s1info_volume"][tstart:tend].sum(axis=0)
            svwap=secdr["s1info_smoney"][tstart:tend].sum(axis=0)/secdr["s1info_svolume"][tstart:tend].sum(axis=0)
            bvwap=(secdr["s1info_money"][tstart:tend].sum(axis=0)-secdr["s1info_smoney"][tstart:tend].sum(axis=0))/\
                (secdr["s1info_volume"][tstart:tend].sum(axis=0)-secdr["s1info_svolume"][tstart:tend].sum(axis=0))
        
            pred_bl_mat=np.array(pred_bl_mat)
            pred_sl_mat=np.array(pred_sl_mat)
            ybl_mat=np.array(ybl_mat)
            ysl_mat=np.array(ysl_mat)
            pred_bl_mat[np.isnan(pred_bl_mat)]=0
            pred_sl_mat[np.isnan(pred_sl_mat)]=0
            ybl_mat[np.isnan(ybl_mat)]=0
            ysl_mat[np.isnan(ysl_mat)]=0
            icb=pearsonr(torch.from_numpy(pred_bl_mat), torch.from_numpy(ybl_mat), batch_first=False)
            ics=pearsonr(torch.from_numpy(pred_sl_mat), torch.from_numpy(ysl_mat), batch_first=False)
            np.nanmean(icb), np.nanmean(ics), np.array(icsls).mean(), np.array(icbls).mean()
            
            mbr=mb_money/money
            mbvwap=mb_money/mb_volume
            mbpd=(mbvwap/allvwap-1.0)*10000.0
            msr=ms_money/money
            msvwap=ms_money/ms_volume
            mspd=(msvwap/allvwap-1.0)*10000.0
            
            mbrs[np.isfinite(mbr)]+=mbr[np.isfinite(mbr)]
            mbpds[np.isfinite(mbpd)]+=mbpd[np.isfinite(mbpd)]
            mb_pred_diffs[np.isfinite(mb_pred_diff)]+=(mb_pred_diff/svf_cnt)[np.isfinite(mb_pred_diff)]
            mb_long_pred_diffs[np.isfinite(mb_long_pred_diff)]+=(mb_long_pred_diff/svf_cnt)[np.isfinite(mb_long_pred_diff)]
            msrs[np.isfinite(msr)]+=msr[np.isfinite(msr)]
            mspds[np.isfinite(mspd)]+=mspd[np.isfinite(mspd)]  
            ms_pred_diffs[np.isfinite(ms_pred_diff)]+=(ms_pred_diff/bvf_cnt)[np.isfinite(ms_pred_diff)]
            ms_long_pred_diffs[np.isfinite(ms_long_pred_diff)]+=(ms_long_pred_diff/bvf_cnt)[np.isfinite(ms_long_pred_diff)]

            tradecnt+=1
            df=pd.DataFrame()
            df["sid"]=dr["sids"]
            df["mbrs"]=mbrs/tradecnt
            df["mbpds"]=mbpds/tradecnt
            df["msrs"]=msrs/tradecnt
            df["mspds"]=mspds/tradecnt
            a=0
            
        
        df=pd.DataFrame()
        df["sid"]=dr["sids"]
        df["mbrs"]=mbrs/tradecnt
        df["mbpds"]=mbpds/tradecnt
        # df["mb_pred_diffs"]=mb_pred_diffs/tradecnt
        # df["mb_long_pred_diffs"]=mb_long_pred_diffs/tradecnt
        
        df["msrs"]=msrs/tradecnt
        df["mspds"]=mspds/tradecnt
        # df["ms_pred_diffs"]=ms_pred_diffs/tradecnt
        # df["ms_long_pred_diffs"]=ms_long_pred_diffs/tradecnt
        return df["mbrs"], df["msrs"]
    
    def get_order_info(self, curtm):
        df=None
        if len(self.order_infos)>0:
            s1i=sk.gtmidx_i(curtm)
            s1i-=s1i%300
            min5vwap=get_vwap(self.secdr, s1i+300, 300)
            for order_info in self.order_infos:
                s1i=sk.gtmidx_i(order_info["curtm"])
                sidx=self.symbolidx[order_info["symbol"]]
                vwap, vf_y, svwap, shigh, slow, vf_ys, bvwap, bhigh, blow, vf_yb, smoney, bmoney, ysl, ybl=getyvalues(
                    self.secdr, s1i, self.delta, longterm=60)
                order_info["min5vwap"]=min5vwap[sidx]
                order_info["vwap"]=vwap[sidx]
                order_info["svwap"]=svwap[sidx]
                order_info["shigh"]=shigh[sidx]
                order_info["slow"]=slow[sidx]
                order_info["bvwap"]=bvwap[sidx]
                order_info["bhigh"]=bhigh[sidx]
                order_info["blow"]=blow[sidx]
                order_info["ysl"]=ysl[sidx]
                order_info["ybl"]=ybl[sidx]
                order_info["all_money"]=smoney[sidx]+bmoney[sidx]
                order_info["sell_money"]=smoney[sidx]
                order_info["buy_money"]=bmoney[sidx]
            df=pd.DataFrame(self.order_infos)
            self.order_infos=[]
        return df
    
    def save_order_info(self, curtm):
        if len(self.order_infos)>0:
            s1i=sk.gtmidx_i(curtm)
            s1i-=s1i%300
            min5vwap=get_vwap(self.secdr, s1i+300, 300)
            for order_info in self.order_infos:
                s1i=sk.gtmidx_i(order_info["curtm"])
                sidx=self.symbolidx[order_info["symbol"]]
                vwap, vf_y, svwap, shigh, slow, vf_ys, bvwap, bhigh, blow, vf_yb, smoney, bmoney, ysl, ybl=getyvalues(
                    self.secdr, s1i, self.delta)
                order_info["min5vwap"]=min5vwap[sidx]
                order_info["vwap"]=vwap[sidx]
                order_info["svwap"]=svwap[sidx]
                order_info["shigh"]=shigh[sidx]
                order_info["slow"]=slow[sidx]
                order_info["bvwap"]=bvwap[sidx]
                order_info["bhigh"]=bhigh[sidx]
                order_info["blow"]=blow[sidx]
                order_info["ysl"]=ysl[sidx]
                order_info["ybl"]=ybl[sidx]
                order_info["all_money"]=smoney[sidx]+bmoney[sidx]
                order_info["sell_money"]=smoney[sidx]
                order_info["buy_money"]=bmoney[sidx]
            df=pd.DataFrame(self.order_infos)
            self.order_infos=[]

            # global_mutex.acquire()
            # global_mutex.release()
            maxretry=3
            while maxretry > 0:
                try:
                    time.sleep(random.random()*5)
                    global_mutex.acquire()
                    save_order_info_path=self.save_order_info_path+"_"+str(curtm)[:8]+".h5"
                    print(curtm, "acquire global_mutex:", self.smpairs.keys(), flush=True)
                    cmd=f'h5clear -s {save_order_info_path}'
                    os.system(cmd)
                    store = pd.HDFStore(save_order_info_path)
                    print(curtm, "succ open order_info:", save_order_info_path, self.smpairs.keys(), df.shape, flush=True)
                    store.append('data', df, data_columns=df.columns, min_itemsize={'symbol':32, 'order_id':128})
                    print(curtm, "succ save order_info:", self.smpairs.keys(), df.shape, "all shape:", store["data"].shape, flush=True)
                    store.close()
                    time.sleep(2.0)
                    global_mutex.release()
                    print(curtm, "release global_mutex:", self.smpairs.keys(), flush=True)
                    break
                except Exception as e:
                    time.sleep(random.random()*5)
                    maxretry-=1
                    if not store is None:
                        store.close()
                    global_mutex.release()
                    print(curtm, "release global_mutex:", self.smpairs.keys(), flush=True)
                    print("wait save order_info:", self.smpairs.keys(), "\n", str(e), flush=True)
                
            # print(curtm, "acquire global_mutex:", self.smpairs.keys(), flush=True)

            # print(curtm, "release global_mutex:", self.smpairs.keys(), flush=True)
            
            
    def update_tratio(self, curtm):
        # curtm=int(sk.gtmidx_i(curtm)/self.delta)*self.delta
        mb_tratio, ms_tratio=self.get_tratio(curtm)
        mb_tratio, ms_tratio=1/(mb_tratio+0.01), 1/(ms_tratio+0.01)
        mb_tratio[~np.isfinite(mb_tratio)]=2.0
        ms_tratio[~np.isfinite(ms_tratio)]=2.0
        mb_tratio=mb_tratio**self.rwp
        ms_tratio=ms_tratio**self.rwp
        mb_tratio=np.clip(mb_tratio, 1, self.maxr)
        ms_tratio=np.clip(mb_tratio, 1, self.maxr)
        self.mb_tratio=mb_tratio*self.trw
        self.ms_tratio=ms_tratio*self.trw
        return
        
    
    
    
    
    
    
    
    
    

    
