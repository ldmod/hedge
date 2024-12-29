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
    

class GenerateOrders:
    def __init__(self, cfg):
        self.symbolidx={}
        self.delta=5
        self.cancel_delay=cfg["cancel_delay"]
        self.mb_tratio=np.ones((ud.g_data["sids"].shape[0]))*2
        self.cfg=cfg
        for idx,sid in enumerate(ud.g_data["sids"]):
            self.symbolidx[sid]=idx
        self.order_infos=[]
        
    
    def gen_orders(self, curtm):
        orders=[]
        s1i=int(sk.gtmidx_i(curtm)/self.delta)*self.delta
        leftcnt=int((self.ends1i-s1i)/self.delta)
        if leftcnt<=0:
            return orders
        dr=dm.dr
        
        for sid,item in self.smpairs.items():
            sidx=self.symbolidx[sid]
            symbol=sid
            left_money=item["target_money"]-item["cur_money"]
            cnt_ratio=(item["total_cnt"]-leftcnt)/item["total_cnt"]
            trade_direct=np.sign((item["target_money"]-item["begin_money"]))
            target=self.smpairs[symbol]
            exceed_ratio=1.0

            if ((abs(item["target_money"]-item["begin_money"])*exceed_ratio < abs(item["cur_money"]-item["begin_money"])) 
                or (sid not in price_precision_map)):
                continue
            
            left_money=target["target_money"]-target["cur_money"]
            total_money=target["target_money"]-target["begin_money"]
            completed_ratio=(total_money-left_money)/total_money
            cnt_ratio=(target["total_cnt"]-leftcnt)/target["total_cnt"]
            expect_completed_ratio = completed_ratio/(cnt_ratio+0.01)
            decay_ratio=0.1
            target["cr_avg"]=target["cr_avg"]*(1-decay_ratio)+decay_ratio*expect_completed_ratio
            
            if abs(left_money) < get_threshold(sid):
                adaptive_money=0
            else:
                adaptive_money=min(total_money/self.cfg["splitcnt"], left_money)
                
            money = adaptive_money
            
            if abs(money) < get_threshold(sid) :
                continue

            order_info=dict(curtm=curtm, symbol=symbol, target_money=target["target_money"], begin_money=target["begin_money"],
                            cur_money=target["cur_money"], order_money=money, order_price=0, order_id=0,
                            completed_ratio=completed_ratio, cr_avg=target["cr_avg"], cnt_ratio=cnt_ratio, 
                            expect_completed_ratio=expect_completed_ratio,
                            leftcnt=leftcnt, 
                            )
            ##############    
            
            tmporders={}
            for bp in self.cfg["expand"]:
                tmporders[bp]=money/len(self.cfg["expand"])

            keylist=sorted(tmporders.keys()) if  money > 0 else sorted(tmporders.keys(), reverse=True)
            for key in keylist:
                tmp_order_info=copy.deepcopy(order_info)
                tmp_order_info["order_price"]=key
                tmp_order_info["order_money"]=tmporders[key]
                self.order_infos.append(tmp_order_info)
                orders.append((symbol, sidx, key, tmporders[key], tmp_order_info))
        
        return  orders
    
    def restart(self, curtm, endtm, delta, smpairs):
        dm.init_less()
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
                values=dict(target_money=target_money, begin_money=cur_money, f_money=target_money-cur_money,
                            cur_money=cur_money, total_cnt=leftcnt, avg_money=(target_money-cur_money)/leftcnt,
                            cr_avg=0.0, alpha=item[1][2])
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
        for sid,item in self.smpairs.items():
            target=item
            exceed_ratio=1.0

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
            
            
    def update_tratio(self, curtm):
        return
        
    
    
    
    
    
    
    
    
    

    
