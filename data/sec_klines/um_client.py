#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 20:01:55 2024

@author: crypto
"""
import time
import cryptoqt.data.tools as tools
from binance.um_futures import UMFutures
import numpy as np
import random

class ClientWrapper(object):
    def __init__(self, myproxies, timeout):
        self.cur_min_cnt=dict(cur_min=0, cnt=0)
        self.flag=True
        self.timeout=timeout
        self.proxies=myproxies
        self.client = UMFutures(proxies=myproxies, timeout=self.timeout)
      
    def set_flag(self, v):
        self.flag=v
        
class UmClient(object):
    def __init__(self, cfg):
        self.timeout=cfg["timeout"]
        self.minmaxcnt=cfg["minmaxcnt"]
        self.um_futures_clients=[]
        for i in range(len(cfg["proxies"])):
            item=cfg["proxies"][i]
            myproxies = {
                    'http': item["http_proxy"],
                    'https': item["https_proxy"]
            }
            um_futures_client = ClientWrapper(myproxies, self.timeout)
            self.um_futures_clients.append(um_futures_client)
        return
    def get_um_client(self):
        while True:
            random_clis=self.um_futures_clients.copy()
            random.shuffle(random_clis)
            cur_min=int(time.time()/60)
            for i in range(len(self.um_futures_clients)):
                client=random_clis[i]
                if not client.flag:
                    continue
                if client.cur_min_cnt["cur_min"] < cur_min:
                    client.cur_min_cnt["cur_min"] =cur_min
                    client.cur_min_cnt["cnt"] = 0
                if client.cur_min_cnt["cnt"] < self.minmaxcnt:
                    client.cur_min_cnt["cnt"]+=1
                    return client
            time.sleep(5)
            
    