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
import threading

class ClientWrapper(object):
    def __init__(self, myproxies, timeout, Client):
        self.cur_min_cnt=dict(cur_min=0, cnt=0)
        self.flag=True
        self.use=0
        self.timeout=timeout
        self.proxies=myproxies
        self.client = Client(proxies=myproxies, timeout=self.timeout)
      
    def set_flag(self, v):
        self.flag=v
        
    def set_use(self, v):
        self.use=v
        
class UmClient(object):
    def __init__(self, cfg, Client):
        self.timeout=cfg["timeout"]
        self.minmaxcnt=cfg["minmaxcnt"]
        self.reuse_min=cfg["reuse_min"]
        self.lock = threading.RLock()
        self.um_futures_clients=[]
        self.maxRetryCnt = cfg["maxRetryCnt"]
        for i in range(len(cfg["proxies"])):
            item=cfg["proxies"][i]
            myproxies = {
                    'http': item["http_proxy"],
                    'https': item["https_proxy"]
            }
            um_futures_client = ClientWrapper(myproxies, self.timeout, Client)
            self.um_futures_clients.append(um_futures_client)
        return
    def recycle(self, client):
        self.lock.acquire()
        client.set_use(client.use-1)
        self.lock.release()
        return
    
    def get_um_client(self):
        self.lock.acquire()
        while True:
            random_clis=self.um_futures_clients.copy()
            random.shuffle(random_clis)
            cur_min=int(time.time()/60)
            for i in range(len(self.um_futures_clients)):
                client=random_clis[i]
                if client.cur_min_cnt["cur_min"] < cur_min-self.reuse_min: # recycle after 60 min
                    client.set_use(0)
                if (not client.flag) or  client.use>=self.maxRetryCnt:
                    print("in-use:", client.proxies, flush=True)
                    continue
                if client.cur_min_cnt["cur_min"] < cur_min:
                    client.cur_min_cnt["cur_min"] =cur_min
                    client.cur_min_cnt["cnt"] = 0
                if client.cur_min_cnt["cnt"] < self.minmaxcnt:
                    client.cur_min_cnt["cnt"]+=1
                    client.set_use(client.use+1)
                    print("get client:", client.proxies, flush=True)
                    self.lock.release()
                    return client
            print("no useful client", flush=True)
            time.sleep(5)
            
    
    
    
    
    
    
    
    
    
