#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 19:45:46 2024

@author: prod
"""

#!/usr/bin/env python
import logging
from binance.um_futures import UMFutures
from binance.lib.utils import config_logging
import os
import cryptoqt.data.tools as tools
from binance.spot import Spot as Client
import yaml
import requests
from datetime import datetime
def loadcfg(path):
    global gv
    with open(path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg
cfg = loadcfg("cfg.yaml")
# os.environ["http_proxy"] = "socks5://127.0.0.1:7891"
# os.environ["https_proxy"] = "socks5h://127.0.0.1:7891"
myproxies = {
        'http': 'http://127.0.0.1:33881',
        'https': 'http://127.0.0.1:33881'
}
config_logging(logging, logging.DEBUG)

timeout=0.5
um_futures_client = UMFutures(proxies=myproxies, timeout=timeout)

exchange_info=um_futures_client.exchange_info()

logging.info(um_futures_client.klines("BTCUSDT", "1d", startTime=1717113600000))

spot_client = Client(base_url="https://api.binance.com", proxies=myproxies, timeout=timeout)

logging.info(spot_client.klines("BTCUSDT", "1m"))
logging.info(spot_client.klines("BTCUSDT", "1h", limit=10))


logging.info(um_futures_client.depth("BTCUSDT", **{"limit": 5}))
logging.info(um_futures_client.agg_trades("DOTUSDT", limit=1000, startTime=1717113600000, endTime=1717113601000))



