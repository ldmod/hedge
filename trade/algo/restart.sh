#!/bin/bash

`ps aux | grep tradeAT.py | grep -v grep | awk '{print $2}' | xargs kill -9`

`nohup python3 /home/crypto/sec_klines_env/cryptoqt/trade/algo/tradeAT.py >> ./logs/tradeAT.log 2>&1 &`

sleep 1