#!/bin/bash

`ps aux | grep calcwv25_1w.py | grep -v grep | awk '{print $2}' | xargs kill -9`

python /home/crypto/sec_klines_env/cryptoqt/trade/algo/calcwv25_1w.py >> ./logs/calcwv25_1w.log 2>&1 &

sleep 1
