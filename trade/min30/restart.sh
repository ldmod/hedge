#!/bin/bash

`ps aux | grep tradeAT.py |grep min30 | grep -v grep | awk '{print $2}' | xargs kill -9`

`nohup python3 ./tradeAT.py >> ./logs/tradeAT.log 2>&1 &`

sleep 1
