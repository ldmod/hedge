#!/bin/bash

`ps aux | grep tradeAT601.py  | grep -v grep | awk '{print $2}' | xargs kill -9`

`nohup python3 ./tradeAT601.py >> ./logs/tradeAT.log 2>&1 &`

sleep 1
