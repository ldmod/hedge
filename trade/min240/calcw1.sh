#!/bin/bash

`ps aux | grep calcwv260_1w.py  | grep -v grep | awk '{print $2}' | xargs kill -9`

`nohup python3 ./calcwv260_1w.py >> ./logs/calcwv260_1w.log 2>&1 &`

sleep 1