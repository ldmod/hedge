#!/bin/bash

`ps aux | grep ratio5m.py | grep -v grep | awk '{print $2}' | xargs kill -9`

`nohup python3 -u ./ratio5m.py > ./logs/ratio5m.log 2>&1 &`

sleep 1
