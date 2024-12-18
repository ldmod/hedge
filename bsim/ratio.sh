#!/bin/bash

`ps aux | grep ratio.py | grep -v grep | awk '{print $2}' | xargs kill -9`

`nohup python3 -u ./ratio.py > ./ratio.log 2>&1 &`

sleep 1