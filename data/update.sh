#!/bin/bash

`ps aux | grep up_minkline.py  | grep -v grep | awk '{print $2}' | xargs kill -9`

`nohup python3 ./up_minkline.py > ./ud.log 2>&1 &`

sleep 1
