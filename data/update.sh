#!/bin/bash

`ps aux | grep updatedata.py  | grep -v grep | awk '{print $2}' | xargs kill -9`

`nohup python3 ./updatedata.py > ./ud.log 2>&1 &`

sleep 1
