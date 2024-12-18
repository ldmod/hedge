#!/bin/bash

`ps aux | grep order.py | grep -v grep | awk '{print $2}' | xargs kill -9`

`nohup python3 -u ./order.py > ./order.log 2>&1 &`

sleep 1