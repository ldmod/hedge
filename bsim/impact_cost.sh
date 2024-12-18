#!/bin/bash

`ps aux | grep impact_cost.py | grep -v grep | awk '{print $2}' | xargs kill -9`

`nohup python3 -u ./impact_cost.py > ./cost.log 2>&1 &`

sleep 1