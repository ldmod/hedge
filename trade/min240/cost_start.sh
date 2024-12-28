#!/bin/bash

`ps aux | grep account_info.py | grep -v grep | awk '{print $2}' | xargs kill -9`

`ps aux | grep cost.py | grep -v grep | awk '{print $2}' | xargs kill -9`

sleep 10
`nohup python3 -u ./account_info.py > ./logs/account_info.log 2>&1 &`
`nohup python3 -u ./cost.py > ./logs/cost.log 2>&1 &`

sleep 1

