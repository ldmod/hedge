#!/bin/bash

`ps aux | grep cost5m.py | grep -v grep | awk '{print $2}' | xargs kill -9`

`nohup python3 -u ./cost5m.py >> ./logs/cost5m.log 2>&1 &`

sleep 1
