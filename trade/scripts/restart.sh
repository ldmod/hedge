#!/bin/bash

`ps aux | grep trade | grep -v grep | awk '{print $2}' | xargs kill -9`

`nohup python3 /home/ubuntu/cryptoqt/trade/multi/trade.py > /tmp/trade.log 2>&1 &`

sleep 1


