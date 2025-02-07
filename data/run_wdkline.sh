#!/bin/bash

`ps aux | grep up_minkline.py  | grep -v grep | awk '{print $2}' | xargs kill -9`

sleep 2

pwd
`ps aux | grep wd_minkline.py | grep -v grep | awk '{print $2}' | xargs kill -9`
sleep 1
llen=10
for (( idx=0 ; idx < ${llen} ; idx++))
do
    #echo $idx
python wd_minkline.py --cfg ./config/wd_minkline.yaml --off ${idx}  --delta ${llen} > ./logs/wd_minkline${idx}.log 2>&1 &
python wd_minkline.py --cfg ./config/wd_spotminkline.yaml --off ${idx}  --delta ${llen} > ./logs/wd_spotminkline${idx}.log 2>&1 &
done

sleep 10
`nohup python3 ./up_minkline.py > ./ud.log 2>&1 &`
