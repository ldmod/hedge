import os
import sys
sys.path.append(os.path.abspath(__file__+"../../../../../"))
import numpy as np
import torch
import cryptoqt.data.updatedata as ud
from functools import partial
import cryptoqt.data.datammap as dmap
import cryptoqt.data.tools as tools
import argparse
import time
import pandas as pd
import cryptoqt.trade.min3.runalpha_at as runalpha
import cryptoqt.data.constants as conts
import cryptoqt.alpha.min15_alpha as min15_alpha
import cryptoqt.alpha.min30_alpha as min30_alpha
import paramiko
from scp import SCPClient
from collections import deque
import cryptoqt.data.data_manager as dm
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test for argparse', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--src', help='src', default='/home/crypto/sec_klines_env/cryptoqt/smlp/model_states/infmodel/smlpv23/res')
    parser.add_argument('--dst',  help='dst', default='/home/crypto/signal/predv23_1w')

    args = parser.parse_args()
    os.makedirs(args.dst, exist_ok=True)

    dm.init()
    dr=dm.dr
    lasteddownload=dmap.read_memmap(dr)
        
    dstfiles=os.listdir(args.dst)
    ddates={}
    for x in dstfiles:
        ddates[x.split('_')[0]]=1
    min1i=dr["min1info_tm"].tolist().index(20240831000000)
    delta=3
    alphas=deque(maxlen=100)
    lastbookw=None
    wait_tm=0
    ban_flag=dm.dr["ban_symbols_at_flag"]
    while True:
        fname=args.src+"/"+str(dr["min1info_tm"][min1i])+"_pred.csv"
        if os.path.exists(fname):
            try:
                if not str(dr["min1info_tm"][min1i]) in ddates:
                    tm=dr["min1info_tm"][min1i]
                    
                    last_fname=args.dst+"/"+str(dr["min1info_tm"][min1i-delta])+"_book.csv"
                    if os.path.exists(last_fname):
                        lastbookw=pd.read_csv(last_fname)["bookw"].values
                    
                    alpha=min15_alpha.readcsv_v2avg(dr, min1i, args.src)
                    alpha[ban_flag.astype(bool)]=0
                    alphas.append(alpha)
                    ori_alpha=alpha
                    if len(alphas)>2:
                        alpha=runalpha.alphaavg2(alphas)
                    bookw, long, short = runalpha.calcwtopkliqV3(dr, min1i, alpha.copy(),
                                        money=10000, tratio=0.03, lastbookw=lastbookw, 
                                        top_limit=20, scale=4,
                                        ratio_limit=100, money_limit=5000000, min_delta=1000)
                    if int((tm % 1000000)/10000) in dr["ban_hours_less"]:
                        bookw[:]=0
                        short[:]=False
                        long[:]=False
                        
                    lastbookw=bookw
                    df=pd.DataFrame({})
                    df["sid"]=dr["sids"]
                    df["bookw"]=bookw
                    df["long"]=long
                    df["short"]=short
                    df["alpha"]=ori_alpha
                    df.set_index("sid")
                    df.set_index("sid").to_csv(args.dst+"/"+str(tm)+"_book.csv")
                    print("save succ", args.dst+"/"+str(tm)+"_book.csv", flush=True)
                    #retryscp(args.dst+"/"+str(tm)+"_book.csv", "/home/ubuntu/signal/predv215_5w/")

                min1i+=delta
            except Exception as e:
                print("fail:", dr["min1info_tm"][min1i], e, flush=True)
        else:
            time.sleep(1.0)
            if wait_tm != min1i:
                print("wait new file:", dr["min1info_tm"][min1i], flush=True)
            wait_tm=min1i
    
    
