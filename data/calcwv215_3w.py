import os
import sys
sys.path.append(os.path.abspath(__file__+"../../../../"))
import numpy as np
import torch
import cryptoqt.data.updatedata as ud
from functools import partial
import cryptoqt.data.datammap as dmap
import cryptoqt.data.tools as tools
import argparse
import time
import pandas as pd
import cryptoqt.bsim.runalpha as runalpha
import cryptoqt.data.constants as conts
import cryptoqt.alpha.min15_alpha as min15_alpha
import cryptoqt.alpha.min30_alpha as min30_alpha
import paramiko
from scp import SCPClient
from collections import deque

dsthostname='ubuntu@ec2-18-166-74-173.ap-east-1.compute.amazonaws.com'
def retryscp(src, dst, maxcnt=100):
    cnt=0
    while cnt < maxcnt:
        try:
            cmd='scp '+src+' '+dsthostname+':'+dst
            res=os.system(cmd)
            if res==0:
                print("upload succ", src, flush=True)
                return
            else:
                print("fail:" , src, flush=True)
                cnt+=1
        except Exception as e:
            print(e)
            print("fail:" , src, flush=True)
            cnt+=1
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test for argparse', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--src', help='src', default='/home/crypto/smlpv2/cryptoqt/smlp/model_states/infmodel/tsmlpv215/res')
    parser.add_argument('--dst',  help='dst', default='/home/crypto/signal/predv215_3w')

    args = parser.parse_args()
    os.makedirs(args.dst, exist_ok=True)

    ud.readuniverse(ud.g_data)
    lasteddownload=dmap.read_memmap(ud.g_data)
    dr=ud.g_data
    
    ban_symbols=["EOSUSDT","XTZUSDT","CRVUSDT","FLMUSDT","LITUSDT","REEFUSDT","XEMUSDT","LINAUSDT",
                 "STMXUSDT","DENTUSDT","OGNUSDT","GTCUSDT","ATAUSDT","CELOUSDT","FLOWUSDT","BIGTIMEUSDT",
                   "TNSRUSDT", "DYDXUSDT",  "SNXUSDT",
                   "BLURUSDT",   "TUSDT", "BICOUSDT"
               ]
    ban_flag=np.zeros(dr["sids"].shape)
    for symbol in ban_symbols:
        ban_flag[dr["sids"].tolist().index(symbol)]=1
        
    dstfiles=os.listdir(args.dst)
    ddates={}
    for x in dstfiles:
        ddates[x.split('_')[0]]=1
    min1i=dr["min1info_tm"].tolist().index(20240831000000)
    delta=15
    alphas=deque(maxlen=100)
    lastbookw=None
    wait_tm=0
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
                    if len(alphas)>2:
                        alpha=runalpha.alphaavg2(alphas)
                    bookw, long, short = runalpha.calcwtopkliqV3(dr, min1i, alpha.copy(), money=30000, tratio=1.0, lastbookw=lastbookw)
                    lastbookw=bookw
                    df=pd.DataFrame({})
                    df["sid"]=dr["sids"]
                    df["bookw"]=bookw
                    df["long"]=long
                    df["short"]=short
                    df.set_index("sid")
                    df.set_index("sid").to_csv(args.dst+"/"+str(tm)+"_book.csv")
                    print("save succ", args.dst+"/"+str(tm)+"_book.csv", flush=True)
                    #retryscp(args.dst+"/"+str(tm)+"_book.csv", "/home/ubuntu/signal/predv215_3w/")

                min1i+=delta
            except Exception as e:
                print("fail:", dr["min1info_tm"][min1i], e, flush=True)
        else:
            time.sleep(1.0)
            if wait_tm != min1i:
                print("wait new file:", dr["min1info_tm"][min1i], flush=True)
            wait_tm=min1i
    
    
