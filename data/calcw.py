import os
import sys
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


def retryscp(scp_client, src, dst, maxcnt=100):
    cnt=0
    while cnt < maxcnt:
        try:
            scp_client.put(src, dst)
        except Exception as e:
            print(e)
            print("fail:" , src, flush=True)
            cnt+=1
        else:
            print("upload succ", src)
            return
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test for argparse', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--src', help='src', default='/home/crypto/cryptoqt/smlp/model_states/infmodel/tsmlp15/res')
    parser.add_argument('--dst',  help='dst', default='/home/crypto/signal/pred15top10')
    parser.add_argument('--remotepath',  help='remotepath',
                        default='ubuntu@ec2-3-99-203-76.ca-central-1.compute.amazonaws.com:/home/ubuntu/signal/smlp15/')
    args = parser.parse_args()
    os.makedirs(args.dst, exist_ok=True)
    
    ssh_client = paramiko.SSHClient()
    pkey='/home/crypto/.ssh/id_rsa'
    key=paramiko.RSAKey.from_private_key_file(pkey)
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy)
    ssh_client.connect('ec2-3-99-203-76.ca-central-1.compute.amazonaws.com',username = 'ubuntu',pkey=key)
    scp_client = SCPClient(ssh_client.get_transport(), socket_timeout=1.0)
    
    ud.readuniverse(ud.g_data)
    lasteddownload=dmap.read_memmap(ud.g_data)
    dr=ud.g_data
    dstfiles=os.listdir(args.dst)
    ddates={}
    for x in dstfiles:
        ddates[x.split('_')[0]]=1
    min1i=100*conts.daymincnt
    delta=15
    while True:
        fname=args.src+"/"+str(dr["min1info_tm"][min1i])+"_pred.csv"
        if os.path.exists(fname):
            if not str(dr["min1info_tm"][min1i]) in ddates:
                tm=dr["min1info_tm"][min1i]
                alpha=min15_alpha.readcsv(dr, min1i, args.src)
                bookw, long, short = runalpha.calcwtopk(alpha, 10)
                df=pd.DataFrame({})
                df["sid"]=dr["sids"]
                df["bookw"]=bookw
                df["long"]=long
                df["short"]=short
                df.set_index("sid")
                df.set_index("sid").to_csv(args.dst+"/"+str(tm)+"_book.csv")
                retryscp(scp_client, args.dst+"/"+str(tm)+"_book.csv", "/home/ubuntu/signal/pred15top10/")

            min1i+=delta
        else:
            time.sleep(2.0)
            print("wait new file:", dr["min1info_tm"][min1i])
    
    

