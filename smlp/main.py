import os
import sys
sys.path.append(os.path.abspath(__file__+"../../../../"))
import numpy as np
import torch
from cryptoqt.smlp.alpha import canon as ln_canon
import cryptoqt.data.updatedata as ud
from functools import partial
import cryptoqt.data.datammap as dmap
import argparse
from cryptoqt.smlp.alpha.yamlcfg import gv
import time
from  cryptoqt.smlp.alpha.tools import flog as flog
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test for argparse', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--did', help='deviceid', default=0)
    parser.add_argument('--cfgpath',  help='cfg path', default='./econf/cfg1.yaml')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.did)
    print(args)

    ud.readuniverse(ud.g_data)
    lasteddownload=dmap.read_memmap(ud.g_data)
    dr=ud.g_data

    cfg=dict(dr=dr, 
             realtime="0", step="48", traindeltanum="6", learning_rate="0.00005", tvrdeltas="1",
             yamlcfgpath=args.cfgpath,
             lasteddownload=lasteddownload
             )
    lnc=ln_canon.create(cfg)
    
    # lnc.generate(0, int(1583340))
    endti=lnc.lastdi
    while True:
        if lasteddownload>=endti:
            lnc.generate(lnc.lastdi, endti)
            endti+=gv["reduce"]
        else:
            lasteddownload=dmap.gettmsize()
            flog("wait lasteddownload:", lasteddownload, endti, dr["min1info_tm"][lasteddownload], dr["min1info_tm"][endti])
            time.sleep(1.0)