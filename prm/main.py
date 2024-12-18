import os
import sys
import numpy as np
import torch
sys.path.append(os.path.abspath(__file__+"../../../../"))
print("sys path:", sys.path, flush=True)
from cryptoqt.prm.alpha import canon as ln_canon
print("canon path:", ln_canon, flush=True)
import cryptoqt.data.updatedata as ud
from functools import partial
import cryptoqt.data.datammap as dmap
import argparse
from cryptoqt.prm.alpha.yamlcfg import gv
import time
from  cryptoqt.prm.alpha.tools import flog as flog
import cryptoqt.data.sec_klines.sec_klines as sk
import datetime
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test for argparse', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--did', help='deviceid', default=1)
    parser.add_argument('--cfgpath',  help='cfg path', default='./econf/cfg1.yaml')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.did)
    print(args)

    ud.readuniverse(ud.g_data)
    # lasteddownload=dmap.read_memmap(ud.g_data)
    dr=ud.g_data
    secdata=sk.SecondData('/home/crypto/proddata/crypto/secdata/', ud.g_data["sids"])
    secdr=secdata.secdr
    llen=secdata.read_len()
    
    
    cfg=dict(dr=dr, secdr=secdr, yamlcfgpath=args.cfgpath)
    lnc=ln_canon.create(cfg)
    
    # lnc.generate(0, int(7*31*1440*60))
    endti=lnc.lastsi
    tmtag=0
    while True:
        if llen>=endti:
            flog("generate llen:", lnc.lastsi, endti, sk.gtm_i(endti), datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'), flush=True)
            lnc.generate(lnc.lastsi, endti)
            endti+=gv["reduce"]
        else:
            if tmtag != endti:
                flog("begin read_len:", llen, endti, sk.gtm_i(llen), sk.gtm_i(endti), datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'), flush=True)
            llen=secdata.read_len()
            if tmtag != endti:
                flog("wait llen:", llen, endti, sk.gtm_i(llen), sk.gtm_i(endti), datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'), flush=True)
            tmtag=endti
            time.sleep(0.01)
