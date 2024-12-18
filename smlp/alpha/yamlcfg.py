#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 11:03:39 2023

@author: dli
"""

#!/usr/bin/env python3

import yaml
import os

gv={"yamlcfg":True}
def loadcfg(path):
    global gv
    with open(path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    for key in cfg.keys():
        assert (not key in gv), "dup key:"+key+","+gv[key]+","+cfg[key]
        gv[key]=cfg[key]
    return

if __name__ == "__main__":
    loadcfg('../econf/cfg.yaml')
