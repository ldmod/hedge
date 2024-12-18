#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 14:35:19 2023

@author: dli
"""
import datetime
class TimeProfilingLoop(object):
    def __init__(self, name):
        self.deltas=[]
        self._s=datetime.datetime.now()
        self._subt={}
        self._name=name
    def end(self):
        self._e=datetime.datetime.now()
        delta=(self._e-self._s).seconds*1000.0+(self._e-self._s).microseconds/1000.0
        self.deltas.append(delta)
    def restart(self):
        self._s=datetime.datetime.now()
    def add(self, name):
        if not name in self._subt:
            self._subt[name]=TimeProfilingLoop(name)  
        else:
            self._subt[name].restart()
        return self._subt[name]
    def to_string(self):
        # assert (self._e != None) << self._name+"timeend"
        deltastr=str(sum(self.deltas))+":"+str(sum(self.deltas)/len(self.deltas))+":"+str(len(self.deltas))
        s='{'+self._name+':'+deltastr
        for k,v in self._subt.items():
            s+=v.to_string()+'--'
        s+='}'
        return s