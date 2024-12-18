#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 10:44:10 2023

@author: dli
"""

import torch
from alpha.tools import *
import gc
from memory_profiler import profile
from alpha.gpu_mem_track import MemTracker
import inspect

def testcss():
    setup_seed(0)
    deviceid=0
    x0=torch.randn(16, 1024, 1024, device=deviceid, requires_grad=True)
    x1=torch.randn(16, 1024, 1024, device=deviceid, requires_grad=True)

    a=torch.cat([x0.detach(), x1.detach()], dim=-1)
    a.requires_grad=True
    a.retain_grad()
    # a=torch.randn(16, 1024, 2048, device=deviceid, requires_grad=True)
    print("a ref count:", sys.getrefcount(a), a.requires_grad, a.shape)
    setup_seed(0)
    cssm1=CssModel(2048,2048).to(deviceid)
    cssm2=CssModel(2048,2048).to(deviceid)
    b=cssm1(a)
    b=cssm2(b)
    b.sum().backward()
    
    setup_seed(0)
    cssm1opt=OptCssModel(2048,2048).to(deviceid)
    cssm2opt=OptCssModel(2048,2048).to(deviceid)
    # b=torch.cat([x0, x1], dim=-1)
    # b.retain_grad()
    c=cssm1opt([x0, x1])
    c=cssm2opt(c)
    c.sum().backward(retain_graph=False)
    
    xxgrad=torch.cat([x0.grad, x1.grad], dim=-1)
    print((a.grad-xxgrad).std())
    print((cssm1.hidden1c[0].weight.grad-cssm1opt.hidden1c.weight.grad).std())
    print((cssm1.hidden1c[0].bias.grad-cssm1opt.hidden1c.bias.grad).std())
    meminfo(deviceid)
    print(torch.cuda.max_memory_allocated()/1024/1024/1024)
    
def testcss2d():
    setup_seed(0)
    deviceid=0
    x0=torch.randn(1024, 1024, device=deviceid, requires_grad=True)
    x1=torch.randn(1024, 1024, device=deviceid, requires_grad=True)

    a=torch.cat([x0.detach(), x1.detach()], dim=-1)
    a.requires_grad=True
    a.retain_grad()
    # a=torch.randn(16, 1024, 2048, device=deviceid, requires_grad=True)
    print("a ref count:", sys.getrefcount(a), a.requires_grad, a.shape)
    setup_seed(0)
    cssm1=CssModel(2048,2048).to(deviceid)
    cssm2=CssModel(2048,2048).to(deviceid)
    b=cssm1(a)
    b=cssm2(b)
    b.sum().backward()
    
    setup_seed(0)
    cssm1opt=OptCssModel(2048,2048).to(deviceid)
    cssm2opt=OptCssModel(2048,2048).to(deviceid)
    b=torch.cat([x0, x1], dim=-1)
    b.retain_grad()
    c=cssm1opt([x0, x1])
    c=cssm2opt(c)
    c.sum().backward(retain_graph=False)
    
    xxgrad=torch.cat([x0.grad, x1.grad], dim=-1)
    print((a.grad-xxgrad).std())
    print((cssm1.hidden1c[0].weight.grad-cssm1opt.hidden1c.weight.grad).std())
    print((cssm1.hidden1c[0].bias.grad-cssm1opt.hidden1c.bias.grad).std())
    meminfo(deviceid)
    print(torch.cuda.max_memory_allocated()/1024/1024/1024)
    
def testbn():
    deviceid=0
    inputsize=1024
    setup_seed(0)
    x0=torch.randn(1024, 1024, device=deviceid, requires_grad=True)+2
    x0.retain_grad()
    setup_seed(0)
    x1=torch.randn(1024, 1024, device=deviceid, requires_grad=True)+2
    x1.retain_grad()

    setup_seed(0)
    bn1=nn.BatchNorm1d(inputsize, track_running_stats=False).to(deviceid)
    b=bn1(x0)
    b.sum().backward()
    
    setup_seed(0)
    bn1opt=OptBnModel(inputsize).to(deviceid)
    c=bn1opt(x1)
    c.sum().backward()
    
    print((b-c).abs().mean())
    print((x0.grad-x1.grad).abs().mean())
    print((bn1.weight.grad-bn1opt.bnlayer.weight.grad).abs().mean())
    print((bn1.bias.grad-bn1opt.bnlayer.bias.grad).abs().mean())
    meminfo(deviceid)
    print(torch.cuda.max_memory_allocated()/1024/1024/1024)
    
def testbncss():
    setup_seed(0)
    deviceid=0
    x0=torch.randn(4, 8, 512, 1024, device=deviceid, requires_grad=True)*2+1
    x1=torch.randn(4, 8, 512, 1024, device=deviceid, requires_grad=True)*2+1
    # x0=torch.randn(4, 512, 1024, device=deviceid, requires_grad=True)*2+1
    # x1=torch.randn(4, 512, 1024, device=deviceid, requires_grad=True)*2+1
    # x0=torch.randn(512, 1024, device=deviceid, requires_grad=True)*2+1
    # x1=torch.randn(512, 1024, device=deviceid, requires_grad=True)*2+1
    x0.retain_grad()
    x1.retain_grad()

    a=torch.cat([x0.detach(), x1.detach()], dim=-1)
    a.requires_grad=True
    a.retain_grad()
    # a=torch.randn(16, 1024, 2048, device=deviceid, requires_grad=True)
    print("a ref count:", sys.getrefcount(a), a.requires_grad, a.shape)
    setup_seed(0)
    cssm1=CssModel(2048,768, bnlen=1, dropout=0.1).to(deviceid)
    cssm2=CssModel(768,512, bnlen=1, dropout=0.1).to(deviceid)
    setup_seed(0)
    b=cssm1(a)
    b.retain_grad()
    b1=b
    b1=cssm2(b)
    b1.sum().backward()
    
    setup_seed(0)
    cssm1opt=OptCssModel(2048,768, bnlen=1, dropout=0.1).to(deviceid)
    cssm2opt=OptCssModel(768,512, bnlen=1, dropout=0.1).to(deviceid)
    setup_seed(0)
    c=cssm1opt([x0, x1])
    c.retain_grad()
    c1=c
    c1=cssm2opt(c)
    c1.sum().backward(retain_graph=False)
    
    xxgrad=torch.cat([x0.grad, x1.grad], dim=-1)
    
    frame = inspect.currentframe()     
    gpu_tracker = MemTracker(frame)      # 创建显存检测对象
    gpu_tracker.track()                  # 开始检测
        
    print((a.grad-xxgrad).std())
    print((cssm1.hidden1c[0].weight.grad-cssm1opt.hidden1c.weight.grad).std())
    print((cssm1.hidden1c[0].bias.grad-cssm1opt.hidden1c.bias.grad).std())
    meminfo(deviceid)
    print(torch.cuda.max_memory_allocated()/1024/1024/1024)
    
if __name__ == "__main__":
    testbncss()
    print("end")