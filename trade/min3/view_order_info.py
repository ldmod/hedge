#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 20:34:39 2024

@author: crypto
"""
import sys
import os
sys.path.append(os.path.abspath(__file__+"../../../../../"))
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import shutil
import cryptoqt.data.sec_klines.sec_klines as sk
import cryptoqt.data.tools as tools 
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option("display.max_colwidth", 60)
pd.set_option('display.width', 10000)

def read_trade(day, path="/home/crypto/sec_klines_env/cryptoqt/trade/algo/data/account"):
    fname=path+"_"+day+".csv"
    # f=pd.read_csv(fname).dropna()
    f=tools.open_csv_copy(fname)['data']
    f['qty']=f['qty']*(f['buyer'].astype(int)*2-1).astype(float)
    f.rename(columns={'t_tm': 'curtm', 'qty': 'c_q'}, inplace=True)
    return f



def process_order_data(df_select):
    df_select["delta_money"]=df_select["target_money"]-df_select["cur_money"]
    df_select["cr"]=1-df_select["delta_money"]/(df_select["target_money"]-df_select["begin_money"])
    df_select["cr"]=df_select["cr"].round(2)
    df_select["expect_completed_ratio"]=df_select["expect_completed_ratio"].round(2)
    df_select["cr_avg"]=df_select["cr_avg"].round(2)
    df_select["order_volume"]=df_select["order_money"]/df_select["order_price"]
    df_select["delta_money"]=df_select["delta_money"].round(2)
    df_select["target_money"]=df_select["target_money"].round(2)
    df_select["begin_money"]=df_select["begin_money"].round(2)
    df_select["cur_money"]=df_select["cur_money"].round(2)
    df_select["order_money"]=df_select["order_money"].round(2)
    df_select["f_money"]=df_select["target_money"]-df_select["begin_money"]
    df_select["alpha"]=df_select["alpha"].round(2)
    df_select["tratio"]=df_select["tratio"].round(2)
    df_select["tr_ratio"]=df_select["tr_ratio"].round(2)
    df_select[["ls_merge", "adav", "accv", "dv", "pred", "pred_mom", "pred_merge",]]=df_select[[
        "ls_merge", "adav", "accv", "dv", "pred", "pred_mom", "pred_merge",]].round(3)
    # df_select["pmr"]=df_select["pmr"].round(2)
    # df_select["pred_sl"]=df_select["pred_sl"].round(2)
    # df_select["ysl"]=df_select["ysl"].round(2)
    # df_select["pred_bl"]=df_select["pred_bl"].round(2)
    # df_select["ybl"]=df_select["ybl"].round(2)
    df_select.rename(columns={'delta_money': 'd_money', 'target_money': 't_money', 
                              'begin_money': 'b_money', 'adaptive_shift' : "a_shift", 'expect_completed_ratio' : 'ecr',
                              "order_price":"o_price", "order_money":"o_money"}, inplace=True)
    
    select_columns=["order_id", "curtm", "symbol", "cr", "d_money", "f_money", "t_money", "b_money",
                    "cur_money", "o_money", "o_price", "min5vwap", 
                    "svwap", "bvwap",
                    # 'pmr', 
                    # "pred_sl", 
                    "ysl",
                    # "bp3_mbr", "bp3_msr", 
                    # "lsr", "ls_mom", 
                    "ls_merge", "adav", "accv", "dv", "pred", "pred_mom", "pred_merge",
                    # "pred_bl", "ybl", 
                    "order_volume",
                    "a_shift",
                    "alpha", "tratio", "tr_ratio",
                    "cr_avg", "ecr", 
                    # "svwap", "shigh", "slow", "bvwap", "bhigh", "blow"
                    ]
    df_select=df_select[select_columns]
    return df_select
if __name__ == "__main__": 
    path="/home/crypto/sec_klines_env/cryptoqt/trade/algo/data/order_info.h5"
    # path="/home/crypto/sec_klines_env/cryptoqt/bsim/data/order_info.h5"
    local_path="./order_infoV4.h5"
    shutil.copyfile(path, local_path)
    store = pd.HDFStore(local_path, mode="r")
    df=store["data"]

    # df_select=df[-10000:][select_columns]
    df_select=df[-50000000:]
    
    df_select=process_order_data(df_select)
    df=df_select.sort_values(by="curtm")
    tradedelta=5*60
    end=df["curtm"].values[-1]
    end=sk.gtmidx_i(end)-15
    end=end-end%tradedelta+tradedelta
    start=end-tradedelta*100
    
    for tidx in range(start, end, tradedelta):
        tstart=tidx
        tend=tstart+tradedelta
        order_data=df[(df["curtm"]>=sk.gtm_i(tstart)) & (df["curtm"]<=sk.gtm_i(tend))]
        trade_data=read_trade(str(sk.gtm_i(tstart))[:8])
        trade_data=trade_data[(trade_data["curtm"]>=sk.gtm_i(tstart+10)) & (trade_data["curtm"]<=sk.gtm_i(tend+10))]
        trade_data["c_m"]=trade_data["price"]*trade_data["c_q"]
        trade_data=trade_data[["order_id", "symbol", "c_m", "c_q"]]
        order_data=pd.merge(order_data, trade_data, on=["symbol", "order_id"], how="right")
        # order_data=order_data[select_columns]
        dfagg=order_data.groupby('symbol')
        df_stats=(dfagg.last())[["curtm", "cr", "f_money", "d_money", "t_money", "b_money",
                        "cur_money", "min5vwap", "svwap", "bvwap",
                        "alpha", "tratio", "tr_ratio",
                        # "pmr", "pred_sl", "ysl",
                        # "pred_bl", "ybl"
                        ]]
        
        df_stats["ysl"] = dfagg["ysl"].agg("mean").round(2)
        df_stats["o_money"]=dfagg["o_money"].agg("sum").round(2)
        df_stats["c_q"]=dfagg["c_q"].agg("sum").round(2)
        df_stats["c_m"]=dfagg["c_m"].agg("sum").round(2)
        df_stats["cost"]=(df_stats["min5vwap"]*df_stats["c_q"]-df_stats["c_m"]).round(2)
        c_price=df_stats["c_m"]/df_stats["c_q"]
        df_stats.insert(df_stats.columns.tolist().index('min5vwap'), 'c_price', c_price)
        # order_price=dfagg["order_money"].agg("sum")/(dfagg["order_volume"]).sum()
        # df_stats.insert(df_stats.columns.tolist().index('min5vwap'), 'order_price', order_price)
        # df_stats["bp"]=(df_stats["order_price"]/df_stats["min5vwap"]-1.0)*10000.0*np.sign(-1.0*df_stats["f_money"].values).round(2)
        # df_stats["cost"]=(df_stats["bp"]*df_stats["f_money"].abs()/10000.0).round(2)
        # df_stats[[ "cr_avg", "ecr", "a_shift"]]=dfagg[[ "cr_avg", "ecr", "a_shift"]].agg("mean").round(2)
        df_stats[[ "cr_avg", "a_shift"]]=dfagg[[ "cr_avg", "a_shift"]].agg("mean").round(2)
        # df_stats[]=dfagg[['pred_sl','ysl']].corr().unstack().iloc[:,1]
        df_stats=df_stats[["curtm", "cr", "f_money", "c_m", "t_money", "b_money",
                        "cur_money", "c_price", "min5vwap", "ysl", "alpha", "svwap", "bvwap",
                        "tratio", "tr_ratio", "a_shift", "cost"]]
        df_stats["cr"]=df_stats["c_m"]/df_stats["f_money"]
        df_stats=df_stats.drop(columns=["cur_money"])
        print(df_stats)
        all_delta_money=(df_stats["f_money"]).abs().sum()
        complete_money=(df_stats["c_m"]).abs().sum()
        print("complete_ratio:", complete_money/all_delta_money,
            "all_delta_money:", all_delta_money,
              "complete_money:", complete_money,
                "cost:", df_stats["cost"].sum()
              )
        order_data=order_data.drop(columns=["order_volume", "cur_money"])  
        a=0
    # print(df_select)
    # df_select[(df_select["symbol"]=="DARUSDT")]

























