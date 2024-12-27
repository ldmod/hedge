from datetime import datetime, timezone, timedelta
import time
import sys
import csv
import numpy as np
import pandas as pd
from binance.um_futures import UMFutures
from dingtalkchatbot.chatbot import DingtalkChatbot
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
import logging
import os
import yaml
import warnings

sys.path.append(os.path.abspath(__file__+"../../../../../"))
import cryptoqt.data.updatedata as ud
import cryptoqt.data.datammap as dmap
import cryptoqt.trade.min60.um_client_trade as um_client_trade
import cryptoqt.data.data_manager as dm
warnings.simplefilter("ignore", category=RuntimeWarning)
np.seterr(invalid='ignore')
import cryptoqt.data.tools as tools 
import cryptoqt.trade.min60.view_order_info as view_order_info
# 初始化日志系统
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 初始状态变量
total_bp_cost = 0.001
total_bp_money = 0.001
today = datetime.now().strftime("%Y%m%d")
total_delta_cost = 0.001
# 信号文件后缀
signal_suffix = '_book.csv'

def notify(msg: str):
    webhook = "https://oapi.dingtalk.com/robot/send?access_token=08981a30db7ee421f6f910cb1dbe9b722bb18420b1c54d9b5cdad300470d2cda"
    xiaoding = DingtalkChatbot(webhook)
    xiaoding.send_text(msg, is_at_all=False)


def update_positions(clients, position_value_map: dict) -> bool:
    try:
        client=clients.get_um_client()
        data_dict = client.client.account()
        clients.recycle(client)
        if not data_dict:
            logger.error("data is null")
            return True
        if "positions" not in data_dict:
            logger.error("positions not exist")
            return True

        # update: position_value_map
        for position in data_dict["positions"]:
            symbol = position["symbol"]
            notional = float(position["notional"])
            if symbol in position_value_map:
                position_value_map[symbol] = notional

        return False
    except Exception as e:
        logger.error(f"update_positions: {str(e)}")
        return True


def int_to_ms(yyyymmddhhmmss: int) -> int:
    dt_str = str(yyyymmddhhmmss)
    dt = datetime.strptime(dt_str, '%Y%m%d%H%M%S').replace(tzinfo=timezone.utc)
    timestamp = int(dt.timestamp() * 1000)
    return timestamp

@retry(stop=stop_after_attempt(50), wait=wait_fixed(15), retry=retry_if_exception_type(Exception))
def get_account_value(clients) -> float:
    try:
        client=clients.get_um_client()
        account_info = client.client.balance()
        usdt_balance = bnb_balance = usdt_crossUnPnl = 0.0
        for entry in account_info:
            if entry['asset'] == 'USDT':
                usdt_balance = float(entry['balance'])
                usdt_crossUnPnl = float(entry['crossUnPnl'])
            elif entry['asset'] == 'BNB':
                bnb_balance = float(entry['balance'])

        depth = client.client.depth(symbol='BNBUSDT', limit=5)
        price = round(float(depth['bids'][0][0]), 2)
        clients.recycle(client)
        account_value = round(bnb_balance * price + usdt_balance + usdt_crossUnPnl, 2)
        return account_value
    except Exception as e:
        logger.error(f"get_account_value error: {str(e)}")
        raise

# Logger Init
def get_logger(logger_name, log_file, level=logging.INFO):
    logger = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s', "%Y-%m-%d %H:%M:%S")
    fileHandler = logging.FileHandler(log_file, mode='a')
    fileHandler.setFormatter(formatter)

    logger.setLevel(level)
    logger.addHandler(fileHandler)
    return logger

if __name__ == '__main__':
    with open('./config/cost.yaml', 'r') as file:
        cfg = yaml.safe_load(file)
    
    # logfile = cfg["log_path"]
    # logger = get_logger(f"costinfo", logfile)
    # 定义需要处理的分钟列表
    lookback_mins = cfg["lookback_mins"]
    min_delta = cfg["min_delta"]
    trade_min = cfg["trade_min"]
    timeout = 30
    clients = um_client_trade.UmClient(cfg["um_client"], UMFutures)
    dm.init()
    dr=dm.dr
    available_token_map = {}
    for sid in dr["sids"]:
        available_token_map[sid]=0
    
    current_time = datetime.now()
    start_of_day = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
    current_index = int(start_of_day.strftime("%Y%m%d%H%M%S"))
    # current_index=20241003180000
    last_sec = 0
    while True:
        time.sleep(1)
        cur_sec = int(time.time())
        if  cur_sec > last_sec + (min_delta * 60) + 300:
            start_time = int(cur_sec/(min_delta * 60))*(min_delta * 60) - (min_delta * 60) 
            start_index = tools.tmu2i(start_time*1000)
            total_account_value = get_account_value(clients)
            
            if str(start_index).endswith("000000"):
                notify("5Min Start: " + str(start_index) + ", Cost: " + str(total_bp_cost) + " ,Money: " + str(
                    total_bp_money) + " , BP: " + str(total_bp_cost / total_bp_money * 10000) + " ,Account: " + str(
                    total_account_value))
                total_bp_cost = 0
                total_bp_money = 0
                total_delta_cost = 0
                # today = start_time.strftime("%Y%m%d")

            acc_value = get_account_value(clients)
            logger.info(f"账户净值$: {acc_value}")
            update_positions(clients, available_token_map)
            long_value = sum(value for value in available_token_map.values() if value > 0)
            short_value = sum(value for value in available_token_map.values() if value < 0)
            logger.info(f"Long仓位$: {long_value}, Short仓位$: {short_value}")

            try:
                df = pd.DataFrame()
                df["symbol"] = dr["sids"]
                df = df.set_index("symbol")

                while current_index <= int(start_index):
                    cur_day=int(current_index/1000000)
                    s_tm = tools.tmi2u(current_index)
                    e_tm = s_tm + trade_min * 60 * 1000
                    bookw_tm = s_tm - min_delta * 60 * 1000
                    min1i = dm.get_minidx(tools.tmi2u(current_index))
                    dm.wait_minidx(min1i+lookback_mins)
                    
                    acc_value = tools.open_csv_copy(cfg["acc_info_path"]+"_"+str(cur_day)+".h5")['data']
                    if acc_value[acc_value["curtm"]==current_index].shape[0] > 0:
                        acc_value = float(acc_value[acc_value["curtm"]==current_index].iloc[0]["value"])
                    else:
                        acc_value=np.nan
                    
                    pos = tools.open_csv_copy(cfg["pos_info_path"]+"_"+str(cur_day)+".h5")['data']
                    pos = pos[pos["curtm"]==current_index]
                    ret = np.nanmean(dr["min1info_vwap"][min1i:min1i + lookback_mins], axis=0) / \
                        np.nanmean(dr["min1info_vwap"][min1i - min_delta:min1i - min_delta + lookback_mins], axis=0) - 1.0
                    target = pd.read_csv(cfg["signal_path"]+"/"+str(tools.tmu2i(bookw_tm))+"_book.csv")
                    target = target[["sid","bookw", "alpha"]]
                    target["ret"]=ret
                    target.rename(columns={'sid': 'symbol'}, inplace=True)
                    target=pd.merge(target, pos, on=["symbol"], how="left").set_index("symbol")
                    target["delta"]=target["vol"]*np.nanmean(
                        dr["min1info_vwap"][min1i - min_delta:min1i - min_delta + lookback_mins], axis=0)-target["bookw"]
                    target["deltacost"] = target["delta"]*target["ret"]
                    target=target.fillna(0)
                    delta_costsum=target["deltacost"].sum()
                    total_delta_cost+=delta_costsum
                    delta_ratio=1-target["delta"].abs().sum()/target["bookw"].abs().sum()
                    logger.info(f"{tools.tmu2i(bookw_tm)} - delta_ratio$: {delta_ratio}")
                    logger.info(f"delta_costsum$: {delta_costsum}")
                    show=target[((target["bookw"].abs()>50) | (target["money"].abs()>50))]
                    print(f"delta detail:\n{show}", flush=True)
                    
                    
                    logger.info(f"---------- 本次信号(current_index): {current_index} - {acc_value} ------------")
                    df["vwap"] = np.nanmean(dr["min1info_vwap"][min1i:min1i + lookback_mins], axis=0)
                              
                    trade_data=tools.open_csv_copy(cfg["trade_info_path"]+"_"+str(cur_day)+".h5")['data']
                    trade_data = trade_data[(trade_data["t_tm"] >= tools.tmu2i(s_tm+10000)) & (trade_data["t_tm"] < tools.tmu2i(e_tm+10000))]
                    trade_data['qty']=trade_data['qty']*(trade_data['buyer'].astype(int)*2-1).astype(float)
                    trade_data.rename(columns={'t_tm': 'curtm', 'qty': 'c_q'}, inplace=True)
                    trade_data.sort_values(by="curtm", inplace=True)
                    trade_data["c_m"]=trade_data["price"]*trade_data["c_q"]
                    trade_data=trade_data[["order_id", "symbol", "c_m", "c_q"]]
                    df_merge=trade_data[["symbol", "c_m", "c_q"]].groupby('symbol').agg("sum")
                    df_merge["c_p"] = df_merge["c_m"]/df_merge["c_q"]
                    df_stats=pd.merge(df, df_merge, on=["symbol"], how="right")

                    df_stats["cost"]=(df_stats["vwap"]*df_stats["c_q"]-df_stats["c_m"]).round(2)
                    complete_money=(df_stats["c_m"]).abs().sum()
                    
                    logger.info(f"本次交易金额$: {complete_money}")
                    logger.info(f"本次冲击成本$: {df_stats['cost'].sum(axis=0)}")
                    logger.info(f"本次冲击成本bp: {df_stats['cost'].sum(axis=0) / complete_money * 10000}")
                    total_bp_cost += df_stats['cost'].sum(axis=0)
                    total_bp_money += complete_money
                    logger.info(f"今日累计冲击成本$: {total_bp_cost}")
                    logger.info(f"今日累计交易金额$: {total_bp_money}")
                    logger.info(f"今日累计冲击成本bp: {total_bp_cost / total_bp_money * 10000}")
                    logger.info(f"\ndetail: \n{df_stats}")
                        

                    # 信号步进
                    current_index = tools.tmu2i(tools.tmi2u(current_index) + min_delta * 60 * 1000)

            except Exception as e:
                logger.error(f"exception: {str(e)}")
                break
            last_sec=int(cur_sec/(min_delta * 60))*(min_delta * 60)
