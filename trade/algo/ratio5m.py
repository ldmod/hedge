import sys
import os
import random
sys.path.append(os.path.abspath(__file__+"../../../../../"))
from datetime import datetime, timezone, timedelta
import time
import csv
from binance.um_futures import UMFutures
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
import logging
import os
import copy
import yaml
import numpy as np
import pandas as pd
import cryptoqt.data.updatedata as ud
import cryptoqt.data.datammap as dmap
import warnings
import cryptoqt.trade.algo.um_client_trade as um_client_trade
warnings.simplefilter("ignore", category=RuntimeWarning)
np.seterr(invalid='ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

cal_minute_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 0]
ret_minute_list = [2, 7, 12, 17, 22, 27, 32, 37, 42, 47, 52, 57]

signal_path = "/home/crypto/signal/predv25_1w"
signal_suffix = '_book.csv'


today = datetime.now().strftime("%Y%m%d")
available_token_map = {'BTCUSDT': 0.0, 'ETHUSDT': 0.0, 'BCHUSDT': 0.0, 'XRPUSDT': 0.0, 'EOSUSDT': 0.0,
                       'LTCUSDT': 0.0, 'TRXUSDT': 0.0, 'ETCUSDT': 0.0, 'LINKUSDT': 0.0, 'XLMUSDT': 0.0,
                       'ADAUSDT': 0.0, 'XMRUSDT': 0.0, 'DASHUSDT': 0.0, 'ZECUSDT': 0.0, 'XTZUSDT': 0.0,
                       'ATOMUSDT': 0.0, 'ONTUSDT': 0.0, 'IOTAUSDT': 0.0, 'BATUSDT': 0.0,
                       'VETUSDT': 0.0, 'NEOUSDT': 0.0, 'QTUMUSDT': 0.0, 'IOSTUSDT': 0.0, 'THETAUSDT': 0.0,
                       'ALGOUSDT': 0.0, 'ZILUSDT': 0.0, 'KNCUSDT': 0.0, 'ZRXUSDT': 0.0, 'COMPUSDT': 0.0,
                       'OMGUSDT': 0.0, 'DOGEUSDT': 0.0, 'SXPUSDT': 0.0, 'KAVAUSDT': 0.0, 'BANDUSDT': 0.0,
                       'RLCUSDT': 0.0, 'WAVESUSDT': 0.0, 'MKRUSDT': 0.0, 'SNXUSDT': 0.0, 'DOTUSDT': 0.0,
                       'DEFIUSDT': 0.0, 'YFIUSDT': 0.0, 'BALUSDT': 0.0, 'CRVUSDT': 0.0, 'TRBUSDT': 0.0,
                       'RUNEUSDT': 0.0, 'SUSHIUSDT': 0.0, 'EGLDUSDT': 0.0, 'SOLUSDT': 0.0, 'ICXUSDT': 0.0,
                       'STORJUSDT': 0.0, 'BLZUSDT': 0.0, 'UNIUSDT': 0.0, 'AVAXUSDT': 0.0, 'FTMUSDT': 0.0,
                       'ENJUSDT': 0.0, 'FLMUSDT': 0.0, 'RENUSDT': 0.0, 'KSMUSDT': 0.0, 'NEARUSDT': 0.0,
                       'AAVEUSDT': 0.0, 'FILUSDT': 0.0, 'RSRUSDT': 0.0, 'LRCUSDT': 0.0, 'MATICUSDT': 0.0,
                       'OCEANUSDT': 0.0, 'CVCUSDT': 0.0, 'BELUSDT': 0.0, 'CTKUSDT': 0.0, 'AXSUSDT': 0.0,
                       'ALPHAUSDT': 0.0, 'ZENUSDT': 0.0, 'SKLUSDT': 0.0, 'GRTUSDT': 0.0, '1INCHUSDT': 0.0,
                       'CHZUSDT': 0.0, 'SANDUSDT': 0.0, 'ANKRUSDT': 0.0, 'LITUSDT': 0.0, 'UNFIUSDT': 0.0,
                       'REEFUSDT': 0.0, 'RVNUSDT': 0.0, 'SFPUSDT': 0.0, 'XEMUSDT': 0.0, 'COTIUSDT': 0.0,
                       'CHRUSDT': 0.0, 'MANAUSDT': 0.0, 'ALICEUSDT': 0.0, 'HBARUSDT': 0.0, 'ONEUSDT': 0.0,
                       'LINAUSDT': 0.0, 'STMXUSDT': 0.0, 'DENTUSDT': 0.0, 'CELRUSDT': 0.0, 'HOTUSDT': 0.0,
                       'MTLUSDT': 0.0, 'OGNUSDT': 0.0, 'NKNUSDT': 0.0, 'SCUSDT': 0.0, 'DGBUSDT': 0.0,
                       '1000SHIBUSDT': 0.0, 'BAKEUSDT': 0.0, 'GTCUSDT': 0.0, 'BTCDOMUSDT': 0.0, 'IOTXUSDT': 0.0,
                       'RAYUSDT': 0.0, 'C98USDT': 0.0, 'MASKUSDT': 0.0, 'ATAUSDT': 0.0, 'DYDXUSDT': 0.0,
                       '1000XECUSDT': 0.0, 'GALAUSDT': 0.0, 'CELOUSDT': 0.0, 'ARUSDT': 0.0, 'KLAYUSDT': 0.0,
                       'ARPAUSDT': 0.0, 'CTSIUSDT': 0.0, 'LPTUSDT': 0.0, 'ENSUSDT': 0.0, 'PEOPLEUSDT': 0.0,
                       'ROSEUSDT': 0.0, 'DUSKUSDT': 0.0, 'FLOWUSDT': 0.0, 'IMXUSDT': 0.0, 'API3USDT': 0.0,
                       'GMTUSDT': 0.0, 'APEUSDT': 0.0, 'WOOUSDT': 0.0, 'FTTUSDT': 0.0, 'JASMYUSDT': 0.0,
                       'DARUSDT': 0.0, 'OPUSDT': 0.0, 'INJUSDT': 0.0, 'STGUSDT': 0.0, 'SPELLUSDT': 0.0,
                       '1000LUNCUSDT': 0.0, 'LUNA2USDT': 0.0, 'LDOUSDT': 0.0, 'CVXUSDT': 0.0, 'ICPUSDT': 0.0,
                       'APTUSDT': 0.0, 'QNTUSDT': 0.0, 'FETUSDT': 0.0, 'FXSUSDT': 0.0, 'HOOKUSDT': 0.0,
                       'MAGICUSDT': 0.0, 'TUSDT': 0.0, 'HIGHUSDT': 0.0, 'MINAUSDT': 0.0, 'ASTRUSDT': 0.0,
                       'AGIXUSDT': 0.0, 'PHBUSDT': 0.0, 'GMXUSDT': 0.0, 'CFXUSDT': 0.0, 'STXUSDT': 0.0,
                       'BNXUSDT': 0.0, 'ACHUSDT': 0.0, 'SSVUSDT': 0.0, 'CKBUSDT': 0.0, 'PERPUSDT': 0.0,
                       'TRUUSDT': 0.0, 'LQTYUSDT': 0.0, 'USDCUSDT': 0.0, 'IDUSDT': 0.0, 'ARBUSDT': 0.0,
                       'JOEUSDT': 0.0, 'TLMUSDT': 0.0, 'AMBUSDT': 0.0, 'LEVERUSDT': 0.0, 'RDNTUSDT': 0.0,
                       'HFTUSDT': 0.0, 'XVSUSDT': 0.0, 'BLURUSDT': 0.0, 'EDUUSDT': 0.0, 'IDEXUSDT': 0.0,
                       'SUIUSDT': 0.0, '1000PEPEUSDT': 0.0, '1000FLOKIUSDT': 0.0, 'UMAUSDT': 0.0, 'RADUSDT': 0.0,
                       'KEYUSDT': 0.0, 'COMBOUSDT': 0.0, 'NMRUSDT': 0.0, 'MAVUSDT': 0.0, 'MDTUSDT': 0.0,
                       'XVGUSDT': 0.0, 'WLDUSDT': 0.0, 'PENDLEUSDT': 0.0, 'ARKMUSDT': 0.0, 'AGLDUSDT': 0.0,
                       'YGGUSDT': 0.0, 'DODOXUSDT': 0.0, 'BNTUSDT': 0.0, 'OXTUSDT': 0.0, 'SEIUSDT': 0.0,
                       'CYBERUSDT': 0.0, 'HIFIUSDT': 0.0, 'ARKUSDT': 0.0, 'FRONTUSDT': 0.0, 'GLMRUSDT': 0.0,
                       'BICOUSDT': 0.0, 'STRAXUSDT': 0.0, 'LOOMUSDT': 0.0, 'BIGTIMEUSDT': 0.0, 'BONDUSDT': 0.0,
                       'ORBSUSDT': 0.0, 'STPTUSDT': 0.0, 'WAXPUSDT': 0.0, 'BSVUSDT': 0.0, 'RIFUSDT': 0.0,
                       'POLYXUSDT': 0.0, 'GASUSDT': 0.0, 'POWRUSDT': 0.0, 'SLPUSDT': 0.0, 'TIAUSDT': 0.0,
                       'SNTUSDT': 0.0, 'CAKEUSDT': 0.0, 'MEMEUSDT': 0.0, 'TWTUSDT': 0.0, 'TOKENUSDT': 0.0,
                       'ORDIUSDT': 0.0, 'STEEMUSDT': 0.0, 'BADGERUSDT': 0.0, 'ILVUSDT': 0.0, 'NTRNUSDT': 0.0,
                       'KASUSDT': 0.0, 'BEAMXUSDT': 0.0, '1000BONKUSDT': 0.0, 'PYTHUSDT': 0.0, 'SUPERUSDT': 0.0,
                       'USTCUSDT': 0.0, 'ONGUSDT': 0.0, 'ETHWUSDT': 0.0, 'JTOUSDT': 0.0, '1000SATSUSDT': 0.0,
                       'AUCTIONUSDT': 0.0, '1000RATSUSDT': 0.0, 'ACEUSDT': 0.0, 'MOVRUSDT': 0.0, 'NFPUSDT': 0.0,
                       'AIUSDT': 0.0, 'XAIUSDT': 0.0, 'WIFUSDT': 0.0, 'MANTAUSDT': 0.0, 'ONDOUSDT': 0.0,
                       'LSKUSDT': 0.0, 'ALTUSDT': 0.0, 'JUPUSDT': 0.0, 'ZETAUSDT': 0.0, 'RONINUSDT': 0.0,
                       'DYMUSDT': 0.0, 'OMUSDT': 0.0, 'PIXELUSDT': 0.0, 'STRKUSDT': 0.0, 'MAVIAUSDT': 0.0,
                       'GLMUSDT': 0.0, 'PORTALUSDT': 0.0, 'TONUSDT': 0.0, 'AXLUSDT': 0.0, 'MYROUSDT': 0.0,
                       'METISUSDT': 0.0, 'AEVOUSDT': 0.0, 'VANRYUSDT': 0.0, 'BOMEUSDT': 0.0, 'ETHFIUSDT': 0.0,
                       'ENAUSDT': 0.0, 'WUSDT': 0.0, 'TNSRUSDT': 0.0, 'SAGAUSDT': 0.0, 'TAOUSDT': 0.0,
                       'OMNIUSDT': 0.0, 'REZUSDT': 0.0, 'BBUSDT': 0.0, 'NOTUSDT': 0.0, 'TURBOUSDT': 0.0,
                       'IOUSDT': 0.0, 'ZKUSDT': 0.0, 'MEWUSDT': 0.0, 'LISTAUSDT': 0.0, 'ZROUSDT': 0.0,
                       'RENDERUSDT': 0.0, 'BANANAUSDT': 0.0, 'RAREUSDT': 0.0, 'GUSDT': 0.0, 'SYNUSDT': 0.0}

available_token_map_volume = copy.deepcopy(available_token_map)

@retry(stop=stop_after_attempt(50), wait=wait_fixed(15), retry=retry_if_exception_type(Exception))
def update_positions(clients, position_value_map, position_volume_map) -> bool:
    try:
        client=clients.get_um_client()
        data_dict = client.client.account()
        clients.recycle(client)
        if not data_dict:
            logger.error("data is null")
            raise
        if "positions" not in data_dict:
            logger.error("positions not exist")
            raise

        # update: position_value_map
        for position in data_dict["positions"]:
            symbol = position["symbol"]
            notional = round(float(position["notional"]), 2)
            positionAmt = round(float(position["positionAmt"]), 6)
            if symbol in position_value_map:
                position_value_map[symbol] = notional
            if symbol in position_volume_map:
                position_volume_map[symbol] = positionAmt
    except Exception as e:
        logger.error(f"update_positions: {str(e)}")
        raise

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

        depth = client.depth(symbol='BNBUSDT', limit=5)
        price = round(float(depth['bids'][0][0]), 2)
        clients.recycle(client)
        account_value = round(bnb_balance * price + usdt_balance + usdt_crossUnPnl, 2)
        return account_value
    except Exception as e:
        logger.error(f"get_account_value error: {str(e)}")
        raise

def get_current_tm():
    start_index = time.strftime("%Y%m%d%H%M%S",
                                time.localtime(int(time.time())))
    return int(start_index)

# 注意脚本启动要在cal_minute_list之前, ret_minute_list之后
if __name__ == '__main__':
    with open('./config/cost.yaml', 'r') as file:
        cfg = yaml.safe_load(file)
    timeout = 30
    clients = um_client_trade.UmClient(cfg["um_client"], UMFutures)

    target_position_map = {}
    previous_position_volume_map = {}
    current_position_volume_map = {}
    previous_delta_position_map = {}
    current_delta_position_map = {}
    sum_wc = 0.0
    while True:
        time.sleep(5)
        current_time = datetime.now()

        # 计算出上一个信号仓位的return (1.使用上一个仓位volume用vwap求差值 2.备份当前仓位为下次计算使用)
        if current_time.minute in ret_minute_list:
            current_str = current_time.strftime("%Y%m%d%H%M00")
            if current_str.endswith("000200"):
                sum_wc = 0.0
            if len(previous_position_volume_map) != 0:
                # get vwap before[] and current[]
                current_time = current_time - timedelta(minutes=7)
                signal_index = int(current_time.strftime("%Y%m%d%H%M00"))
                df = pd.DataFrame()
                ud.readuniverse(ud.g_data)
                lasteddownload = dmap.read_memmap(ud.g_data)
                dr = ud.g_data
                dr["uid"] = dr["sids"]
                tmlist = dr["min1info_tm"].tolist()
                sidlist = dr["sids"].tolist()
                df["sid"] = np.array(sidlist)
                min1i = tmlist.index(signal_index)
                df["vwap_before"] = np.nanmean(dr["min1info_vwap"][min1i-5:min1i], axis=0)
                df["vwap_current"] = np.nanmean(dr["min1info_vwap"][min1i:min1i+5], axis=0)
                df['delta'] = df['vwap_current'] - df['vwap_before']
                # cal the return based on vwap and positon
                total_sum = 0.0
                for sid in df['sid']:
                    if sid in previous_position_volume_map:
                        try:
                            position_volume = float(previous_position_volume_map[sid])
                            delta_value = float(df.loc[df['sid'] == sid, 'delta'].values[0])
                            if np.isnan(position_volume):
                                logger.error(f"position_volume is nan for sid {sid}. Skipping this entry.")
                                continue
                            if np.isnan(delta_value):
                                before_value = float(df.loc[df['sid'] == sid, 'vwap_before'].values[0])
                                current_value = float(df.loc[df['sid'] == sid, 'vwap_current'].values[0])
                                logger.error(f"delta_value is nan for sid {sid}, {before_value}, {current_value}. Skipping this entry.")
                                continue
                            total_sum += position_volume * delta_value
                        except Exception as e:
                            logger.error(f"Unexpected error for sid {sid}: {str(e)}. Skipping this entry.")
                total_delta_sum = 0.0
                for sid in df['sid']:
                    if sid in previous_delta_position_map:
                        try:
                            target_current_delta = float(previous_delta_position_map[sid])
                            if np.isnan(target_current_delta):
                                logger.error(f"target_current_delta is nan for sid {sid}. Skipping this entry.")
                                continue
                            delta_value = float(df.loc[df['sid'] == sid, 'delta'].values[0])
                            if np.isnan(delta_value):
                                logger.error(f"delta_value is nan for sid {sid}. Skipping this entry.")
                                continue
                            delta_volume = float(target_current_delta/float(df.loc[df['sid'] == sid, 'vwap_before'].values[0]))
                            if np.isnan(delta_volume):
                                logger.error(f"delta_volume is nan for sid {sid}. Skipping this entry.")
                                continue
                            total_delta_sum += delta_volume * delta_value
                        except Exception as e:
                            logger.error(f"Unexpected error for sid {sid}: {str(e)}")
                if not np.isnan(total_delta_sum):
                    sum_wc += total_delta_sum
                current_time = current_time - timedelta(minutes=5)
                signal_index_start = int(current_time.strftime("%Y%m%d%H%M00"))
                logger.info(f"本次信号: [{signal_index_start},{signal_index}]")
                logger.info(f"持仓Return$: {total_sum}")
                logger.info(f"本次误差$: {total_delta_sum}")
                logger.info(f"累计误差$: {sum_wc}")
            # reset previous position map
            previous_position_volume_map = copy.deepcopy(current_position_volume_map)
            previous_delta_position_map = copy.deepcopy(current_delta_position_map)
            time.sleep(60)

        if current_time.minute in cal_minute_list:
            # Reset map to 0.0
            for key in available_token_map:
                available_token_map[key] = 0.0
            for key in available_token_map_volume:
                available_token_map_volume[key] = 0.0
            for key in current_delta_position_map:
                current_delta_position_map[key] = 0.0
            total_delta = 0.0
            # Fetch current position value(money)
            update_positions(clients, available_token_map, available_token_map_volume)
            current_time = current_time - timedelta(minutes=5)
            signal_index = int(current_time.strftime("%Y%m%d%H%M00"))
            # reset current position volume map
            current_position_volume_map = {k: v for k, v in available_token_map_volume.items() if abs(v) > 0}
            # Record position only
            #if current_time.minute in fetch_minute_list:
            #    previous_position_map = copy.deepcopy(available_token_map)
            # Calculate the ratio
            file_path = f'{signal_path}/{signal_index}{signal_suffix}'
            if os.path.exists(file_path):
                with open(file_path, mode='r') as file:
                    csv_reader = csv.reader(file)
                    next(csv_reader)
                    for row in csv_reader:
                        key = row[0]
                        value = round(float(row[1]), 2)
                        target_position_map[key] = value
                    # Cal the delta
                    for key, value in available_token_map.items():
                        current_value = round(float(value), 2)
                        target_value = 0.00
                        if key in target_position_map:
                            target_value = round(target_position_map[key], 2)
                        ratio_value = round(abs(current_value-target_value), 2)
                        if ratio_value > 50:
                            logger.info(f"Symbol: {key}, Current: {current_value}, Target: {target_value}")
                        if ratio_value > 10.0:
                            current_delta_position_map[key] = target_value - current_value
                        total_delta = total_delta + round(ratio_value, 2)
                    # Output the result
                    long_positions = {k: v for k, v in available_token_map.items() if v > 0}
                    short_positions = {k: v for k, v in available_token_map.items() if v < 0}
                    long_value = sum(long_positions.values())
                    short_value = sum(short_positions.values())
                    total_value = abs(long_value) + abs(short_value)
                    positions_volume = {k: v for k, v in available_token_map_volume.items() if abs(v) > 0}
                    if total_value == 0:
                        logger.error("Error")
                        continue
                    logger.info(f"本次信号: {signal_index}")
                    logger.info(f"Long持仓金额$: {long_value}")
                    logger.info(f"Long持仓明细$: {long_positions}")
                    logger.info(f"Short持仓金额$: {short_value}")
                    logger.info(f"Short持仓明细$: {short_positions}")
                    logger.info(f"持仓数量: {positions_volume}")
                    logger.info(f"Delta合计: {total_delta}")
                    logger.info(f"贴合率: {round(1-total_delta/total_value, 2)}")

            time.sleep(60)
