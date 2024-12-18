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

minute_list = [6, 21, 36, 51]
output_directory = "/home/crypto/order/"
today = datetime.now().strftime("%Y%m%d")
csv_file_path = os.path.join(output_directory, f"{today}.csv")
signal_suffix = '_book.csv'

headers = ["buyer", "commission", "commissionAsset", "id", "maker", "orderId", "price",
           "qty", "quoteQty", "realizedPnl", "side", "positionSide", "symbol", "time"]

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

available_token_map = {
    'BTCUSDT': 0.0, 'ETHUSDT': 0.0, 'BCHUSDT': 0.0, 'XRPUSDT': 0.0, 'EOSUSDT': 0.0,
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
    'REEFUSDT': 0.0, 'RVNUSDT': 0.0, 'SFPUSDT': 0.0, 'XEMUSDT': 0.0, 'BTCSTUSDT': 0.0,
    'COTIUSDT': 0.0, 'CHRUSDT': 0.0, 'MANAUSDT': 0.0, 'ALICEUSDT': 0.0, 'HBARUSDT': 0.0,
    'ONEUSDT': 0.0, 'LINAUSDT': 0.0, 'STMXUSDT': 0.0, 'DENTUSDT': 0.0, 'CELRUSDT': 0.0,
    'HOTUSDT': 0.0, 'MTLUSDT': 0.0, 'OGNUSDT': 0.0, 'NKNUSDT': 0.0, 'SCUSDT': 0.0,
    'DGBUSDT': 0.0, '1000SHIBUSDT': 0.0, 'BAKEUSDT': 0.0, 'GTCUSDT': 0.0, 'BTCDOMUSDT': 0.0,
    'IOTXUSDT': 0.0, 'RAYUSDT': 0.0, 'C98USDT': 0.0, 'MASKUSDT': 0.0, 'ATAUSDT': 0.0,
    'DYDXUSDT': 0.0, '1000XECUSDT': 0.0, 'GALAUSDT': 0.0, 'CELOUSDT': 0.0, 'ARUSDT': 0.0,
    'KLAYUSDT': 0.0, 'ARPAUSDT': 0.0, 'CTSIUSDT': 0.0, 'LPTUSDT': 0.0, 'ENSUSDT': 0.0,
    'PEOPLEUSDT': 0.0, 'ROSEUSDT': 0.0, 'DUSKUSDT': 0.0, 'FLOWUSDT': 0.0, 'IMXUSDT': 0.0,
    'API3USDT': 0.0, 'GMTUSDT': 0.0, 'APEUSDT': 0.0, 'WOOUSDT': 0.0, 'FTTUSDT': 0.0,
    'JASMYUSDT': 0.0, 'DARUSDT': 0.0, 'GALUSDT': 0.0, 'OPUSDT': 0.0, 'INJUSDT': 0.0,
    'STGUSDT': 0.0, 'SPELLUSDT': 0.0, '1000LUNCUSDT': 0.0, 'LUNA2USDT': 0.0, 'LDOUSDT': 0.0,
    'CVXUSDT': 0.0, 'ICPUSDT': 0.0, 'APTUSDT': 0.0, 'QNTUSDT': 0.0, 'FETUSDT': 0.0,
    'FXSUSDT': 0.0, 'HOOKUSDT': 0.0, 'MAGICUSDT': 0.0, 'TUSDT': 0.0, 'RNDRUSDT': 0.0,
    'HIGHUSDT': 0.0, 'MINAUSDT': 0.0, 'ASTRUSDT': 0.0, 'AGIXUSDT': 0.0, 'PHBUSDT': 0.0,
    'GMXUSDT': 0.0, 'CFXUSDT': 0.0, 'STXUSDT': 0.0, 'BNXUSDT': 0.0, 'ACHUSDT': 0.0,
    'SSVUSDT': 0.0, 'CKBUSDT': 0.0, 'PERPUSDT': 0.0, 'TRUUSDT': 0.0, 'LQTYUSDT': 0.0,
    'USDCUSDT': 0.0, 'IDUSDT': 0.0, 'ARBUSDT': 0.0, 'JOEUSDT': 0.0, 'TLMUSDT': 0.0,
    'AMBUSDT': 0.0, 'LEVERUSDT': 0.0, 'RDNTUSDT': 0.0, 'HFTUSDT': 0.0, 'XVSUSDT': 0.0,
    'BLURUSDT': 0.0, 'EDUUSDT': 0.0, 'IDEXUSDT': 0.0, 'SUIUSDT': 0.0, '1000PEPEUSDT': 0.0,
    '1000FLOKIUSDT': 0.0, 'UMAUSDT': 0.0, 'RADUSDT': 0.0, 'KEYUSDT': 0.0, 'COMBOUSDT': 0.0,
    'NMRUSDT': 0.0, 'MAVUSDT': 0.0, 'MDTUSDT': 0.0, 'XVGUSDT': 0.0, 'WLDUSDT': 0.0,
    'PENDLEUSDT': 0.0, 'ARKMUSDT': 0.0, 'AGLDUSDT': 0.0, 'YGGUSDT': 0.0, 'DODOXUSDT': 0.0,
    'BNTUSDT': 0.0, 'OXTUSDT': 0.0, 'SEIUSDT': 0.0, 'CYBERUSDT': 0.0, 'HIFIUSDT': 0.0,
    'ARKUSDT': 0.0, 'FRONTUSDT': 0.0, 'GLMRUSDT': 0.0, 'BICOUSDT': 0.0, 'STRAXUSDT': 0.0,
    'LOOMUSDT': 0.0, 'BIGTIMEUSDT': 0.0, 'BONDUSDT': 0.0, 'ORBSUSDT': 0.0, 'STPTUSDT': 0.0,
    'WAXPUSDT': 0.0, 'BSVUSDT': 0.0, 'RIFUSDT': 0.0, 'POLYXUSDT': 0.0, 'GASUSDT': 0.0,
    'POWRUSDT': 0.0, 'SLPUSDT': 0.0, 'TIAUSDT': 0.0, 'SNTUSDT': 0.0, 'CAKEUSDT': 0.0,
    'MEMEUSDT': 0.0, 'TWTUSDT': 0.0, 'TOKENUSDT': 0.0, 'ORDIUSDT': 0.0, 'STEEMUSDT': 0.0,
    'BADGERUSDT': 0.0, 'ILVUSDT': 0.0, 'NTRNUSDT': 0.0, 'KASUSDT': 0.0, 'BEAMXUSDT': 0.0,
    '1000BONKUSDT': 0.0, 'PYTHUSDT': 0.0, 'SUPERUSDT': 0.0, 'USTCUSDT': 0.0, 'ONGUSDT': 0.0,
    'ETHWUSDT': 0.0, 'JTOUSDT': 0.0, '1000SATSUSDT': 0.0, 'AUCTIONUSDT': 0.0, '1000RATSUSDT': 0.0,
    'ACEUSDT': 0.0, 'MOVRUSDT': 0.0, 'NFPUSDT': 0.0, 'AIUSDT': 0.0, 'XAIUSDT': 0.0,
    'WIFUSDT': 0.0, 'MANTAUSDT': 0.0, 'ONDOUSDT': 0.0, 'LSKUSDT': 0.0, 'ALTUSDT': 0.0,
    'JUPUSDT': 0.0, 'ZETAUSDT': 0.0, 'RONINUSDT': 0.0, 'DYMUSDT': 0.0, 'OMUSDT': 0.0,
    'PIXELUSDT': 0.0, 'STRKUSDT': 0.0, 'MAVIAUSDT': 0.0, 'GLMUSDT': 0.0, 'PORTALUSDT': 0.0,
    'TONUSDT': 0.0, 'AXLUSDT': 0.0, 'MYROUSDT': 0.0, 'METISUSDT': 0.0, 'AEVOUSDT': 0.0,
    'VANRYUSDT': 0.0, 'BOMEUSDT': 0.0, 'ETHFIUSDT': 0.0, 'ENAUSDT': 0.0, 'WUSDT': 0.0,
    'TNSRUSDT': 0.0, 'SAGAUSDT': 0.0, 'TAOUSDT': 0.0, 'OMNIUSDT': 0.0, 'REZUSDT': 0.0,
    'BBUSDT': 0.0, 'NOTUSDT': 0.0, 'TURBOUSDT': 0.0, 'IOUSDT': 0.0
}

def get_available_tokens(signal_base_path: str, current_index: int) -> set:
    symbol_set = set()
    try:
        for i in range(1, 5):
            file_path = f'{signal_base_path}/{current_index}{signal_suffix}'
            with open(file_path, mode='r') as file:
                csv_reader = csv.reader(file)
                next(csv_reader)  # 跳过表头行
                for row in csv_reader:
                    key = row[0]
                    value = float(row[1])
                    if value != 0:
                        symbol_set.add(key)
                # 获取前一小时坐标
                time_str = str(current_index)
                dt = datetime.strptime(time_str, '%Y%m%d%H%M%S')
                previous_hours = dt - timedelta(minutes=15 * i)
                current_index = int(previous_hours.strftime('%Y%m%d%H%M%S'))
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
    except Exception as e:
        logger.error(f"get_available_tokens error: {str(e)}")
    return symbol_set

def account_balance(client: UMFutures) -> list:
    try:
        return client.balance()
    except Exception as e:
        logger.error(f"account_balance error: {str(e)}")
        return []

@retry(stop=stop_after_attempt(50), wait=wait_fixed(15), retry=retry_if_exception_type(Exception))
def get_account_value(client: UMFutures) -> float:
    try:
        account_info = client.balance()
        usdt_balance = bnb_balance = usdt_crossUnPnl = 0.0
        for entry in account_info:
            if entry['asset'] == 'USDT':
                usdt_balance = float(entry['balance'])
                usdt_crossUnPnl = float(entry['crossUnPnl'])
            elif entry['asset'] == 'BNB':
                bnb_balance = float(entry['balance'])

        depth = client.depth(symbol='BNBUSDT', limit=5)
        price = round(float(depth['bids'][0][0]), 2)

        account_value = round(bnb_balance * price + usdt_balance + usdt_crossUnPnl, 2)
        return account_value
    except Exception as e:
        logger.error(f"get_account_value error: {str(e)}")
        raise

def write_order_data_to_csv(file_path, order_data):
    file_exists = os.path.isfile(file_path)

    with open(file_path, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=headers)

        if not file_exists:
            writer.writeheader()

        for data in order_data:
            writer.writerow(data)

@retry(stop=stop_after_attempt(50), wait=wait_fixed(15), retry=retry_if_exception_type(Exception))
def get_order_data(client: UMFutures, symbol: str, start_time: int, end_time: int) -> list:
    try:
        data = client.get_account_trades(symbol=symbol, startTime=start_time, endTime=end_time)
        return data
    except Exception as e:
        logger.error(f"get_order_data error: {str(e)}")
        raise

if __name__ == '__main__':
    myproxies = {
        'http': 'http://127.0.0.1:14890',
        'https': 'http://127.0.0.1:14890'
    }
    timeout = 30
    client = UMFutures(key='OugVodq3VF9syMAWI9iOGBwaIb4G0Pv3xVXSUOto24Oc4vImXaZBcpOyqL4uwkxF',
                       secret='tNGt7sG9kMatPYqn1TcX8fcbds4jdgN21TY8TKD60rZRjuKxy2W6yZCLMuOCSyKw', proxies=myproxies,
                       timeout=timeout)
    signal_path = "/home/crypto/signal/predv215_3w"
    lookback_mins = 6

    while True:
        time.sleep(1)
        current_time = datetime.now()

        if current_time.minute in minute_list:
            start_time = current_time - timedelta(minutes=6)
            start_index = start_time.strftime("%Y%m%d%H%M00")

            if start_index.endswith("000000"):
                today = start_time.strftime("%Y%m%d")
                csv_file_path = os.path.join(output_directory, f"{today}.csv")

            symbol_set = get_available_tokens(signal_path, int(start_index))
            key_datetime = datetime.strptime(str(int(start_index)), "%Y%m%d%H%M%S")
            server_time_datetime = key_datetime + timedelta(minutes=lookback_mins)
            server_time = int(server_time_datetime.timestamp() * 1000)
            for token in symbol_set:
                order_data = get_order_data(client, token, server_time - 1000 * 60 * lookback_mins, server_time)
                if order_data:
                    write_order_data_to_csv(csv_file_path, order_data)

            time.sleep(60)