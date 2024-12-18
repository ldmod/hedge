import sys
import os
sys.path.append(os.path.abspath(__file__+"../../../../../"))
from datetime import datetime, timezone, timedelta
import time
import csv
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
from binance.um_futures import UMFutures
from dingtalkchatbot.chatbot import DingtalkChatbot
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
import logging
import warnings
warnings.simplefilter("ignore", category=RuntimeWarning)
import yaml
import cryptoqt.data.updatedata as ud
import cryptoqt.data.datammap as dmap
np.seterr(invalid='ignore')
import cryptoqt.data.tools as tools
import cryptoqt.trade.algo.um_client_trade as um_client_trade
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option("display.max_colwidth", 60)
pd.set_option('display.width', 10000)
# 初始化日志系统
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 定义需要处理的分钟列表
minute_list = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]


# 初始状态变量
total_bp_cost = 0
total_bp_money = 0
today = datetime.now().strftime("%Y%m%d")

# 信号文件后缀
signal_suffix = '_book.csv'

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


# 当前账户余额查询
def account_balance(client) -> list:
    try:
        client=clients.get_um_client()
        balance = client.client.balance()
        clients.recycle(client)
        return balance
    except Exception as e:
        logger.error(f"account_balance error: {str(e)}")
        return []


def get_available_tokens(signal_base_path: str, current_index: int) -> set:
    symbol_set = set(available_token_map.keys())
    return symbol_set

def int_to_ms(yyyymmddhhmmss: int) -> int:
    dt_str = str(yyyymmddhhmmss)
    dt = datetime.strptime(dt_str, '%Y%m%d%H%M%S').replace(tzinfo=timezone.utc)
    timestamp = int(dt.timestamp() * 1000)
    return timestamp


@retry(stop=stop_after_attempt(50), wait=wait_fixed(15), retry=retry_if_exception_type(Exception))
def get_server_time(client: UMFutures) -> dict:
    try:
        return client.time()
    except Exception as e:
        logger.error(f"get_server_time error: {str(e)}")
        raise


@retry(stop=stop_after_attempt(50), wait=wait_fixed(15), retry=retry_if_exception_type(Exception))
def get_order_data(client, symbol: str, start_time: int, end_time: int) -> list:
    try:
        client=clients.get_um_client()
        data = client.client.get_account_trades(symbol=symbol, startTime=start_time, endTime=end_time)
        clients.recycle(client)
        return data
    except Exception as e:
        logger.error(f"get_order_data error: {str(e)}")
        raise


@retry(stop=stop_after_attempt(50), wait=wait_fixed(15), retry=retry_if_exception_type(Exception))
def get_account_value(client) -> float:
    try:
        client=clients.get_um_client()
        account_info = client.client.balance()
        clients.recycle(client)
        
        usdt_balance = bnb_balance = usdt_crossUnPnl = 0.0
        for entry in account_info:
            if entry['asset'] == 'USDT':
                usdt_balance = float(entry['balance'])
                usdt_crossUnPnl = float(entry['crossUnPnl'])
            elif entry['asset'] == 'BNB':
                bnb_balance = float(entry['balance'])

        client=clients.get_um_client()
        depth = client.client.depth(symbol='BNBUSDT', limit=5)
        clients.recycle(client)
        price = round(float(depth['bids'][0][0]), 2)

        account_value = round(bnb_balance * price + usdt_balance + usdt_crossUnPnl, 2)
        return account_value
    except Exception as e:
        logger.error(f"get_account_value error: {str(e)}")
        raise

if __name__ == '__main__':
    myproxies = {
        'http': 'http://127.0.0.1:12890',
        'https': 'http://127.0.0.1:12890'
    }
    with open('./config/cost.yaml', 'r') as file:
        cfg = yaml.safe_load(file)
    timeout = 30
    clients = um_client_trade.UmClient(cfg["um_client"], UMFutures)
    # client = UMFutures(key='SULw1eShlgM3JK5fPGH2In5lQl4YujhQzZMEJnbaeUYzEGQW8OxJu2q5wUak6P65',
    #                    secret='GqlJkGbpJX5GSChIhbcTc1Z1bHFagFy95frQcD8dWlJHQdlkOwYtBVca4VNipR8d', proxies=myproxies,
    #                    timeout=timeout)

    current_time = datetime.now()
    start_of_day = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
    current_index = int(start_of_day.strftime("%Y%m%d%H%M%S"))
    current_index = 20240924210000

    while True:
        time.sleep(5)
        current_time = datetime.now()

        if current_time.minute in minute_list :
            start_time = current_time - timedelta(minutes=5)
            start_index = start_time.strftime("%Y%m%d%H%M00")
            total_account_value = get_account_value(clients)
            if start_index.endswith("000000"):
                notify("5Min Start: " + str(start_index) + ", Cost: " + str(total_bp_cost) + " ,Money: " + str(
                    total_bp_money) + " , BP: " + str(total_bp_cost / total_bp_money * 10000) + " ,Account: " + str(
                    total_account_value))
                total_bp_cost = 0
                total_bp_money = 0
                today = start_time.strftime("%Y%m%d")

            logger.info(f"账户净值$: {get_account_value(clients)}")
            update_positions(clients, available_token_map)
            long_value = sum(value for value in available_token_map.values() if value > 0)
            short_value = sum(value for value in available_token_map.values() if value < 0)
            logger.info(f"Long仓位$: {long_value}, Short仓位$: {short_value}")


            try:
                signal_path = "/home/crypto/signal/predv25_1w"
                lookback_mins = 5
                server_time = 0

                df = pd.DataFrame()
                ud.readuniverse(ud.g_data)
                lasteddownload = dmap.read_memmap(ud.g_data)
                dr = ud.g_data
                dr["uid"] = dr["sids"]

                tmlist = dr["min1info_tm"].tolist()
                sidlist = dr["sids"].tolist()
                df["sid"] = np.array(sidlist)

                while current_index <= int(start_index):
                    logger.info(f"-----------------本次信号: {current_index} ---------------------")
                    current_index_unix = tools.tmi2u(current_index) + 5
                    server_time = current_index_unix + lookback_mins * 1000 * 60 + 5
                    min1i = tmlist.index(current_index)
                    symbol_set = get_available_tokens(signal_path, current_index)
                    #df["vwap"] = dr["min1info_vwap"][min1i:min1i + 5].mean(axis=0)
                    
                    while True:
                        lasteddownload=dmap.gettmsize()
                        if lasteddownload>=min1i+lookback_mins:
                            break
                        time.sleep(5)
                        
                    df["vwap"] = np.nanmean(dr["min1info_vwap"][min1i:min1i + 5], axis=0)
                    df['price'] = df['1m'] = df['2m'] = df['3m'] = df['4m'] = df['5m'] = df['money'] = df[
                        'volume'] = df['cost'] = 0.0
                    df['last_time'] = ''

                    for token in symbol_set:
                        total_money = total_qty = money_1m = qty_1m = money_2m = qty_2m = money_3m = qty_3m = money_4m = qty_4m = money_5m = qty_5m = 0
                        fee_bnb = 0
                        buyer = None
                        max_order_time = 0
                        order_data = get_order_data(clients, token, server_time - 1000 * 60 * lookback_mins, server_time)
                        start_time_index = server_time - 1000 * 60 * lookback_mins
                        if order_data:
                            for order in order_data:
                                if buyer is None:
                                    buyer = order['buyer']
                                if buyer == order['buyer']:
                                    price = float(order['price'])
                                    qty = float(order['qty'])
                                    total_money += price * qty
                                    total_qty += qty
                                    order_timestamp = int(order['time'])
                                    if (order_timestamp > start_time_index) and (
                                            order_timestamp <= (start_time_index + 60 * 1000 * 1)):
                                        money_1m += price * qty
                                        qty_1m += qty
                                    if (order_timestamp > (start_time_index + 60 * 1000 * 1)) and (
                                            order_timestamp <= (start_time_index + 60 * 1000 * 2)):
                                        money_2m += price * qty
                                        qty_2m += qty
                                    if (order_timestamp > (start_time_index + 60 * 1000 * 2)) and (
                                            order_timestamp <= (start_time_index + 60 * 1000 * 3)):
                                        money_3m += price * qty
                                        qty_3m += qty
                                    if (order_timestamp > (start_time_index + 60 * 1000 * 3)) and (
                                            order_timestamp <= (start_time_index + 60 * 1000 * 4)):
                                        money_4m += price * qty
                                        qty_4m += qty
                                    if (order_timestamp > (start_time_index + 60 * 1000 * 4)) and (
                                            order_timestamp <= (start_time_index + 60 * 1000 * 5)):
                                        money_5m += price * qty
                                        qty_5m += qty
                                    if order_timestamp > max_order_time:
                                        max_order_time = order_timestamp
                            average_price_own = total_money / total_qty if total_qty != 0 else 0
                            price_own_1m = money_1m / qty_1m if qty_1m != 0 else 0
                            price_own_2m = money_2m / qty_2m if qty_2m != 0 else 0
                            price_own_3m = money_3m / qty_3m if qty_3m != 0 else 0
                            price_own_4m = money_4m / qty_4m if qty_4m != 0 else 0
                            price_own_5m = money_5m / qty_5m if qty_5m != 0 else 0
                            df.loc[df['sid'] == token, 'price'] = average_price_own
                            df.loc[df['sid'] == token, '1m'] = price_own_1m
                            df.loc[df['sid'] == token, '2m'] = price_own_2m
                            df.loc[df['sid'] == token, '3m'] = price_own_3m
                            df.loc[df['sid'] == token, '4m'] = price_own_4m
                            df.loc[df['sid'] == token, '5m'] = price_own_5m
                            df.loc[df['sid'] == token, 'money'] = total_money
                            df.loc[df['sid'] == token, 'volume'] = total_qty
                            df.loc[df['sid'] == token, 'buyer'] = buyer
                            order_time = datetime.fromtimestamp(max_order_time / 1000.0)
                            df.loc[df['sid'] == token, 'last_time'] = order_time.strftime('%H:%M:%S')

                    rf = df[df['money'] > 0]
                    if rf['money'].sum(axis=0) > 0:
                        df['delta'] = df['price'] - df['vwap']
                        df.loc[df['buyer'] == True, 'delta'] *= -1
                        df['cost'] = df['delta'] * df['volume']
                        rf = df[df['money'] > 0]
                        select_columns=["sid","vwap", "price","money", "volume" , "cost", "last_time", "buyer",  "delta",
                                        "1m","2m", "3m",  "4m",  "5m"]
                        rf=rf[select_columns]
                        rf[["money", "volume", "1m","2m", "3m",  "4m",  "5m"]]=rf[["money", "volume", "1m","2m", "3m",  "4m",  "5m"]].round(2)
                        if rf['money'].sum(axis=0) > 0:
                            logger.info(f"本次交易金额$: {rf['money'].sum(axis=0)}")
                            logger.info(f"本次冲击成本$: {rf['cost'].sum(axis=0)}")
                            logger.info(f"本次冲击成本bp: {rf['cost'].sum(axis=0) / rf['money'].sum(axis=0) * 10000}")
                            total_bp_cost += rf['cost'].sum(axis=0)
                            total_bp_money += rf['money'].sum(axis=0)
                            logger.info(f"今日累计冲击成本$: {total_bp_cost}")
                            logger.info(f"今日累计交易金额$: {total_bp_money}")
                            logger.info(f"今日累计冲击成本bp: {total_bp_cost / total_bp_money * 10000}")
                            logger.info(f"\ndetail: \n{rf}")
                            
                    else:
                        logger.info(f"rf: {rf['money'].sum(axis=0) }")
                    # 信号步进
                    current_index_unix += 5 * 60 * 1000
                    current_index = tools.tmu2i(current_index_unix)
                            

                time.sleep(60)  # 确保每次循环间隔一分钟
            except Exception as e:
                logger.error(f"exception: {str(e)}")
                break

