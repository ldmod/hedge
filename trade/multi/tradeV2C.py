import yaml
import copy
import random
import os
import csv
import datetime
import time
import logging
import multiprocessing
from binance.um_futures import UMFutures
from datetime import datetime, timedelta
from dingtalkchatbot.chatbot import DingtalkChatbot

# Post Maker Order Fail
SUB_STR = "Post Only order will be rejected"
# Signal file index delta
time_delta = timedelta(minutes=15)
# Suffix for signal file
signal_suffix = '_book.csv'


# DingDing Notify
def notify(msg):
    webhook = "https://oapi.dingtalk.com/robot/send?access_token=08981a30db7ee421f6f910cb1dbe9b722bb18420b1c54d9b5cdad300470d2cda"
    xiaoding = DingtalkChatbot(webhook)
    xiaoding.send_text(msg, is_at_all=False)


# Logger Init
def get_logger(logger_name, log_file, level=logging.INFO):
    logger = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s', "%Y-%m-%d %H:%M:%S")
    fileHandler = logging.FileHandler(log_file, mode='a')
    fileHandler.setFormatter(formatter)

    logger.setLevel(level)
    logger.addHandler(fileHandler)
    return logger


def query_bids(client, symbol, precision, logger_error):
    try:
        depth = client.depth(symbol=symbol, limit=10)
        if depth['bids'] is None:
            return -1, -1, -1, -1, -1, -1, -1
        if depth['bids'][0] is None:
            return -1, -1, -1, -1, -1, -1, -1
        return round(float(depth['bids'][0][0]), precision), round(float(depth['bids'][1][0]), precision), round(
            float(depth['bids'][2][0]), precision), round(float(depth['bids'][3][0]), precision), round(
            float(depth['bids'][4][0]), precision), round(float(depth['bids'][5][0]), precision), round(
            float(depth['bids'][6][0]), precision)
    except Exception as e:
        logger_error.info("query_bids " + str(e))
        return -1, -1, -1, -1, -1, -1, -1


def query_asks(client, symbol, precision, logger_error):
    try:
        depth = client.depth(symbol=symbol, limit=10)
        if depth['asks'] is None:
            return -1, -1, -1, -1, -1, -1, -1
        if depth['asks'][0] is None:
            return -1, -1, -1, -1, -1, -1, -1
        return round(float(depth['asks'][0][0]), precision), round(float(depth['asks'][1][0]), precision), round(
            float(depth['asks'][2][0]), precision), round(float(depth['asks'][3][0]), precision), round(
            float(depth['asks'][4][0]), precision), round(float(depth['asks'][5][0]), precision), round(
            float(depth['asks'][6][0]), precision)
    except Exception as e:
        logger_error.info("query_asks: " + str(e))
        return -1, -1, -1, -1, -1, -1, -1


# Create sell order(maker)
def sell(client, amount, symbol, price, logger_trade, logger_error):
    try:
        params = {
            'symbol': symbol,
            'side': 'SELL',
            'type': 'LIMIT',
            'quantity': amount,
            'price': price,
            'timeInForce': 'GTX'
        }
        logger_trade.info("Sell: " + str(symbol) + ", amount: " + str(amount) + ", price: " + str(price))
        client.new_order(**params)
        return False
    except Exception as e:
        if SUB_STR not in str(e):
            logger_error.info(
                "Sell error: " + str(symbol) + ", amount: " + str(amount) + ", price: " + str(price) + str(e))
        return True


# Create buy order(maker)
def buy(client, amount, symbol, price, logger_trade, logger_error):
    try:
        params = {
            'symbol': symbol,
            'side': 'BUY',
            'type': 'LIMIT',
            'quantity': amount,
            'price': price,
            'timeInForce': 'GTX'
        }
        logger_trade.info("Buy: " + str(symbol) + ", amount: " + str(amount) + ", price: " + str(price))
        client.new_order(**params)
        return False
    except Exception as e:
        if SUB_STR not in str(e):
            logger_error.info(
                "Buy error: " + str(symbol) + ", amount: " + str(amount) + ", price: " + str(price) + str(e))
        return True


# Update position side(signal side)
def modify_position_side(client, logger_error):
    try:
        client.change_position_mode(dualSidePosition="false")
        return False
    except Exception as e:
        logger_error.info("modify_position_side: " + str(e))
        return True


# Cancel all open orders for symbol
def cancel_open_orders(client, symbol, logger_trade, logger_error):
    try:
        logger_trade.info("Cancel order: " + str(symbol))
        client.cancel_open_orders(symbol=symbol, recvWindow=2000)
    except Exception as e:
        logger_error.info("modify_position_side: " + str(e))
        return True


# Update position info
def update_positions(client, position_value_map, position_amount_map, logger_error):
    try:
        data_dict = client.account()

        if not data_dict:
            logger_error.info("data is null")
            return True
        if "positions" not in data_dict:
            logger_error.info("positions not exist")
            return True

        # update: position_value_map & position_amount_map
        for position in data_dict["positions"]:
            symbol = position["symbol"]
            notional = float(position["notional"])
            positionAmt = float(position["positionAmt"])
            if symbol in position_value_map:
                position_value_map[symbol] = notional
            if symbol in position_amount_map:
                position_amount_map[symbol] = positionAmt

        return False
    except Exception as e:
        logger_error.info("update_positions: " + str(e))
        return True


# Exchange Info
def get_exchange_info(client, logger_error):
    try:
        data = client.exchange_info()
        symbol_map = {}
        for symbol_info in data.get("symbols", []):
            symbol = symbol_info.get("symbol")
            if symbol and symbol.endswith("USDT"):
                price_precision = symbol_info.get("quantityPrecision")
                symbol_map[symbol] = price_precision
        print(symbol_map)
    except Exception as e:
        logger_error.info("get_exchange_info: " + str(e))


# Change leverage(1)
def change_leverage(client, symbol, logger_error):
    try:
        client.change_leverage(symbol=symbol, leverage=6)
    except Exception as e:
        logger_error.info("change_leverage: " + str(symbol) + str(e))


# Query account balance
def account_balance(client, logger_error):
    try:
        return client.balance()
    except Exception as e:
        logger_error.info("account_balance: " + str(e))


# Get order threshold(+10% Buffer)
def get_threshold(symbol):
    values = {
        "BTCUSDT": 110.0,
        "ETHUSDT": 22.0,
        "BCHUSDT": 22.0,
        "LTCUSDT": 22.0,
        "ETCUSDT": 22.0,
        "LINKUSDT": 22.0
    }
    return values.get(symbol, 5.5)


# Multi threads
def process_task(position_value_map, position_amount_map, api_key, api_secret, i, cfg):
    # Config setting
    base_path = cfg['signal']['base_path']
    pred_dir = cfg['signal']['pred_dir']

    # Init client
    client = UMFutures(key=api_key, secret=api_secret)
    # Init logger
    logfile_trade = f"/home/ubuntu/logs/alpha-trade{i}.log"
    logfile_error = f"/home/ubuntu/logs/alpha-error{i}.log"
    logger_trade = get_logger(f"logger_trade{i}", logfile_trade)
    logger_error = get_logger(f"logger_error{i}", logfile_error)

    # Start index of signal file(based on current timestamp)
    signal_index = time.strftime("%Y%m%d%H%M00", time.localtime(int(time.time() / 900) * 900 + 900 + 3600 * 8))

    # Precision map
    price_precision_map = {'BTCUSDT': 1, 'ETHUSDT': 2, 'BCHUSDT': 2, 'XRPUSDT': 4, 'EOSUSDT': 3, 'LTCUSDT': 2,
                           'TRXUSDT': 5, 'ETCUSDT': 3, 'LINKUSDT': 3, 'XLMUSDT': 5, 'ADAUSDT': 4, 'XMRUSDT': 2,
                           'DASHUSDT': 2, 'ZECUSDT': 2, 'XTZUSDT': 3, 'BNBUSDT': 2, 'ATOMUSDT': 3, 'ONTUSDT': 4,
                           'IOTAUSDT': 4, 'BATUSDT': 4, 'VETUSDT': 6, 'NEOUSDT': 3, 'QTUMUSDT': 3, 'IOSTUSDT': 6,
                           'THETAUSDT': 4, 'ALGOUSDT': 4, 'ZILUSDT': 5, 'KNCUSDT': 4, 'ZRXUSDT': 4, 'COMPUSDT': 2,
                           'OMGUSDT': 4, 'DOGEUSDT': 5, 'SXPUSDT': 4, 'KAVAUSDT': 4, 'BANDUSDT': 4, 'RLCUSDT': 4,
                           'WAVESUSDT': 4, 'MKRUSDT': 1, 'SNXUSDT': 3, 'DOTUSDT': 3, 'DEFIUSDT': 1, 'YFIUSDT': 0,
                           'BALUSDT': 3, 'CRVUSDT': 3, 'TRBUSDT': 3, 'RUNEUSDT': 3, 'SUSHIUSDT': 4, 'EGLDUSDT': 3,
                           'SOLUSDT': 3, 'ICXUSDT': 4, 'STORJUSDT': 4, 'BLZUSDT': 5, 'UNIUSDT': 3, 'AVAXUSDT': 3,
                           'FTMUSDT': 4, 'ENJUSDT': 5, 'FLMUSDT': 4, 'RENUSDT': 5, 'KSMUSDT': 3, 'NEARUSDT': 3,
                           'AAVEUSDT': 2, 'FILUSDT': 3, 'RSRUSDT': 6, 'LRCUSDT': 5, 'MATICUSDT': 4, 'OCEANUSDT': 4,
                           'CVCUSDT': 5, 'BELUSDT': 4, 'CTKUSDT': 4, 'AXSUSDT': 3, 'ALPHAUSDT': 5, 'ZENUSDT': 3,
                           'SKLUSDT': 5, 'GRTUSDT': 5, '1INCHUSDT': 4, 'CHZUSDT': 5, 'SANDUSDT': 4, 'ANKRUSDT': 5,
                           'LITUSDT': 3, 'UNFIUSDT': 3, 'REEFUSDT': 6, 'RVNUSDT': 5, 'SFPUSDT': 4, 'XEMUSDT': 4,
                           'BTCSTUSDT': 3, 'COTIUSDT': 5, 'CHRUSDT': 4, 'MANAUSDT': 4, 'ALICEUSDT': 3, 'HBARUSDT': 5,
                           'ONEUSDT': 5, 'LINAUSDT': 5, 'STMXUSDT': 5, 'DENTUSDT': 6, 'CELRUSDT': 5, 'HOTUSDT': 6,
                           'MTLUSDT': 4, 'OGNUSDT': 4, 'NKNUSDT': 5, 'SCUSDT': 6, 'DGBUSDT': 5, '1000SHIBUSDT': 6,
                           'BAKEUSDT': 4, 'GTCUSDT': 3, 'BTCDOMUSDT': 1, 'IOTXUSDT': 5, 'RAYUSDT': 3, 'C98USDT': 4,
                           'MASKUSDT': 4, 'ATAUSDT': 4, 'DYDXUSDT': 3, '1000XECUSDT': 5, 'GALAUSDT': 5, 'CELOUSDT': 3,
                           'ARUSDT': 3, 'KLAYUSDT': 4, 'ARPAUSDT': 5, 'CTSIUSDT': 4, 'LPTUSDT': 3, 'ENSUSDT': 3,
                           'PEOPLEUSDT': 5, 'ROSEUSDT': 5, 'DUSKUSDT': 5, 'FLOWUSDT': 3, 'IMXUSDT': 4, 'API3USDT': 4,
                           'GMTUSDT': 5, 'APEUSDT': 4, 'WOOUSDT': 5, 'FTTUSDT': 4, 'JASMYUSDT': 6, 'DARUSDT': 4,
                           'OPUSDT': 4, 'INJUSDT': 3, 'STGUSDT': 4, 'SPELLUSDT': 7, '1000LUNCUSDT': 5, 'LUNA2USDT': 4,
                           'LDOUSDT': 4, 'CVXUSDT': 3, 'ICPUSDT': 3, 'APTUSDT': 3, 'QNTUSDT': 2, 'FETUSDT': 4,
                           'FXSUSDT': 4, 'HOOKUSDT': 4, 'MAGICUSDT': 4, 'TUSDT': 5, 'HIGHUSDT': 4, 'MINAUSDT': 4,
                           'ASTRUSDT': 5, 'AGIXUSDT': 4, 'PHBUSDT': 4, 'GMXUSDT': 3, 'CFXUSDT': 5, 'STXUSDT': 4,
                           'BNXUSDT': 4, 'ACHUSDT': 6, 'SSVUSDT': 3, 'CKBUSDT': 6, 'PERPUSDT': 4, 'TRUUSDT': 5,
                           'LQTYUSDT': 4, 'USDCUSDT': 6, 'IDUSDT': 5, 'ARBUSDT': 4, 'JOEUSDT': 4, 'TLMUSDT': 6,
                           'AMBUSDT': 6, 'LEVERUSDT': 7, 'RDNTUSDT': 5, 'HFTUSDT': 5, 'XVSUSDT': 3, 'BLURUSDT': 4,
                           'EDUUSDT': 4, 'IDEXUSDT': 5, 'SUIUSDT': 4, '1000PEPEUSDT': 7, '1000FLOKIUSDT': 5,
                           'UMAUSDT': 3, 'RADUSDT': 4, 'KEYUSDT': 6, 'COMBOUSDT': 4, 'NMRUSDT': 3, 'MAVUSDT': 5,
                           'MDTUSDT': 5, 'XVGUSDT': 6, 'WLDUSDT': 4, 'PENDLEUSDT': 4, 'ARKMUSDT': 4, 'AGLDUSDT': 4,
                           'YGGUSDT': 4, 'DODOXUSDT': 6, 'BNTUSDT': 5, 'OXTUSDT': 5, 'SEIUSDT': 4, 'CYBERUSDT': 3,
                           'HIFIUSDT': 4, 'ARKUSDT': 4, 'FRONTUSDT': 4, 'GLMRUSDT': 5, 'BICOUSDT': 4, 'STRAXUSDT': 4,
                           'LOOMUSDT': 5, 'BIGTIMEUSDT': 4, 'BONDUSDT': 3, 'ORBSUSDT': 5, 'STPTUSDT': 5, 'WAXPUSDT': 5,
                           'BSVUSDT': 2, 'RIFUSDT': 5, 'POLYXUSDT': 5, 'GASUSDT': 3, 'POWRUSDT': 4, 'SLPUSDT': 6,
                           'TIAUSDT': 4, 'SNTUSDT': 5, 'CAKEUSDT': 4, 'MEMEUSDT': 6, 'TWTUSDT': 4, 'TOKENUSDT': 5,
                           'ORDIUSDT': 3, 'STEEMUSDT': 5, 'BADGERUSDT': 4, 'ILVUSDT': 2, 'NTRNUSDT': 4, 'KASUSDT': 5,
                           'BEAMXUSDT': 6, '1000BONKUSDT': 6, 'PYTHUSDT': 4, 'SUPERUSDT': 4, 'USTCUSDT': 5,
                           'ONGUSDT': 5, 'ETHWUSDT': 4, 'JTOUSDT': 4, '1000SATSUSDT': 7, 'AUCTIONUSDT': 3,
                           '1000RATSUSDT': 5, 'ACEUSDT': 4, 'MOVRUSDT': 3, 'NFPUSDT': 4, 'AIUSDT': 5, 'XAIUSDT': 4,
                           'WIFUSDT': 4, 'MANTAUSDT': 4, 'ONDOUSDT': 4, 'LSKUSDT': 4, 'ALTUSDT': 5, 'JUPUSDT': 4,
                           'ZETAUSDT': 4, 'RONINUSDT': 4, 'DYMUSDT': 4, 'OMUSDT': 5, 'PIXELUSDT': 4, 'STRKUSDT': 4,
                           'MAVIAUSDT': 4, 'GLMUSDT': 4, 'PORTALUSDT': 4, 'TONUSDT': 4, 'AXLUSDT': 4, 'MYROUSDT': 5,
                           'METISUSDT': 2, 'AEVOUSDT': 4, 'VANRYUSDT': 5, 'BOMEUSDT': 6, 'ETHFIUSDT': 3, 'ENAUSDT': 4,
                           'WUSDT': 4, 'TNSRUSDT': 4, 'SAGAUSDT': 4, 'TAOUSDT': 2, 'OMNIUSDT': 3, 'REZUSDT': 5,
                           'BBUSDT': 4, 'NOTUSDT': 6, 'TURBOUSDT': 6, 'IOUSDT': 3, 'ZKUSDT': 5, 'MEWUSDT': 6,
                           'LISTAUSDT': 4, 'ZROUSDT': 3, 'RENDERUSDT': 3}
    quantity_precision_map = {'BTCUSDT': 3, 'ETHUSDT': 3, 'BCHUSDT': 3, 'XRPUSDT': 1, 'EOSUSDT': 1, 'LTCUSDT': 3,
                              'TRXUSDT': 0, 'ETCUSDT': 2, 'LINKUSDT': 2, 'XLMUSDT': 0, 'ADAUSDT': 0, 'XMRUSDT': 3,
                              'DASHUSDT': 3, 'ZECUSDT': 3, 'XTZUSDT': 1, 'BNBUSDT': 2, 'ATOMUSDT': 2, 'ONTUSDT': 1,
                              'IOTAUSDT': 1, 'BATUSDT': 1, 'VETUSDT': 0, 'NEOUSDT': 2, 'QTUMUSDT': 1, 'IOSTUSDT': 0,
                              'THETAUSDT': 1, 'ALGOUSDT': 1, 'ZILUSDT': 0, 'KNCUSDT': 0, 'ZRXUSDT': 1, 'COMPUSDT': 3,
                              'OMGUSDT': 1, 'DOGEUSDT': 0, 'SXPUSDT': 1, 'KAVAUSDT': 1, 'BANDUSDT': 1, 'RLCUSDT': 1,
                              'WAVESUSDT': 1, 'MKRUSDT': 3, 'SNXUSDT': 1, 'DOTUSDT': 1, 'DEFIUSDT': 3, 'YFIUSDT': 3,
                              'BALUSDT': 1, 'CRVUSDT': 1, 'TRBUSDT': 1, 'RUNEUSDT': 0, 'SUSHIUSDT': 0, 'EGLDUSDT': 1,
                              'SOLUSDT': 0, 'ICXUSDT': 0, 'STORJUSDT': 0, 'BLZUSDT': 0, 'UNIUSDT': 0, 'AVAXUSDT': 0,
                              'FTMUSDT': 0, 'ENJUSDT': 0, 'FLMUSDT': 0, 'RENUSDT': 0, 'KSMUSDT': 1, 'NEARUSDT': 0,
                              'AAVEUSDT': 1, 'FILUSDT': 1, 'RSRUSDT': 0, 'LRCUSDT': 0, 'MATICUSDT': 0, 'OCEANUSDT': 0,
                              'CVCUSDT': 0, 'BELUSDT': 0, 'CTKUSDT': 0, 'AXSUSDT': 0, 'ALPHAUSDT': 0, 'ZENUSDT': 1,
                              'SKLUSDT': 0, 'GRTUSDT': 0, '1INCHUSDT': 0, 'CHZUSDT': 0, 'SANDUSDT': 0, 'ANKRUSDT': 0,
                              'LITUSDT': 1, 'UNFIUSDT': 1, 'REEFUSDT': 0, 'RVNUSDT': 0, 'SFPUSDT': 0, 'XEMUSDT': 0,
                              'BTCSTUSDT': 1, 'COTIUSDT': 0, 'CHRUSDT': 0, 'MANAUSDT': 0, 'ALICEUSDT': 1, 'HBARUSDT': 0,
                              'ONEUSDT': 0, 'LINAUSDT': 0, 'STMXUSDT': 0, 'DENTUSDT': 0, 'CELRUSDT': 0, 'HOTUSDT': 0,
                              'MTLUSDT': 0, 'OGNUSDT': 0, 'NKNUSDT': 0, 'SCUSDT': 0, 'DGBUSDT': 0, '1000SHIBUSDT': 0,
                              'BAKEUSDT': 0, 'GTCUSDT': 1, 'BTCDOMUSDT': 3, 'IOTXUSDT': 0, 'RAYUSDT': 1, 'C98USDT': 0,
                              'MASKUSDT': 0, 'ATAUSDT': 0, 'DYDXUSDT': 1, '1000XECUSDT': 0, 'GALAUSDT': 0,
                              'CELOUSDT': 1, 'ARUSDT': 1, 'KLAYUSDT': 1, 'ARPAUSDT': 0, 'CTSIUSDT': 0, 'LPTUSDT': 1,
                              'ENSUSDT': 1, 'PEOPLEUSDT': 0, 'ROSEUSDT': 0, 'DUSKUSDT': 0, 'FLOWUSDT': 1, 'IMXUSDT': 0,
                              'API3USDT': 1, 'GMTUSDT': 0, 'APEUSDT': 0, 'WOOUSDT': 0, 'FTTUSDT': 1, 'JASMYUSDT': 0,
                              'DARUSDT': 1, 'OPUSDT': 1, 'INJUSDT': 1, 'STGUSDT': 0, 'SPELLUSDT': 0, '1000LUNCUSDT': 0,
                              'LUNA2USDT': 0, 'LDOUSDT': 0, 'CVXUSDT': 0, 'ICPUSDT': 0, 'APTUSDT': 1, 'QNTUSDT': 1,
                              'FETUSDT': 0, 'FXSUSDT': 1, 'HOOKUSDT': 1, 'MAGICUSDT': 1, 'TUSDT': 0, 'HIGHUSDT': 1,
                              'MINAUSDT': 0, 'ASTRUSDT': 0, 'AGIXUSDT': 0, 'PHBUSDT': 0, 'GMXUSDT': 2, 'CFXUSDT': 0,
                              'STXUSDT': 0, 'BNXUSDT': 1, 'ACHUSDT': 0, 'SSVUSDT': 2, 'CKBUSDT': 0, 'PERPUSDT': 1,
                              'TRUUSDT': 0, 'LQTYUSDT': 1, 'USDCUSDT': 0, 'IDUSDT': 0, 'ARBUSDT': 1, 'JOEUSDT': 0,
                              'TLMUSDT': 0, 'AMBUSDT': 0, 'LEVERUSDT': 0, 'RDNTUSDT': 0, 'HFTUSDT': 0, 'XVSUSDT': 1,
                              'BLURUSDT': 0, 'EDUUSDT': 0, 'IDEXUSDT': 0, 'SUIUSDT': 1, '1000PEPEUSDT': 0,
                              '1000FLOKIUSDT': 0, 'UMAUSDT': 0, 'RADUSDT': 0, 'KEYUSDT': 0, 'COMBOUSDT': 1,
                              'NMRUSDT': 1, 'MAVUSDT': 0, 'MDTUSDT': 0, 'XVGUSDT': 0, 'WLDUSDT': 0, 'PENDLEUSDT': 0,
                              'ARKMUSDT': 0, 'AGLDUSDT': 0, 'YGGUSDT': 0, 'DODOXUSDT': 0, 'BNTUSDT': 0, 'OXTUSDT': 0,
                              'SEIUSDT': 0, 'CYBERUSDT': 1, 'HIFIUSDT': 0, 'ARKUSDT': 0, 'FRONTUSDT': 0, 'GLMRUSDT': 0,
                              'BICOUSDT': 0, 'STRAXUSDT': 0, 'LOOMUSDT': 0, 'BIGTIMEUSDT': 0, 'BONDUSDT': 1,
                              'ORBSUSDT': 0, 'STPTUSDT': 0, 'WAXPUSDT': 0, 'BSVUSDT': 1, 'RIFUSDT': 0, 'POLYXUSDT': 0,
                              'GASUSDT': 1, 'POWRUSDT': 0, 'SLPUSDT': 0, 'TIAUSDT': 0, 'SNTUSDT': 0, 'CAKEUSDT': 0,
                              'MEMEUSDT': 0, 'TWTUSDT': 0, 'TOKENUSDT': 0, 'ORDIUSDT': 1, 'STEEMUSDT': 0,
                              'BADGERUSDT': 0, 'ILVUSDT': 1, 'NTRNUSDT': 0, 'KASUSDT': 0, 'BEAMXUSDT': 0,
                              '1000BONKUSDT': 0, 'PYTHUSDT': 0, 'SUPERUSDT': 0, 'USTCUSDT': 0, 'ONGUSDT': 0,
                              'ETHWUSDT': 0, 'JTOUSDT': 0, '1000SATSUSDT': 0, 'AUCTIONUSDT': 2, '1000RATSUSDT': 0,
                              'ACEUSDT': 2, 'MOVRUSDT': 2, 'NFPUSDT': 1, 'AIUSDT': 0, 'XAIUSDT': 0, 'WIFUSDT': 1,
                              'MANTAUSDT': 1, 'ONDOUSDT': 1, 'LSKUSDT': 0, 'ALTUSDT': 0, 'JUPUSDT': 0, 'ZETAUSDT': 0,
                              'RONINUSDT': 1, 'DYMUSDT': 1, 'OMUSDT': 1, 'PIXELUSDT': 0, 'STRKUSDT': 1, 'MAVIAUSDT': 1,
                              'GLMUSDT': 0, 'PORTALUSDT': 1, 'TONUSDT': 1, 'AXLUSDT': 1, 'MYROUSDT': 0, 'METISUSDT': 2,
                              'AEVOUSDT': 1, 'VANRYUSDT': 0, 'BOMEUSDT': 0, 'ETHFIUSDT': 1, 'ENAUSDT': 0, 'WUSDT': 1,
                              'TNSRUSDT': 1, 'SAGAUSDT': 1, 'TAOUSDT': 3, 'OMNIUSDT': 2, 'REZUSDT': 0, 'BBUSDT': 0,
                              'NOTUSDT': 0, 'TURBOUSDT': 0, 'IOUSDT': 1, 'ZKUSDT': 0, 'MEWUSDT': 0, 'LISTAUSDT': 0,
                              'ZROUSDT': 1, 'RENDERUSDT': 1}

    # Just one time
    modify_position_side(client, logger_error)
    for key, value in position_value_map.items():
        change_leverage(client, key, logger_error)

    retry_count = 50
    signal_wait_max = 600  # 60s*20min/2s

    while (True):
        try:
            symbol_set = set()
            real_delta_dict = {}

            # Step1: Wait signal file and handle (before trade, once)
            file_path = f'{base_path}{pred_dir}/{signal_index}{signal_suffix}'
            logger_trade.info("------ Wait signal file: " + str(file_path) + " ------")
            signal_wait_count = 0
            while not os.path.exists(file_path):
                if (signal_wait_count > signal_wait_max):
                    signal_time = datetime.strptime(signal_index, "%Y%m%d%H%M%S")
                    signal_time = signal_time + time_delta
                    signal_index = signal_time.strftime("%Y%m%d%H%M00")
                    file_path = f'{base_path}{pred_dir}/{signal_index}{signal_suffix}'
                    notify("miss signal file, skip!")
                    logger_trade.info("------ Wait signal file: " + str(file_path) + " ------")
                    signal_wait_count = 0
                    continue
                time.sleep(2)
                signal_wait_count = signal_wait_count + 1

            alpha_map = {}
            try:
                with open(file_path, mode='r') as file:
                    csv_reader = csv.reader(file)
                    next(csv_reader)
                    for row in csv_reader:
                        key = row[0]
                        value = round(float(row[1]) / 1.5, 2)  # TODO checkpoint: 20000$
                        alpha_map[key] = value
            except Exception as e:
                logger_error.info(str(e))
                notify("bad signal file, skip!")
                continue

            for i in range(retry_count):
                logger_trade.info("------ Round " + str(i + 1) + " ------")
                # Step2: Update position data(True: fail)
                if (update_positions(client, position_value_map, position_amount_map, logger_error)):
                    continue
                filtered_values = {k: v for k, v in position_value_map.items() if v != 0.0}
                filtered_amounts = {k: v for k, v in position_amount_map.items() if v != 0.0}
                logger_trade.info("Positions value: " + str(filtered_values))
                logger_trade.info("Positions amount: " + str(filtered_amounts))

                # Step3: Random the symbols for fairness
                random_position_value_items = list(position_value_map.items())
                random.shuffle(random_position_value_items)

                for key, value in random_position_value_items:
                    if key in alpha_map:
                        alpha_value = alpha_map[key]
                        if i == 0:
                            real_delta_dict[key] = alpha_value
                        position_value = position_value_map[key]
                        real_delta = alpha_value - position_value
                        delta = real_delta * ((i // 10 + 1) / 5)
                        if abs(delta) < 60.0:
                            delta = real_delta
                        if delta > 250.0:
                            delta = 250.0
                        if delta < -250.0:
                            delta = -250.0

                        positive_threshold = get_threshold(key)
                        negative_threshold = positive_threshold * -1

                        first_divide_delta = delta * 0.0
                        first_divide_delta = first_divide_delta if abs(first_divide_delta) > positive_threshold else 0.0

                        second_divide_delta = delta * 0.2
                        second_divide_delta = second_divide_delta if abs(
                            second_divide_delta) > positive_threshold else 0.0

                        third_divide_delta = delta * 0.3
                        third_divide_delta = third_divide_delta if abs(third_divide_delta) > positive_threshold else 0.0

                        forth_divide_delta = delta * 0.3
                        forth_divide_delta = forth_divide_delta if abs(forth_divide_delta) > positive_threshold else 0.0

                        fifth_divide_delta = delta * 0.2
                        fifth_divide_delta = fifth_divide_delta if abs(fifth_divide_delta) > positive_threshold else 0.0

                        sixth_divide_delta = delta * 0.0
                        sixth_divide_delta = sixth_divide_delta if abs(sixth_divide_delta) > positive_threshold else 0.0

                        seventh_divide_delta = delta * 0.0
                        seventh_divide_delta = seventh_divide_delta if abs(
                            seventh_divide_delta) > positive_threshold else 0.0

                        if abs(delta) > positive_threshold:
                            logger_trade.info(str(key) + ", real_delta: " + str(real_delta) + ", delta: " + str(delta))
                            logger_trade.info("first: " + str(first_divide_delta) + ", second: " + str(
                                second_divide_delta) + ", third: " + str(third_divide_delta) + ", forth: " + str(
                                forth_divide_delta) + ", fifth: " + str(fifth_divide_delta) + ", six: " + str(
                                sixth_divide_delta) + ", seventh: " + str(seventh_divide_delta))

                        # For long symbols
                        if delta > positive_threshold:
                            symbol_set.add(key)
                            bid0, bid1, bid2, bid3, bid4, bid5, bid6 = query_bids(client, key, price_precision_map[key],
                                                                                  logger_error)
                            if (bid0 > 0) and (bid1 > 0) and (bid2 > 0) and (bid3 > 0) and (bid4 > 0) and (
                                    bid5 > 0) and (bid6 > 0):
                                bid1 = max(bid1, round(bid0 * 0.9999, price_precision_map[key]))
                                bid2 = max(bid2, round(bid0 * 0.9998, price_precision_map[key]))
                                bid3 = max(bid3, round(bid0 * 0.9997, price_precision_map[key]))
                                bid4 = max(bid4, round(bid0 * 0.9996, price_precision_map[key]))
                                bid5 = max(bid5, round(bid0 * 0.9995, price_precision_map[key]))
                                bid6 = max(bid6, round(bid0 * 0.9994, price_precision_map[key]))
                                logger_trade.info(
                                    "bid0: " + str(bid0) + " bid1: " + str(bid1) + " bid2: " + str(
                                        bid2) + " bid3: " + str(
                                        bid3) + " bid4: " + str(bid4) + " bid5: " + str(bid5) + " bid6: " + str(bid6))
                                # Close Position
                                if (alpha_value == 0) and (real_delta < 2 * positive_threshold):
                                    trade_amount = abs((position_amount_map[key]))
                                    if trade_amount > 0:
                                        buy(client, trade_amount, key, bid0, logger_trade, logger_error)
                                    continue
                                # 1/7
                                if first_divide_delta > 0:
                                    trade_amount = round(first_divide_delta / bid0, quantity_precision_map[key])
                                    if trade_amount > 0:
                                        buy(client, trade_amount, key, bid0, logger_trade, logger_error)
                                # 2/7
                                if second_divide_delta > 0:
                                    trade_amount = round(second_divide_delta / bid1, quantity_precision_map[key])
                                    if trade_amount > 0:
                                        buy(client, trade_amount, key, bid1, logger_trade, logger_error)
                                # 3/7
                                if third_divide_delta > 0:
                                    trade_amount = round(third_divide_delta / bid2, quantity_precision_map[key])
                                    if trade_amount > 0:
                                        buy(client, trade_amount, key, bid2, logger_trade, logger_error)
                                # 4/7
                                if forth_divide_delta > 0:
                                    trade_amount = round(forth_divide_delta / bid3, quantity_precision_map[key])
                                    if trade_amount > 0:
                                        buy(client, trade_amount, key, bid3, logger_trade, logger_error)
                                # 5/7
                                if fifth_divide_delta > 0:
                                    trade_amount = round(fifth_divide_delta / bid4, quantity_precision_map[key])
                                    if trade_amount > 0:
                                        buy(client, trade_amount, key, bid4, logger_trade, logger_error)
                                # 6/7
                                if sixth_divide_delta > 0:
                                    trade_amount = round(sixth_divide_delta / bid5, quantity_precision_map[key])
                                    if trade_amount > 0:
                                        buy(client, trade_amount, key, bid5, logger_trade, logger_error)
                                # 7/7
                                if seventh_divide_delta > 0:
                                    trade_amount = round(seventh_divide_delta / bid6, quantity_precision_map[key])
                                    if trade_amount > 0:
                                        buy(client, trade_amount, key, bid6, logger_trade, logger_error)
                        # For short symbols
                        elif delta < negative_threshold:
                            symbol_set.add(key)
                            ask0, ask1, ask2, ask3, ask4, ask5, ask6 = query_asks(client, key, price_precision_map[key],
                                                                                  logger_error)
                            if (ask0 > 0) and (ask1 > 0) and (ask2 > 0) and (ask3 > 0) and (ask4 > 0) and (
                                    ask5 > 0) and (ask6 > 0):
                                ask1 = min(ask1, round(ask0 * 1.0001, price_precision_map[key]))
                                ask2 = min(ask2, round(ask0 * 1.0002, price_precision_map[key]))
                                ask3 = min(ask3, round(ask0 * 1.0003, price_precision_map[key]))
                                ask4 = min(ask4, round(ask0 * 1.0004, price_precision_map[key]))
                                ask5 = min(ask5, round(ask0 * 1.0005, price_precision_map[key]))
                                ask6 = min(ask6, round(ask0 * 1.0006, price_precision_map[key]))
                                logger_trade.info(
                                    "ask0: " + str(ask0) + " ask1: " + str(ask1) + " ask2: " + str(
                                        ask2) + " ask3: " + str(
                                        ask3) + " ask4: " + str(ask4) + " ask5: " + str(ask5) + " ask6: " + str(ask6))
                                # Close Position
                                if (alpha_value == 0) and (real_delta > 2 * negative_threshold):
                                    trade_amount = abs((position_amount_map[key]))
                                    if trade_amount > 0:
                                        sell(client, trade_amount, key, ask0, logger_trade, logger_error)
                                    continue
                                # 1/7
                                if first_divide_delta < 0:
                                    trade_amount = round(first_divide_delta / ask0, quantity_precision_map[key])
                                    trade_amount = abs(trade_amount)
                                    if trade_amount > 0:
                                        sell(client, trade_amount, key, ask0, logger_trade, logger_error)
                                # 2/7
                                if second_divide_delta < 0:
                                    trade_amount = round(second_divide_delta / ask1, quantity_precision_map[key])
                                    trade_amount = abs(trade_amount)
                                    if trade_amount > 0:
                                        sell(client, trade_amount, key, ask1, logger_trade, logger_error)
                                # 3/7
                                if third_divide_delta < 0:
                                    trade_amount = round(third_divide_delta / ask2, quantity_precision_map[key])
                                    trade_amount = abs(trade_amount)
                                    if trade_amount > 0:
                                        sell(client, trade_amount, key, ask2, logger_trade, logger_error)
                                # 4/7
                                if forth_divide_delta < 0:
                                    trade_amount = round(forth_divide_delta / ask3, quantity_precision_map[key])
                                    trade_amount = abs(trade_amount)
                                    if trade_amount > 0:
                                        sell(client, trade_amount, key, ask3, logger_trade, logger_error)
                                # 5/7
                                if fifth_divide_delta < 0:
                                    trade_amount = round(fifth_divide_delta / ask4, quantity_precision_map[key])
                                    trade_amount = abs(trade_amount)
                                    if trade_amount > 0:
                                        sell(client, trade_amount, key, ask4, logger_trade, logger_error)
                                # 6/7
                                if sixth_divide_delta < 0:
                                    trade_amount = round(sixth_divide_delta / ask5, quantity_precision_map[key])
                                    trade_amount = abs(trade_amount)
                                    if trade_amount > 0:
                                        sell(client, trade_amount, key, ask5, logger_trade, logger_error)
                                # 7/7
                                if seventh_divide_delta < 0:
                                    trade_amount = round(seventh_divide_delta / ask6, quantity_precision_map[key])
                                    trade_amount = abs(trade_amount)
                                    if trade_amount > 0:
                                        sell(client, trade_amount, key, ask6, logger_trade, logger_error)
                # Step8: Wait 5 seconds for orders
                time.sleep(5)

                # Step9: Cancel all open orders
                for key in symbol_set:
                    cancel_status = True
                    while (cancel_status):
                        cancel_status = cancel_open_orders(client, key, logger_trade, logger_error)

            # Step10: Update next signal file index (after trade, once)
            signal_time = datetime.strptime(signal_index, "%Y%m%d%H%M%S")
            signal_time = signal_time + time_delta
            signal_index = signal_time.strftime("%Y%m%d%H%M00")

            # Trade success ratio
            target_value = 0.0
            real_value = 0.0
            update_positions(client, position_value_map, position_amount_map, logger_error)
            for key, value in real_delta_dict.items():
                f_value = abs(float(value))
                f_position_value = abs(float(position_value_map[key]))
                if (f_value > 5.0) and (f_position_value > 5.0):
                    real_value = real_value + f_position_value
                    target_value = target_value + f_value
            if target_value > 0:
                deal_ratio = real_value / target_value
                logger_trade.info(f"Real Volume: {real_value}, Target Volume: {target_value}, Ratio: {deal_ratio}")
        except Exception as e:
            logger_error.info(str(e))
            notify(str(e))
            exit(1)


if __name__ == "__main__":
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

    items = list(available_token_map.items())
    random.shuffle(items)
    num_submaps = 5
    submaps = [{} for _ in range(num_submaps)]
    for idx, (key, value) in enumerate(items):
        submaps[idx % num_submaps][key] = value

    position_value_map = [copy.deepcopy(submap) for submap in submaps]
    position_amount_map = [copy.deepcopy(submap) for submap in submaps]

    with open('/home/ubuntu/cryptoqt/trade/multi/configV2C.yaml', 'r') as file:
        cfg = yaml.safe_load(file)

    processes = []
    for process_config in cfg['processes']:
        i = process_config['id']
        api_key = process_config['api_key']
        api_secret = process_config['api_secret']

        process = multiprocessing.Process(
            target=process_task,
            args=(position_value_map[i], position_amount_map[i], api_key, api_secret, i, cfg)
        )
        processes.append(process)
        process.start()

    for process in processes:
        process.join()