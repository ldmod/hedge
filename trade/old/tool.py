import random
import os
import math
import csv
import datetime
import requests
import pandas as pd
from dingtalkchatbot.chatbot import DingtalkChatbot
import time
import logging
import sys
from itertools import repeat
from decimal import Decimal, ROUND_HALF_UP
from binance.um_futures import UMFutures
from datetime import datetime, timedelta

# 钉钉通知
def notify(msg):
    webhook = "https://oapi.dingtalk.com/robot/send?access_token=08981a30db7ee421f6f910cb1dbe9b722bb18420b1c54d9b5cdad300470d2cda"
    xiaoding = DingtalkChatbot(webhook)
    xiaoding.send_text(msg, is_at_all=False)

# 日志初始化
def get_logger(logger_name,log_file,level=logging.INFO):
  logger = logging.getLogger(logger_name)
  formatter = logging.Formatter('%(asctime)s : %(message)s', "%Y-%m-%d %H:%M:%S")
  fileHandler = logging.FileHandler(log_file, mode='a')
  fileHandler.setFormatter(formatter)

  logger.setLevel(level)
  logger.addHandler(fileHandler)
  return logger

# 获取买1价格
def query_bids(client, symbol, precision, logger_error):
    try:
        depth = client.depth(symbol=symbol, limit=5)
        if depth['bids'] is None:
          logger_error.info("get bids fail 1"+"  ")
          return Decimal('-1'), Decimal('-1')
        if depth['bids'][0] is None:
          logger_error.info("get bids fail 2"+"  ")
          return Decimal('-1'), Decimal('-1')
        return (round(Decimal(depth['bids'][0][0]), precision), round(Decimal(depth['bids'][1][0]), precision))
    except Exception as e:
        logger_error.info("error"+str(e)+"  ")
        return Decimal('-1'), Decimal('-1')


# 获取卖1价格
def query_asks(client, symbol, precision, logger_error):
    try:
        depth = client.depth(symbol=symbol, limit=5)
        if depth['asks'] is None:
          logger_error.info("get asks fail 1"+"  ")
          return Decimal('-1'), Decimal('-1')
        if depth['asks'][0] is None:
          logger_error.info("get asks fail 2"+"  ")
          return Decimal('-1'), Decimal('-1')
        return (round(Decimal(depth['asks'][0][0]), precision), round(Decimal(depth['asks'][1][0]), precision))
    except Exception as e:
        logger_error.info("error"+str(e)+"  ")
        return Decimal('-1'), Decimal('-1')

# 卖
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
        logger_trade.info("sell start: " + str(symbol) + ", amount: " + str(amount) + ", price: " + str(price))
        client.new_order(**params)
        logger_trade.info("卖出: " + str(symbol) + ", amount: " + str(amount) + ", price: " + str(price))
        return False
    except Exception as e:
        logger_error.info(str(e)+"  ")
        time.sleep(1)
        return True

# 买
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
        logger_trade.info("buy start: " + str(symbol) + ", amount: " + str(amount) + ", price: " + str(price))
        client.new_order(**params)
        logger_trade.info("买入: " + str(symbol) + ", amount: " + str(amount) + ", price: " + str(price))
        return False
    except Exception as e:
        logger_error.info(str(e)+"  ")
        time.sleep(1)
        return True

# 更新持仓模式(单向)
def modify_position_side(client, logger_error):
    try:
      client.change_position_mode(dualSidePosition="false")
      return False
    except Exception as e:
      logger_error.info("error"+str(e)+"  ")
      return True

# 取消某Token所有Open订单
def cancel_open_orders(client, symbol, logger_trade, logger_error):
    try:
      client.cancel_open_orders(symbol=symbol, recvWindow=2000)
      #logger_trade.info("取消订单: " + str(symbol))
    except Exception as e:
      logger_error.info("error"+str(e)+"  ")
      return True

# 更新账户目前持仓
def update_positions(client, position_value_map, position_amount_map, logger_error):
    try:
      # 读取目前持仓数据
      data_dict = client.account()

      if not data_dict:
        logger_error.info("data is null")
        return True
      if "positions" not in data_dict:
        logger_error.info("positions not exist")
        return True

      # 更新 position_value_map 和 position_amount_map
      for position in data_dict["positions"]:
        symbol = position["symbol"]
        notional = Decimal(position["notional"])
        positionAmt = Decimal(position["positionAmt"])
        if symbol in position_value_map:
            position_value_map[symbol] = notional
        if symbol in position_amount_map:
            position_amount_map[symbol] = positionAmt

      return False
    except Exception as e:
      logger_error.info("error"+str(e)+"  ")
      return True

# 平仓(多)
def close_long_position(client, symbol, amount, logger_trade, logger_error):
    try:
        params = {
          'symbol': symbol,
          'side': 'SELL',
          'type': 'MARKET',
          'quantity': amount
        }
        client.new_order(**params)
        logger_trade.info("平仓Long: " + str(symbol) + ", amount: " + str(amount))
        return False
    except Exception as e:
        logger_error.info(str(e)+"  ")
        return True

# 平仓(空)
def close_short_position(client, symbol, amount, logger_trade, logger_error):
    try:
        params = {
          'symbol': symbol,
          'side': 'BUY',
          'type': 'MARKET',
          'quantity': abs(amount)
        }
        client.new_order(**params)
        logger_trade.info("平仓Short: " + str(symbol) + ", amount: " + str(amount))
        return False
    except Exception as e:
        logger_error.info(str(e)+"  ")
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
      logger_error.info(str(e)+"  ")

# 调整开仓杠杆(1)
def change_leverage(client, symbol, logger_error):
    try:
      client.change_leverage(symbol=symbol, leverage=2)
    except Exception as e:
      logger_error.info(str(e)+"  ")

# 当前账户余额查询
def account_balance(client, logger_error):
    try:
      return client.balance()
    except Exception as e:
      logger_error.info(str(e)+"  ")


# 获得最小交易限额(增加了10%的Buffer)
def get_threshold(symbol):
    values = {
        "BTCUSDT": Decimal(110),
        "ETHUSDT": Decimal(22),
        "BCHUSDT": Decimal(22),
        "LTCUSDT": Decimal(22),
        "ETCUSDT": Decimal(22),
        "LINKUSDT": Decimal(22)
    }
    return values.get(symbol, Decimal(5.5))

# 交易框架主体
def main():
  # 初始化客户端
  client = UMFutures(key='PpyT1CX1jt0CnPawxwAL9RcItlHv3nbVb8Vi121sQt4mmNGppvW4HthNPtBcPWuB', secret='CUQSHQK2DDG4Q3qEY4bpmUG7MJam8fo96lWtmCdl5Z6MpvehRv8uBXLFk1p0K8yV')
  # 初始化日志文件
  logfile_trade = "/tmp/alpha-trade.md"
  logfile_error = "/tmp/alpha-error.md"
  logger_trade = get_logger("logger1", logfile_trade)
  logger_error = get_logger("logger2", logfile_error)

  # 信号文件起始下标
  signal_index = '20240625210000'
  # 信号文件下标步进
  time_delta = timedelta(minutes=30)
  # 信号文件后缀
  signal_suffix = '_book.csv'

  # Token的价格精度(3表示价格是小数点后3位)
  price_precision_map = {'BTCUSDT': 2, 'ETHUSDT': 2, 'BCHUSDT': 2, 'XRPUSDT': 4, 'EOSUSDT': 3, 'LTCUSDT': 2, 'TRXUSDT': 5, 'ETCUSDT': 3, 'LINKUSDT': 3, 'XLMUSDT': 5, 'ADAUSDT': 5, 'XMRUSDT': 2, 'DASHUSDT': 2, 'ZECUSDT': 2, 'XTZUSDT': 3, 'BNBUSDT': 3, 'ATOMUSDT': 3, 'ONTUSDT': 4, 'IOTAUSDT': 4, 'BATUSDT': 4, 'VETUSDT': 6, 'NEOUSDT': 3, 'QTUMUSDT': 3, 'IOSTUSDT': 6, 'THETAUSDT': 4, 'ALGOUSDT': 4, 'ZILUSDT': 5, 'KNCUSDT': 5, 'ZRXUSDT': 4, 'COMPUSDT': 2, 'OMGUSDT': 4, 'DOGEUSDT': 6, 'SXPUSDT': 4, 'KAVAUSDT': 4, 'BANDUSDT': 4, 'RLCUSDT': 4, 'WAVESUSDT': 4, 'MKRUSDT': 2, 'SNXUSDT': 3, 'DOTUSDT': 3, 'DEFIUSDT': 1, 'YFIUSDT': 1, 'BALUSDT': 3, 'CRVUSDT': 3, 'TRBUSDT': 3, 'RUNEUSDT': 4, 'SUSHIUSDT': 4, 'EGLDUSDT': 3, 'SOLUSDT': 4, 'ICXUSDT': 4, 'STORJUSDT': 4, 'BLZUSDT': 5, 'UNIUSDT': 4, 'AVAXUSDT': 4, 'FTMUSDT': 6, 'ENJUSDT': 5, 'FLMUSDT': 4, 'RENUSDT': 5, 'KSMUSDT': 3, 'NEARUSDT': 4, 'AAVEUSDT': 3, 'FILUSDT': 3, 'RSRUSDT': 6, 'LRCUSDT': 5, 'MATICUSDT': 5, 'OCEANUSDT': 5, 'CVCUSDT': 5, 'BELUSDT': 5, 'CTKUSDT': 5, 'AXSUSDT': 5, 'ALPHAUSDT': 5, 'ZENUSDT': 3, 'SKLUSDT': 5, 'GRTUSDT': 5, '1INCHUSDT': 4, 'CHZUSDT': 5, 'SANDUSDT': 5, 'ANKRUSDT': 6, 'LITUSDT': 3, 'UNFIUSDT': 3, 'REEFUSDT': 6, 'RVNUSDT': 5, 'SFPUSDT': 4, 'XEMUSDT': 4, 'BTCSTUSDT': 3, 'COTIUSDT': 5, 'CHRUSDT': 4, 'MANAUSDT': 4, 'ALICEUSDT': 3, 'HBARUSDT': 5, 'ONEUSDT': 5, 'LINAUSDT': 5, 'STMXUSDT': 5, 'DENTUSDT': 6, 'CELRUSDT': 5, 'HOTUSDT': 6, 'MTLUSDT': 4, 'OGNUSDT': 4, 'NKNUSDT': 5, 'SCUSDT': 6, 'DGBUSDT': 5, '1000SHIBUSDT': 6, 'BAKEUSDT': 4, 'GTCUSDT': 3, 'BTCDOMUSDT': 1, 'IOTXUSDT': 5, 'RAYUSDT': 3, 'C98USDT': 4, 'MASKUSDT': 4, 'ATAUSDT': 4, 'DYDXUSDT': 3, '1000XECUSDT': 5, 'GALAUSDT': 5, 'CELOUSDT': 3, 'ARUSDT': 3, 'KLAYUSDT': 4, 'ARPAUSDT': 5, 'CTSIUSDT': 4, 'LPTUSDT': 3, 'ENSUSDT': 3, 'PEOPLEUSDT': 5, 'ROSEUSDT': 5, 'DUSKUSDT': 5, 'FLOWUSDT': 3, 'IMXUSDT': 4, 'API3USDT': 4, 'GMTUSDT': 5, 'APEUSDT': 4, 'WOOUSDT': 5, 'FTTUSDT': 4, 'JASMYUSDT': 6, 'DARUSDT': 4, 'GALUSDT': 5, 'OPUSDT': 7, 'INJUSDT': 6, 'STGUSDT': 7, 'SPELLUSDT': 7, '1000LUNCUSDT': 7, 'LUNA2USDT': 7, 'LDOUSDT': 6, 'CVXUSDT': 6, 'ICPUSDT': 6, 'APTUSDT': 5, 'QNTUSDT': 6, 'FETUSDT': 7, 'FXSUSDT': 6, 'HOOKUSDT': 6, 'MAGICUSDT': 6, 'TUSDT': 7, 'RNDRUSDT': 6, 'HIGHUSDT': 6, 'MINAUSDT': 7, 'ASTRUSDT': 7, 'AGIXUSDT': 7, 'PHBUSDT': 7, 'GMXUSDT': 6, 'CFXUSDT': 7, 'STXUSDT': 7, 'BNXUSDT': 6, 'ACHUSDT': 7, 'SSVUSDT': 6, 'CKBUSDT': 7, 'PERPUSDT': 6, 'TRUUSDT': 7, 'LQTYUSDT': 6, 'USDCUSDT': 7, 'IDUSDT': 7, 'ARBUSDT': 6, 'JOEUSDT': 7, 'TLMUSDT': 7, 'AMBUSDT': 7, 'LEVERUSDT': 7, 'RDNTUSDT': 7, 'HFTUSDT': 7, 'XVSUSDT': 6, 'BLURUSDT': 7, 'EDUUSDT': 7, 'IDEXUSDT': 7, 'SUIUSDT': 6, '1000PEPEUSDT': 7, '1000FLOKIUSDT': 7, 'UMAUSDT': 6, 'RADUSDT': 6, 'KEYUSDT': 7, 'COMBOUSDT': 6, 'NMRUSDT': 6, 'MAVUSDT': 7, 'MDTUSDT': 7, 'XVGUSDT': 7, 'WLDUSDT': 7, 'PENDLEUSDT': 7, 'ARKMUSDT': 7, 'AGLDUSDT': 7, 'YGGUSDT': 7, 'DODOXUSDT': 7, 'BNTUSDT': 7, 'OXTUSDT': 7, 'SEIUSDT': 7, 'CYBERUSDT': 6, 'HIFIUSDT': 7, 'ARKUSDT': 7, 'FRONTUSDT': 7, 'GLMRUSDT': 7, 'BICOUSDT': 7, 'STRAXUSDT': 7, 'LOOMUSDT': 7, 'BIGTIMEUSDT': 7, 'BONDUSDT': 6, 'ORBSUSDT': 7, 'STPTUSDT': 7, 'WAXPUSDT': 7, 'BSVUSDT': 5, 'RIFUSDT': 7, 'POLYXUSDT': 7, 'GASUSDT': 6, 'POWRUSDT': 7, 'SLPUSDT': 7, 'TIAUSDT': 7, 'SNTUSDT': 7, 'CAKEUSDT': 7, 'MEMEUSDT': 7, 'TWTUSDT': 6, 'TOKENUSDT': 7, 'ORDIUSDT': 6, 'STEEMUSDT': 6, 'BADGERUSDT': 6, 'ILVUSDT': 5, 'NTRNUSDT': 6, 'KASUSDT': 7, 'BEAMXUSDT': 7, '1000BONKUSDT': 7, 'PYTHUSDT': 7, 'SUPERUSDT': 7, 'USTCUSDT': 7, 'ONGUSDT': 7, 'ETHWUSDT': 6, 'JTOUSDT': 6, '1000SATSUSDT': 7, 'AUCTIONUSDT': 6, '1000RATSUSDT': 7, 'ACEUSDT': 6, 'MOVRUSDT': 6, 'NFPUSDT': 7, 'AIUSDT': 6, 'XAIUSDT': 7, 'WIFUSDT': 7, 'MANTAUSDT': 7, 'ONDOUSDT': 7, 'LSKUSDT': 6, 'ALTUSDT': 7, 'JUPUSDT': 7, 'ZETAUSDT': 6, 'RONINUSDT': 6, 'DYMUSDT': 6, 'OMUSDT': 7, 'PIXELUSDT': 7, 'STRKUSDT': 7, 'MAVIAUSDT': 7, 'GLMUSDT': 7, 'PORTALUSDT': 7, 'TONUSDT': 7, 'AXLUSDT': 7, 'MYROUSDT': 7, 'METISUSDT': 4, 'AEVOUSDT': 7, 'VANRYUSDT': 7, 'BOMEUSDT': 7, 'ETHFIUSDT': 7, 'ENAUSDT': 7, 'WUSDT': 7, 'TNSRUSDT': 7, 'SAGAUSDT': 7, 'TAOUSDT': 2, 'OMNIUSDT': 4, 'REZUSDT': 7, 'BBUSDT': 7, 'NOTUSDT': 7, 'TURBOUSDT': 7, 'IOUSDT': 7}
  quantity_precision_map = {'BTCUSDT': 3, 'ETHUSDT': 3, 'BCHUSDT': 3, 'XRPUSDT': 1, 'EOSUSDT': 1, 'LTCUSDT': 3, 'TRXUSDT': 0, 'ETCUSDT': 2, 'LINKUSDT': 2, 'XLMUSDT': 0, 'ADAUSDT': 0, 'XMRUSDT': 3, 'DASHUSDT': 3, 'ZECUSDT': 3, 'XTZUSDT': 1, 'BNBUSDT': 2, 'ATOMUSDT': 2, 'ONTUSDT': 1, 'IOTAUSDT': 1, 'BATUSDT': 1, 'VETUSDT': 0, 'NEOUSDT': 2, 'QTUMUSDT': 1, 'IOSTUSDT': 0, 'THETAUSDT': 1, 'ALGOUSDT': 1, 'ZILUSDT': 0, 'KNCUSDT': 0, 'ZRXUSDT': 1, 'COMPUSDT': 3, 'OMGUSDT': 1, 'DOGEUSDT': 0, 'SXPUSDT': 1, 'KAVAUSDT': 1, 'BANDUSDT': 1, 'RLCUSDT': 1, 'WAVESUSDT': 1, 'MKRUSDT': 3, 'SNXUSDT': 1, 'DOTUSDT': 1, 'DEFIUSDT': 3, 'YFIUSDT': 3, 'BALUSDT': 1, 'CRVUSDT': 1, 'TRBUSDT': 1, 'RUNEUSDT': 0, 'SUSHIUSDT': 0, 'EGLDUSDT': 1, 'SOLUSDT': 0, 'ICXUSDT': 0, 'STORJUSDT': 0, 'BLZUSDT': 0, 'UNIUSDT': 0, 'AVAXUSDT': 0, 'FTMUSDT': 0, 'ENJUSDT': 0, 'FLMUSDT': 0, 'RENUSDT': 0, 'KSMUSDT': 1, 'NEARUSDT': 0, 'AAVEUSDT': 1, 'FILUSDT': 1, 'RSRUSDT': 0, 'LRCUSDT': 0, 'MATICUSDT': 0, 'OCEANUSDT': 0, 'CVCUSDT': 0, 'BELUSDT': 0, 'CTKUSDT': 0, 'AXSUSDT': 0, 'ALPHAUSDT': 0, 'ZENUSDT': 1, 'SKLUSDT': 0, 'GRTUSDT': 0, '1INCHUSDT': 0, 'CHZUSDT': 0, 'SANDUSDT': 0, 'ANKRUSDT': 0, 'LITUSDT': 1, 'UNFIUSDT': 1, 'REEFUSDT': 0, 'RVNUSDT': 0, 'SFPUSDT': 0, 'XEMUSDT': 0, 'BTCSTUSDT': 1, 'COTIUSDT': 0, 'CHRUSDT': 0, 'MANAUSDT': 0, 'ALICEUSDT': 1, 'HBARUSDT': 0, 'ONEUSDT': 0, 'LINAUSDT': 0, 'STMXUSDT': 0, 'DENTUSDT': 0, 'CELRUSDT': 0, 'HOTUSDT': 0, 'MTLUSDT': 0, 'OGNUSDT': 0, 'NKNUSDT': 0, 'SCUSDT': 0, 'DGBUSDT': 0, '1000SHIBUSDT': 0, 'BAKEUSDT': 0, 'GTCUSDT': 1, 'BTCDOMUSDT': 3, 'IOTXUSDT': 0, 'RAYUSDT': 1, 'C98USDT': 0, 'MASKUSDT': 0, 'ATAUSDT': 0, 'DYDXUSDT': 1, '1000XECUSDT': 0, 'GALAUSDT': 0, 'CELOUSDT': 1, 'ARUSDT': 1, 'KLAYUSDT': 1, 'ARPAUSDT': 0, 'CTSIUSDT': 0, 'LPTUSDT': 1, 'ENSUSDT': 1, 'PEOPLEUSDT': 0, 'ROSEUSDT': 0, 'DUSKUSDT': 0, 'FLOWUSDT': 1, 'IMXUSDT': 0, 'API3USDT': 1, 'GMTUSDT': 0, 'APEUSDT': 0, 'WOOUSDT': 0, 'FTTUSDT': 1, 'JASMYUSDT': 0, 'DARUSDT': 1, 'GALUSDT': 0, 'OPUSDT': 1, 'INJUSDT': 1, 'STGUSDT': 0, 'SPELLUSDT': 0, '1000LUNCUSDT': 0, 'LUNA2USDT': 0, 'LDOUSDT': 0, 'CVXUSDT': 0, 'ICPUSDT': 0, 'APTUSDT': 1, 'QNTUSDT': 1, 'FETUSDT': 0, 'FXSUSDT': 1, 'HOOKUSDT': 1, 'MAGICUSDT': 1, 'TUSDT': 0, 'RNDRUSDT': 1, 'HIGHUSDT': 1, 'MINAUSDT': 0, 'ASTRUSDT': 0, 'AGIXUSDT': 0, 'PHBUSDT': 0, 'GMXUSDT': 2, 'CFXUSDT': 0, 'STXUSDT': 0, 'BNXUSDT': 1, 'ACHUSDT': 0, 'SSVUSDT': 2, 'CKBUSDT': 0, 'PERPUSDT': 1, 'TRUUSDT': 0, 'LQTYUSDT': 1, 'USDCUSDT': 0, 'IDUSDT': 0, 'ARBUSDT': 1, 'JOEUSDT': 0, 'TLMUSDT': 0, 'AMBUSDT': 0, 'LEVERUSDT': 0, 'RDNTUSDT': 0, 'HFTUSDT': 0, 'XVSUSDT': 1, 'BLURUSDT': 0, 'EDUUSDT': 0, 'IDEXUSDT': 0, 'SUIUSDT': 1, '1000PEPEUSDT': 0, '1000FLOKIUSDT': 0, 'UMAUSDT': 0, 'RADUSDT': 0, 'KEYUSDT': 0, 'COMBOUSDT': 1, 'NMRUSDT': 1, 'MAVUSDT': 0, 'MDTUSDT': 0, 'XVGUSDT': 0, 'WLDUSDT': 0, 'PENDLEUSDT': 0, 'ARKMUSDT': 0, 'AGLDUSDT': 0, 'YGGUSDT': 0, 'DODOXUSDT': 0, 'BNTUSDT': 0, 'OXTUSDT': 0, 'SEIUSDT': 0, 'CYBERUSDT': 1, 'HIFIUSDT': 0, 'ARKUSDT': 0, 'FRONTUSDT': 0, 'GLMRUSDT': 0, 'BICOUSDT': 0, 'STRAXUSDT': 0, 'LOOMUSDT': 0, 'BIGTIMEUSDT': 0, 'BONDUSDT': 1, 'ORBSUSDT': 0, 'STPTUSDT': 0, 'WAXPUSDT': 0, 'BSVUSDT': 1, 'RIFUSDT': 0, 'POLYXUSDT': 0, 'GASUSDT': 1, 'POWRUSDT': 0, 'SLPUSDT': 0, 'TIAUSDT': 0, 'SNTUSDT': 0, 'CAKEUSDT': 0, 'MEMEUSDT': 0, 'TWTUSDT': 0, 'TOKENUSDT': 0, 'ORDIUSDT': 1, 'STEEMUSDT': 0, 'BADGERUSDT': 0, 'ILVUSDT': 1, 'NTRNUSDT': 0, 'KASUSDT': 0, 'BEAMXUSDT': 0, '1000BONKUSDT': 0, 'PYTHUSDT': 0, 'SUPERUSDT': 0, 'USTCUSDT': 0, 'ONGUSDT': 0, 'ETHWUSDT': 0, 'JTOUSDT': 0, '1000SATSUSDT': 0, 'AUCTIONUSDT': 2, '1000RATSUSDT': 0, 'ACEUSDT': 2, 'MOVRUSDT': 2, 'NFPUSDT': 1, 'AIUSDT': 0, 'XAIUSDT': 0, 'WIFUSDT': 1, 'MANTAUSDT': 1, 'ONDOUSDT': 1, 'LSKUSDT': 0, 'ALTUSDT': 0, 'JUPUSDT': 0, 'ZETAUSDT': 0, 'RONINUSDT': 1, 'DYMUSDT': 1, 'OMUSDT': 1, 'PIXELUSDT': 0, 'STRKUSDT': 1, 'MAVIAUSDT': 1, 'GLMUSDT': 0, 'PORTALUSDT': 1, 'TONUSDT': 1, 'AXLUSDT': 1, 'MYROUSDT': 0, 'METISUSDT': 2, 'AEVOUSDT': 1, 'VANRYUSDT': 0, 'BOMEUSDT': 0, 'ETHFIUSDT': 1, 'ENAUSDT': 0, 'WUSDT': 1, 'TNSRUSDT': 1, 'SAGAUSDT': 1, 'TAOUSDT': 3, 'OMNIUSDT': 2, 'REZUSDT': 0, 'BBUSDT': 0, 'NOTUSDT': 0, 'TURBOUSDT': 0, 'IOUSDT': 1}
  base_asset_precision_map = {'BTCUSDT': 8, 'ETHUSDT': 8, 'BCHUSDT': 8, 'XRPUSDT': 8, 'EOSUSDT': 8, 'LTCUSDT': 8, 'TRXUSDT': 8, 'ETCUSDT': 8, 'LINKUSDT': 8, 'XLMUSDT': 8, 'ADAUSDT': 8, 'XMRUSDT': 8, 'DASHUSDT': 8, 'ZECUSDT': 8, 'XTZUSDT': 8, 'BNBUSDT': 8, 'ATOMUSDT': 8, 'ONTUSDT': 8, 'IOTAUSDT': 8, 'BATUSDT': 8, 'VETUSDT': 8, 'NEOUSDT': 8, 'QTUMUSDT': 8, 'IOSTUSDT': 8, 'THETAUSDT': 8, 'ALGOUSDT': 8, 'ZILUSDT': 8, 'KNCUSDT': 8, 'ZRXUSDT': 8, 'COMPUSDT': 8, 'OMGUSDT': 8, 'DOGEUSDT': 8, 'SXPUSDT': 8, 'KAVAUSDT': 8, 'BANDUSDT': 8, 'RLCUSDT': 8, 'WAVESUSDT': 8, 'MKRUSDT': 8, 'SNXUSDT': 8, 'DOTUSDT': 8, 'DEFIUSDT': 8, 'YFIUSDT': 8, 'BALUSDT': 8, 'CRVUSDT': 8, 'TRBUSDT': 8, 'RUNEUSDT': 8, 'SUSHIUSDT': 8, 'EGLDUSDT': 8, 'SOLUSDT': 8, 'ICXUSDT': 8, 'STORJUSDT': 8, 'BLZUSDT': 8, 'UNIUSDT': 8, 'AVAXUSDT': 8, 'FTMUSDT': 8, 'ENJUSDT': 8, 'FLMUSDT': 8, 'RENUSDT': 8, 'KSMUSDT': 8, 'NEARUSDT': 8, 'AAVEUSDT': 8, 'FILUSDT': 8, 'RSRUSDT': 8, 'LRCUSDT': 8, 'MATICUSDT': 8, 'OCEANUSDT': 8, 'CVCUSDT': 8, 'BELUSDT': 8, 'CTKUSDT': 8, 'AXSUSDT': 8, 'ALPHAUSDT': 8, 'ZENUSDT': 8, 'SKLUSDT': 8, 'GRTUSDT': 8, '1INCHUSDT': 8, 'CHZUSDT': 8, 'SANDUSDT': 8, 'ANKRUSDT': 8, 'LITUSDT': 8, 'UNFIUSDT': 8, 'REEFUSDT': 8, 'RVNUSDT': 8, 'SFPUSDT': 8, 'XEMUSDT': 8, 'BTCSTUSDT': 8, 'COTIUSDT': 8, 'CHRUSDT': 8, 'MANAUSDT': 8, 'ALICEUSDT': 8, 'HBARUSDT': 8, 'ONEUSDT': 8, 'LINAUSDT': 8, 'STMXUSDT': 8, 'DENTUSDT': 8, 'CELRUSDT': 8, 'HOTUSDT': 8, 'MTLUSDT': 8, 'OGNUSDT': 8, 'NKNUSDT': 8, 'SCUSDT': 8, 'DGBUSDT': 8, '1000SHIBUSDT': 8, 'BAKEUSDT': 8, 'GTCUSDT': 8, 'BTCDOMUSDT': 8, 'IOTXUSDT': 8, 'RAYUSDT': 8, 'C98USDT': 8, 'MASKUSDT': 8, 'ATAUSDT': 8, 'DYDXUSDT': 8, '1000XECUSDT': 8, 'GALAUSDT': 8, 'CELOUSDT': 8, 'ARUSDT': 8, 'KLAYUSDT': 8, 'ARPAUSDT': 8, 'CTSIUSDT': 8, 'LPTUSDT': 8, 'ENSUSDT': 8, 'PEOPLEUSDT': 8, 'ROSEUSDT': 8, 'DUSKUSDT': 8, 'FLOWUSDT': 8, 'IMXUSDT': 8, 'API3USDT': 8, 'GMTUSDT': 8, 'APEUSDT': 8, 'WOOUSDT': 8, 'FTTUSDT': 8, 'JASMYUSDT': 8, 'DARUSDT': 8, 'GALUSDT': 8, 'OPUSDT': 8, 'INJUSDT': 8, 'STGUSDT': 8, 'SPELLUSDT': 8, '1000LUNCUSDT': 8, 'LUNA2USDT': 8, 'LDOUSDT': 8, 'CVXUSDT': 8, 'ICPUSDT': 8, 'APTUSDT': 8, 'QNTUSDT': 8, 'FETUSDT': 8, 'FXSUSDT': 8, 'HOOKUSDT': 8, 'MAGICUSDT': 8, 'TUSDT': 8, 'RNDRUSDT': 8, 'HIGHUSDT': 8, 'MINAUSDT': 8, 'ASTRUSDT': 8, 'AGIXUSDT': 8, 'PHBUSDT': 8, 'GMXUSDT': 8, 'CFXUSDT': 8, 'STXUSDT': 8, 'BNXUSDT': 8, 'ACHUSDT': 8, 'SSVUSDT': 8, 'CKBUSDT': 8, 'PERPUSDT': 8, 'TRUUSDT': 8, 'LQTYUSDT': 8, 'USDCUSDT': 8, 'IDUSDT': 8, 'ARBUSDT': 8, 'JOEUSDT': 8, 'TLMUSDT': 8, 'AMBUSDT': 8, 'LEVERUSDT': 8, 'RDNTUSDT': 8, 'HFTUSDT': 8, 'XVSUSDT': 8, 'BLURUSDT': 8, 'EDUUSDT': 8, 'IDEXUSDT': 8, 'SUIUSDT': 8, '1000PEPEUSDT': 8, '1000FLOKIUSDT': 8, 'UMAUSDT': 8, 'RADUSDT': 8, 'KEYUSDT': 8, 'COMBOUSDT': 8, 'NMRUSDT': 8, 'MAVUSDT': 8, 'MDTUSDT': 8, 'XVGUSDT': 8, 'WLDUSDT': 8, 'PENDLEUSDT': 8, 'ARKMUSDT': 8, 'AGLDUSDT': 8, 'YGGUSDT': 8, 'DODOXUSDT': 8, 'BNTUSDT': 8, 'OXTUSDT': 8, 'SEIUSDT': 8, 'CYBERUSDT': 8, 'HIFIUSDT': 8, 'ARKUSDT': 8, 'FRONTUSDT': 8, 'GLMRUSDT': 8, 'BICOUSDT': 8, 'STRAXUSDT': 8, 'LOOMUSDT': 8, 'BIGTIMEUSDT': 8, 'BONDUSDT': 8, 'ORBSUSDT': 8, 'STPTUSDT': 8, 'WAXPUSDT': 8, 'BSVUSDT': 8, 'RIFUSDT': 8, 'POLYXUSDT': 8, 'GASUSDT': 8, 'POWRUSDT': 8, 'SLPUSDT': 8, 'TIAUSDT': 8, 'SNTUSDT': 8, 'CAKEUSDT': 8, 'MEMEUSDT': 8, 'TWTUSDT': 8, 'TOKENUSDT': 8, 'ORDIUSDT': 8, 'STEEMUSDT': 8, 'BADGERUSDT': 8, 'ILVUSDT': 8, 'NTRNUSDT': 8, 'KASUSDT': 8, 'BEAMXUSDT': 8, '1000BONKUSDT': 8, 'PYTHUSDT': 8, 'SUPERUSDT': 8, 'USTCUSDT': 8, 'ONGUSDT': 8, 'ETHWUSDT': 8, 'JTOUSDT': 8, '1000SATSUSDT': 8, 'AUCTIONUSDT': 8, '1000RATSUSDT': 8, 'ACEUSDT': 8, 'MOVRUSDT': 8, 'NFPUSDT': 8, 'AIUSDT': 8, 'XAIUSDT': 8, 'WIFUSDT': 8, 'MANTAUSDT': 8, 'ONDOUSDT': 8, 'LSKUSDT': 8, 'ALTUSDT': 8, 'JUPUSDT': 8, 'ZETAUSDT': 8, 'RONINUSDT': 8, 'DYMUSDT': 8, 'OMUSDT': 8, 'PIXELUSDT': 8, 'STRKUSDT': 8, 'MAVIAUSDT': 8, 'GLMUSDT': 8, 'PORTALUSDT': 8, 'TONUSDT': 8, 'AXLUSDT': 8, 'MYROUSDT': 8, 'METISUSDT': 8, 'AEVOUSDT': 8, 'VANRYUSDT': 8, 'BOMEUSDT': 8, 'ETHFIUSDT': 8, 'ENAUSDT': 8, 'WUSDT': 8, 'TNSRUSDT': 8, 'SAGAUSDT': 8, 'TAOUSDT': 8, 'OMNIUSDT': 8, 'REZUSDT': 8, 'BBUSDT': 8, 'NOTUSDT': 8, 'TURBOUSDT': 8, 'IOUSDT': 8}
  quote_precision_map = {'BTCUSDT': 8, 'ETHUSDT': 8, 'BCHUSDT': 8, 'XRPUSDT': 8, 'EOSUSDT': 8, 'LTCUSDT': 8, 'TRXUSDT': 8, 'ETCUSDT': 8, 'LINKUSDT': 8, 'XLMUSDT': 8, 'ADAUSDT': 8, 'XMRUSDT': 8, 'DASHUSDT': 8, 'ZECUSDT': 8, 'XTZUSDT': 8, 'BNBUSDT': 8, 'ATOMUSDT': 8, 'ONTUSDT': 8, 'IOTAUSDT': 8, 'BATUSDT': 8, 'VETUSDT': 8, 'NEOUSDT': 8, 'QTUMUSDT': 8, 'IOSTUSDT': 8, 'THETAUSDT': 8, 'ALGOUSDT': 8, 'ZILUSDT': 8, 'KNCUSDT': 8, 'ZRXUSDT': 8, 'COMPUSDT': 8, 'OMGUSDT': 8, 'DOGEUSDT': 8, 'SXPUSDT': 8, 'KAVAUSDT': 8, 'BANDUSDT': 8, 'RLCUSDT': 8, 'WAVESUSDT': 8, 'MKRUSDT': 8, 'SNXUSDT': 8, 'DOTUSDT': 8, 'DEFIUSDT': 8, 'YFIUSDT': 8, 'BALUSDT': 8, 'CRVUSDT': 8, 'TRBUSDT': 8, 'RUNEUSDT': 8, 'SUSHIUSDT': 8, 'EGLDUSDT': 8, 'SOLUSDT': 8, 'ICXUSDT': 8, 'STORJUSDT': 8, 'BLZUSDT': 8, 'UNIUSDT': 8, 'AVAXUSDT': 8, 'FTMUSDT': 8, 'ENJUSDT': 8, 'FLMUSDT': 8, 'RENUSDT': 8, 'KSMUSDT': 8, 'NEARUSDT': 8, 'AAVEUSDT': 8, 'FILUSDT': 8, 'RSRUSDT': 8, 'LRCUSDT': 8, 'MATICUSDT': 8, 'OCEANUSDT': 8, 'CVCUSDT': 8, 'BELUSDT': 8, 'CTKUSDT': 8, 'AXSUSDT': 8, 'ALPHAUSDT': 8, 'ZENUSDT': 8, 'SKLUSDT': 8, 'GRTUSDT': 8, '1INCHUSDT': 8, 'CHZUSDT': 8, 'SANDUSDT': 8, 'ANKRUSDT': 8, 'LITUSDT': 8, 'UNFIUSDT': 8, 'REEFUSDT': 8, 'RVNUSDT': 8, 'SFPUSDT': 8, 'XEMUSDT': 8, 'BTCSTUSDT': 8, 'COTIUSDT': 8, 'CHRUSDT': 8, 'MANAUSDT': 8, 'ALICEUSDT': 8, 'HBARUSDT': 8, 'ONEUSDT': 8, 'LINAUSDT': 8, 'STMXUSDT': 8, 'DENTUSDT': 8, 'CELRUSDT': 8, 'HOTUSDT': 8, 'MTLUSDT': 8, 'OGNUSDT': 8, 'NKNUSDT': 8, 'SCUSDT': 8, 'DGBUSDT': 8, '1000SHIBUSDT': 8, 'BAKEUSDT': 8, 'GTCUSDT': 8, 'BTCDOMUSDT': 8, 'IOTXUSDT': 8, 'RAYUSDT': 8, 'C98USDT': 8, 'MASKUSDT': 8, 'ATAUSDT': 8, 'DYDXUSDT': 8, '1000XECUSDT': 8, 'GALAUSDT': 8, 'CELOUSDT': 8, 'ARUSDT': 8, 'KLAYUSDT': 8, 'ARPAUSDT': 8, 'CTSIUSDT': 8, 'LPTUSDT': 8, 'ENSUSDT': 8, 'PEOPLEUSDT': 8, 'ROSEUSDT': 8, 'DUSKUSDT': 8, 'FLOWUSDT': 8, 'IMXUSDT': 8, 'API3USDT': 8, 'GMTUSDT': 8, 'APEUSDT': 8, 'WOOUSDT': 8, 'FTTUSDT': 8, 'JASMYUSDT': 8, 'DARUSDT': 8, 'GALUSDT': 8, 'OPUSDT': 8, 'INJUSDT': 8, 'STGUSDT': 8, 'SPELLUSDT': 8, '1000LUNCUSDT': 8, 'LUNA2USDT': 8, 'LDOUSDT': 8, 'CVXUSDT': 8, 'ICPUSDT': 8, 'APTUSDT': 8, 'QNTUSDT': 8, 'FETUSDT': 8, 'FXSUSDT': 8, 'HOOKUSDT': 8, 'MAGICUSDT': 8, 'TUSDT': 8, 'RNDRUSDT': 8, 'HIGHUSDT': 8, 'MINAUSDT': 8, 'ASTRUSDT': 8, 'AGIXUSDT': 8, 'PHBUSDT': 8, 'GMXUSDT': 8, 'CFXUSDT': 8, 'STXUSDT': 8, 'BNXUSDT': 8, 'ACHUSDT': 8, 'SSVUSDT': 8, 'CKBUSDT': 8, 'PERPUSDT': 8, 'TRUUSDT': 8, 'LQTYUSDT': 8, 'USDCUSDT': 8, 'IDUSDT': 8, 'ARBUSDT': 8, 'JOEUSDT': 8, 'TLMUSDT': 8, 'AMBUSDT': 8, 'LEVERUSDT': 8, 'RDNTUSDT': 8, 'HFTUSDT': 8, 'XVSUSDT': 8, 'BLURUSDT': 8, 'EDUUSDT': 8, 'IDEXUSDT': 8, 'SUIUSDT': 8, '1000PEPEUSDT': 8, '1000FLOKIUSDT': 8, 'UMAUSDT': 8, 'RADUSDT': 8, 'KEYUSDT': 8, 'COMBOUSDT': 8, 'NMRUSDT': 8, 'MAVUSDT': 8, 'MDTUSDT': 8, 'XVGUSDT': 8, 'WLDUSDT': 8, 'PENDLEUSDT': 8, 'ARKMUSDT': 8, 'AGLDUSDT': 8, 'YGGUSDT': 8, 'DODOXUSDT': 8, 'BNTUSDT': 8, 'OXTUSDT': 8, 'SEIUSDT': 8, 'CYBERUSDT': 8, 'HIFIUSDT': 8, 'ARKUSDT': 8, 'FRONTUSDT': 8, 'GLMRUSDT': 8, 'BICOUSDT': 8, 'STRAXUSDT': 8, 'LOOMUSDT': 8, 'BIGTIMEUSDT': 8, 'BONDUSDT': 8, 'ORBSUSDT': 8, 'STPTUSDT': 8, 'WAXPUSDT': 8, 'BSVUSDT': 8, 'RIFUSDT': 8, 'POLYXUSDT': 8, 'GASUSDT': 8, 'POWRUSDT': 8, 'SLPUSDT': 8, 'TIAUSDT': 8, 'SNTUSDT': 8, 'CAKEUSDT': 8, 'MEMEUSDT': 8, 'TWTUSDT': 8, 'TOKENUSDT': 8, 'ORDIUSDT': 8, 'STEEMUSDT': 8, 'BADGERUSDT': 8, 'ILVUSDT': 8, 'NTRNUSDT': 8, 'KASUSDT': 8, 'BEAMXUSDT': 8, '1000BONKUSDT': 8, 'PYTHUSDT': 8, 'SUPERUSDT': 8, 'USTCUSDT': 8, 'ONGUSDT': 8, 'ETHWUSDT': 8, 'JTOUSDT': 8, '1000SATSUSDT': 8, 'AUCTIONUSDT': 8, '1000RATSUSDT': 8, 'ACEUSDT': 8, 'MOVRUSDT': 8, 'NFPUSDT': 8, 'AIUSDT': 8, 'XAIUSDT': 8, 'WIFUSDT': 8, 'MANTAUSDT': 8, 'ONDOUSDT': 8, 'LSKUSDT': 8, 'ALTUSDT': 8, 'JUPUSDT': 8, 'ZETAUSDT': 8, 'RONINUSDT': 8, 'DYMUSDT': 8, 'OMUSDT': 8, 'PIXELUSDT': 8, 'STRKUSDT': 8, 'MAVIAUSDT': 8, 'GLMUSDT': 8, 'PORTALUSDT': 8, 'TONUSDT': 8, 'AXLUSDT': 8, 'MYROUSDT': 8, 'METISUSDT': 8, 'AEVOUSDT': 8, 'VANRYUSDT': 8, 'BOMEUSDT': 8, 'ETHFIUSDT': 8, 'ENAUSDT': 8, 'WUSDT': 8, 'TNSRUSDT': 8, 'SAGAUSDT': 8, 'TAOUSDT': 8, 'OMNIUSDT': 8, 'REZUSDT': 8, 'BBUSDT': 8, 'NOTUSDT': 8, 'TURBOUSDT': 8, 'IOUSDT': 8}

  # 目前持仓
  position_value_map = {'BTCUSDT': 0.0, 'ETHUSDT': 0.0, 'BCHUSDT': 0.0, 'XRPUSDT': 0.0, 'EOSUSDT': 0.0, 'LTCUSDT': 0.0, 'TRXUSDT': 0.0, 'ETCUSDT': 0.0, 'LINKUSDT': 0.0, 'XLMUSDT': 0.0, 'ADAUSDT': 0.0, 'XMRUSDT': 0.0, 'DASHUSDT': 0.0, 'ZECUSDT': 0.0, 'XTZUSDT': 0.0, 'BNBUSDT': 0.0, 'ATOMUSDT': 0.0, 'ONTUSDT': 0.0, 'IOTAUSDT': 0.0, 'BATUSDT': 0.0, 'VETUSDT': 0.0, 'NEOUSDT': 0.0, 'QTUMUSDT': 0.0, 'IOSTUSDT': 0.0, 'THETAUSDT': 0.0, 'ALGOUSDT': 0.0, 'ZILUSDT': 0.0, 'KNCUSDT': 0.0, 'ZRXUSDT': 0.0, 'COMPUSDT': 0.0, 'OMGUSDT': 0.0, 'DOGEUSDT': 0.0, 'SXPUSDT': 0.0, 'KAVAUSDT': 0.0, 'BANDUSDT': 0.0, 'RLCUSDT': 0.0, 'WAVESUSDT': 0.0, 'MKRUSDT': 0.0, 'SNXUSDT': 0.0, 'DOTUSDT': 0.0, 'DEFIUSDT': 0.0, 'YFIUSDT': 0.0, 'BALUSDT': 0.0, 'CRVUSDT': 0.0, 'TRBUSDT': 0.0, 'RUNEUSDT': 0.0, 'SUSHIUSDT': 0.0, 'EGLDUSDT': 0.0, 'SOLUSDT': 0.0, 'ICXUSDT': 0.0, 'STORJUSDT': 0.0, 'BLZUSDT': 0.0, 'UNIUSDT': 0.0, 'AVAXUSDT': 0.0, 'FTMUSDT': 0.0, 'ENJUSDT': 0.0, 'FLMUSDT': 0.0, 'RENUSDT': 0.0, 'KSMUSDT': 0.0, 'NEARUSDT': 0.0, 'AAVEUSDT': 0.0, 'FILUSDT': 0.0, 'RSRUSDT': 0.0, 'LRCUSDT': 0.0, 'MATICUSDT': 0.0, 'OCEANUSDT': 0.0, 'CVCUSDT': 0.0, 'BELUSDT': 0.0, 'CTKUSDT': 0.0, 'AXSUSDT': 0.0, 'ALPHAUSDT': 0.0, 'ZENUSDT': 0.0, 'SKLUSDT': 0.0, 'GRTUSDT': 0.0, '1INCHUSDT': 0.0, 'CHZUSDT': 0.0, 'SANDUSDT': 0.0, 'ANKRUSDT': 0.0, 'LITUSDT': 0.0, 'UNFIUSDT': 0.0, 'REEFUSDT': 0.0, 'RVNUSDT': 0.0, 'SFPUSDT': 0.0, 'XEMUSDT': 0.0, 'BTCSTUSDT': 0.0, 'COTIUSDT': 0.0, 'CHRUSDT': 0.0, 'MANAUSDT': 0.0, 'ALICEUSDT': 0.0, 'HBARUSDT': 0.0, 'ONEUSDT': 0.0, 'LINAUSDT': 0.0, 'STMXUSDT': 0.0, 'DENTUSDT': 0.0, 'CELRUSDT': 0.0, 'HOTUSDT': 0.0, 'MTLUSDT': 0.0, 'OGNUSDT': 0.0, 'NKNUSDT': 0.0, 'SCUSDT': 0.0, 'DGBUSDT': 0.0, '1000SHIBUSDT': 0.0, 'BAKEUSDT': 0.0, 'GTCUSDT': 0.0, 'BTCDOMUSDT': 0.0, 'IOTXUSDT': 0.0, 'RAYUSDT': 0.0, 'C98USDT': 0.0, 'MASKUSDT': 0.0, 'ATAUSDT': 0.0, 'DYDXUSDT': 0.0, '1000XECUSDT': 0.0, 'GALAUSDT': 0.0, 'CELOUSDT': 0.0, 'ARUSDT': 0.0, 'KLAYUSDT': 0.0, 'ARPAUSDT': 0.0, 'CTSIUSDT': 0.0, 'LPTUSDT': 0.0, 'ENSUSDT': 0.0, 'PEOPLEUSDT': 0.0, 'ROSEUSDT': 0.0, 'DUSKUSDT': 0.0, 'FLOWUSDT': 0.0, 'IMXUSDT': 0.0, 'API3USDT': 0.0, 'GMTUSDT': 0.0, 'APEUSDT': 0.0, 'WOOUSDT': 0.0, 'FTTUSDT': 0.0, 'JASMYUSDT': 0.0, 'DARUSDT': 0.0, 'GALUSDT': 0.0, 'OPUSDT': 0.0, 'INJUSDT': 0.0, 'STGUSDT': 0.0, 'SPELLUSDT': 0.0, '1000LUNCUSDT': 0.0, 'LUNA2USDT': 0.0, 'LDOUSDT': 0.0, 'CVXUSDT': 0.0, 'ICPUSDT': 0.0, 'APTUSDT': 0.0, 'QNTUSDT': 0.0, 'FETUSDT': 0.0, 'FXSUSDT': 0.0, 'HOOKUSDT': 0.0, 'MAGICUSDT': 0.0, 'TUSDT': 0.0, 'RNDRUSDT': 0.0, 'HIGHUSDT': 0.0, 'MINAUSDT': 0.0, 'ASTRUSDT': 0.0, 'AGIXUSDT': 0.0, 'PHBUSDT': 0.0, 'GMXUSDT': 0.0, 'CFXUSDT': 0.0, 'STXUSDT': 0.0, 'BNXUSDT': 0.0, 'ACHUSDT': 0.0, 'SSVUSDT': 0.0, 'CKBUSDT': 0.0, 'PERPUSDT': 0.0, 'TRUUSDT': 0.0, 'LQTYUSDT': 0.0, 'USDCUSDT': 0.0, 'IDUSDT': 0.0, 'ARBUSDT': 0.0, 'JOEUSDT': 0.0, 'TLMUSDT': 0.0, 'AMBUSDT': 0.0, 'LEVERUSDT': 0.0, 'RDNTUSDT': 0.0, 'HFTUSDT': 0.0, 'XVSUSDT': 0.0, 'BLURUSDT': 0.0, 'EDUUSDT': 0.0, 'IDEXUSDT': 0.0, 'SUIUSDT': 0.0, '1000PEPEUSDT': 0.0, '1000FLOKIUSDT': 0.0, 'UMAUSDT': 0.0, 'RADUSDT': 0.0, 'KEYUSDT': 0.0, 'COMBOUSDT': 0.0, 'NMRUSDT': 0.0, 'MAVUSDT': 0.0, 'MDTUSDT': 0.0, 'XVGUSDT': 0.0, 'WLDUSDT': 0.0, 'PENDLEUSDT': 0.0, 'ARKMUSDT': 0.0, 'AGLDUSDT': 0.0, 'YGGUSDT': 0.0, 'DODOXUSDT': 0.0, 'BNTUSDT': 0.0, 'OXTUSDT': 0.0, 'SEIUSDT': 0.0, 'CYBERUSDT': 0.0, 'HIFIUSDT': 0.0, 'ARKUSDT': 0.0, 'FRONTUSDT': 0.0, 'GLMRUSDT': 0.0, 'BICOUSDT': 0.0, 'STRAXUSDT': 0.0, 'LOOMUSDT': 0.0, 'BIGTIMEUSDT': 0.0, 'BONDUSDT': 0.0, 'ORBSUSDT': 0.0, 'STPTUSDT': 0.0, 'WAXPUSDT': 0.0, 'BSVUSDT': 0.0, 'RIFUSDT': 0.0, 'POLYXUSDT': 0.0, 'GASUSDT': 0.0, 'POWRUSDT': 0.0, 'SLPUSDT': 0.0, 'TIAUSDT': 0.0, 'SNTUSDT': 0.0, 'CAKEUSDT': 0.0, 'MEMEUSDT': 0.0, 'TWTUSDT': 0.0, 'TOKENUSDT': 0.0, 'ORDIUSDT': 0.0, 'STEEMUSDT': 0.0, 'BADGERUSDT': 0.0, 'ILVUSDT': 0.0, 'NTRNUSDT': 0.0, 'KASUSDT': 0.0, 'BEAMXUSDT': 0.0, '1000BONKUSDT': 0.0, 'PYTHUSDT': 0.0, 'SUPERUSDT': 0.0, 'USTCUSDT': 0.0, 'ONGUSDT': 0.0, 'ETHWUSDT': 0.0, 'JTOUSDT': 0.0, '1000SATSUSDT': 0.0, 'AUCTIONUSDT': 0.0, '1000RATSUSDT': 0.0, 'ACEUSDT': 0.0, 'MOVRUSDT': 0.0, 'NFPUSDT': 0.0, 'AIUSDT': 0.0, 'XAIUSDT': 0.0, 'WIFUSDT': 0.0, 'MANTAUSDT': 0.0, 'ONDOUSDT': 0.0, 'LSKUSDT': 0.0, 'ALTUSDT': 0.0, 'JUPUSDT': 0.0, 'ZETAUSDT': 0.0, 'RONINUSDT': 0.0, 'DYMUSDT': 0.0, 'OMUSDT': 0.0, 'PIXELUSDT': 0.0, 'STRKUSDT': 0.0, 'MAVIAUSDT': 0.0, 'GLMUSDT': 0.0, 'PORTALUSDT': 0.0, 'TONUSDT': 0.0, 'AXLUSDT': 0.0, 'MYROUSDT': 0.0, 'METISUSDT': 0.0, 'AEVOUSDT': 0.0, 'VANRYUSDT': 0.0, 'BOMEUSDT': 0.0, 'ETHFIUSDT': 0.0, 'ENAUSDT': 0.0, 'WUSDT': 0.0, 'TNSRUSDT': 0.0, 'SAGAUSDT': 0.0, 'TAOUSDT': 0.0, 'OMNIUSDT': 0.0, 'REZUSDT': 0.0, 'BBUSDT': 0.0, 'NOTUSDT': 0.0, 'TURBOUSDT': 0.0, 'IOUSDT': 0.0}
  position_amount_map = {'BTCUSDT': 0.0, 'ETHUSDT': 0.0, 'BCHUSDT': 0.0, 'XRPUSDT': 0.0, 'EOSUSDT': 0.0, 'LTCUSDT': 0.0, 'TRXUSDT': 0.0, 'ETCUSDT': 0.0, 'LINKUSDT': 0.0, 'XLMUSDT': 0.0, 'ADAUSDT': 0.0, 'XMRUSDT': 0.0, 'DASHUSDT': 0.0, 'ZECUSDT': 0.0, 'XTZUSDT': 0.0, 'BNBUSDT': 0.0, 'ATOMUSDT': 0.0, 'ONTUSDT': 0.0, 'IOTAUSDT': 0.0, 'BATUSDT': 0.0, 'VETUSDT': 0.0, 'NEOUSDT': 0.0, 'QTUMUSDT': 0.0, 'IOSTUSDT': 0.0, 'THETAUSDT': 0.0, 'ALGOUSDT': 0.0, 'ZILUSDT': 0.0, 'KNCUSDT': 0.0, 'ZRXUSDT': 0.0, 'COMPUSDT': 0.0, 'OMGUSDT': 0.0, 'DOGEUSDT': 0.0, 'SXPUSDT': 0.0, 'KAVAUSDT': 0.0, 'BANDUSDT': 0.0, 'RLCUSDT': 0.0, 'WAVESUSDT': 0.0, 'MKRUSDT': 0.0, 'SNXUSDT': 0.0, 'DOTUSDT': 0.0, 'DEFIUSDT': 0.0, 'YFIUSDT': 0.0, 'BALUSDT': 0.0, 'CRVUSDT': 0.0, 'TRBUSDT': 0.0, 'RUNEUSDT': 0.0, 'SUSHIUSDT': 0.0, 'EGLDUSDT': 0.0, 'SOLUSDT': 0.0, 'ICXUSDT': 0.0, 'STORJUSDT': 0.0, 'BLZUSDT': 0.0, 'UNIUSDT': 0.0, 'AVAXUSDT': 0.0, 'FTMUSDT': 0.0, 'ENJUSDT': 0.0, 'FLMUSDT': 0.0, 'RENUSDT': 0.0, 'KSMUSDT': 0.0, 'NEARUSDT': 0.0, 'AAVEUSDT': 0.0, 'FILUSDT': 0.0, 'RSRUSDT': 0.0, 'LRCUSDT': 0.0, 'MATICUSDT': 0.0, 'OCEANUSDT': 0.0, 'CVCUSDT': 0.0, 'BELUSDT': 0.0, 'CTKUSDT': 0.0, 'AXSUSDT': 0.0, 'ALPHAUSDT': 0.0, 'ZENUSDT': 0.0, 'SKLUSDT': 0.0, 'GRTUSDT': 0.0, '1INCHUSDT': 0.0, 'CHZUSDT': 0.0, 'SANDUSDT': 0.0, 'ANKRUSDT': 0.0, 'LITUSDT': 0.0, 'UNFIUSDT': 0.0, 'REEFUSDT': 0.0, 'RVNUSDT': 0.0, 'SFPUSDT': 0.0, 'XEMUSDT': 0.0, 'BTCSTUSDT': 0.0, 'COTIUSDT': 0.0, 'CHRUSDT': 0.0, 'MANAUSDT': 0.0, 'ALICEUSDT': 0.0, 'HBARUSDT': 0.0, 'ONEUSDT': 0.0, 'LINAUSDT': 0.0, 'STMXUSDT': 0.0, 'DENTUSDT': 0.0, 'CELRUSDT': 0.0, 'HOTUSDT': 0.0, 'MTLUSDT': 0.0, 'OGNUSDT': 0.0, 'NKNUSDT': 0.0, 'SCUSDT': 0.0, 'DGBUSDT': 0.0, '1000SHIBUSDT': 0.0, 'BAKEUSDT': 0.0, 'GTCUSDT': 0.0, 'BTCDOMUSDT': 0.0, 'IOTXUSDT': 0.0, 'RAYUSDT': 0.0, 'C98USDT': 0.0, 'MASKUSDT': 0.0, 'ATAUSDT': 0.0, 'DYDXUSDT': 0.0, '1000XECUSDT': 0.0, 'GALAUSDT': 0.0, 'CELOUSDT': 0.0, 'ARUSDT': 0.0, 'KLAYUSDT': 0.0, 'ARPAUSDT': 0.0, 'CTSIUSDT': 0.0, 'LPTUSDT': 0.0, 'ENSUSDT': 0.0, 'PEOPLEUSDT': 0.0, 'ROSEUSDT': 0.0, 'DUSKUSDT': 0.0, 'FLOWUSDT': 0.0, 'IMXUSDT': 0.0, 'API3USDT': 0.0, 'GMTUSDT': 0.0, 'APEUSDT': 0.0, 'WOOUSDT': 0.0, 'FTTUSDT': 0.0, 'JASMYUSDT': 0.0, 'DARUSDT': 0.0, 'GALUSDT': 0.0, 'OPUSDT': 0.0, 'INJUSDT': 0.0, 'STGUSDT': 0.0, 'SPELLUSDT': 0.0, '1000LUNCUSDT': 0.0, 'LUNA2USDT': 0.0, 'LDOUSDT': 0.0, 'CVXUSDT': 0.0, 'ICPUSDT': 0.0, 'APTUSDT': 0.0, 'QNTUSDT': 0.0, 'FETUSDT': 0.0, 'FXSUSDT': 0.0, 'HOOKUSDT': 0.0, 'MAGICUSDT': 0.0, 'TUSDT': 0.0, 'RNDRUSDT': 0.0, 'HIGHUSDT': 0.0, 'MINAUSDT': 0.0, 'ASTRUSDT': 0.0, 'AGIXUSDT': 0.0, 'PHBUSDT': 0.0, 'GMXUSDT': 0.0, 'CFXUSDT': 0.0, 'STXUSDT': 0.0, 'BNXUSDT': 0.0, 'ACHUSDT': 0.0, 'SSVUSDT': 0.0, 'CKBUSDT': 0.0, 'PERPUSDT': 0.0, 'TRUUSDT': 0.0, 'LQTYUSDT': 0.0, 'USDCUSDT': 0.0, 'IDUSDT': 0.0, 'ARBUSDT': 0.0, 'JOEUSDT': 0.0, 'TLMUSDT': 0.0, 'AMBUSDT': 0.0, 'LEVERUSDT': 0.0, 'RDNTUSDT': 0.0, 'HFTUSDT': 0.0, 'XVSUSDT': 0.0, 'BLURUSDT': 0.0, 'EDUUSDT': 0.0, 'IDEXUSDT': 0.0, 'SUIUSDT': 0.0, '1000PEPEUSDT': 0.0, '1000FLOKIUSDT': 0.0, 'UMAUSDT': 0.0, 'RADUSDT': 0.0, 'KEYUSDT': 0.0, 'COMBOUSDT': 0.0, 'NMRUSDT': 0.0, 'MAVUSDT': 0.0, 'MDTUSDT': 0.0, 'XVGUSDT': 0.0, 'WLDUSDT': 0.0, 'PENDLEUSDT': 0.0, 'ARKMUSDT': 0.0, 'AGLDUSDT': 0.0, 'YGGUSDT': 0.0, 'DODOXUSDT': 0.0, 'BNTUSDT': 0.0, 'OXTUSDT': 0.0, 'SEIUSDT': 0.0, 'CYBERUSDT': 0.0, 'HIFIUSDT': 0.0, 'ARKUSDT': 0.0, 'FRONTUSDT': 0.0, 'GLMRUSDT': 0.0, 'BICOUSDT': 0.0, 'STRAXUSDT': 0.0, 'LOOMUSDT': 0.0, 'BIGTIMEUSDT': 0.0, 'BONDUSDT': 0.0, 'ORBSUSDT': 0.0, 'STPTUSDT': 0.0, 'WAXPUSDT': 0.0, 'BSVUSDT': 0.0, 'RIFUSDT': 0.0, 'POLYXUSDT': 0.0, 'GASUSDT': 0.0, 'POWRUSDT': 0.0, 'SLPUSDT': 0.0, 'TIAUSDT': 0.0, 'SNTUSDT': 0.0, 'CAKEUSDT': 0.0, 'MEMEUSDT': 0.0, 'TWTUSDT': 0.0, 'TOKENUSDT': 0.0, 'ORDIUSDT': 0.0, 'STEEMUSDT': 0.0, 'BADGERUSDT': 0.0, 'ILVUSDT': 0.0, 'NTRNUSDT': 0.0, 'KASUSDT': 0.0, 'BEAMXUSDT': 0.0, '1000BONKUSDT': 0.0, 'PYTHUSDT': 0.0, 'SUPERUSDT': 0.0, 'USTCUSDT': 0.0, 'ONGUSDT': 0.0, 'ETHWUSDT': 0.0, 'JTOUSDT': 0.0, '1000SATSUSDT': 0.0, 'AUCTIONUSDT': 0.0, '1000RATSUSDT': 0.0, 'ACEUSDT': 0.0, 'MOVRUSDT': 0.0, 'NFPUSDT': 0.0, 'AIUSDT': 0.0, 'XAIUSDT': 0.0, 'WIFUSDT': 0.0, 'MANTAUSDT': 0.0, 'ONDOUSDT': 0.0, 'LSKUSDT': 0.0, 'ALTUSDT': 0.0, 'JUPUSDT': 0.0, 'ZETAUSDT': 0.0, 'RONINUSDT': 0.0, 'DYMUSDT': 0.0, 'OMUSDT': 0.0, 'PIXELUSDT': 0.0, 'STRKUSDT': 0.0, 'MAVIAUSDT': 0.0, 'GLMUSDT': 0.0, 'PORTALUSDT': 0.0, 'TONUSDT': 0.0, 'AXLUSDT': 0.0, 'MYROUSDT': 0.0, 'METISUSDT': 0.0, 'AEVOUSDT': 0.0, 'VANRYUSDT': 0.0, 'BOMEUSDT': 0.0, 'ETHFIUSDT': 0.0, 'ENAUSDT': 0.0, 'WUSDT': 0.0, 'TNSRUSDT': 0.0, 'SAGAUSDT': 0.0, 'TAOUSDT': 0.0, 'OMNIUSDT': 0.0, 'REZUSDT': 0.0, 'BBUSDT': 0.0, 'NOTUSDT': 0.0, 'TURBOUSDT': 0.0, 'IOUSDT': 0.0}


  # 更改持仓模式为单向
  modify_position_side(client, logger_error)
  # 调整开仓杠杆到1
  for key, value in position_value_map.items():
    change_leverage(client, key, logger_error)

  # 重试次数
  retry_count = 30

  while(True):
    try:
      # 交易列表
      symbol_set = set()

      # 成交率的分子和分母
      deal_numerator = 0.0
      deal_denominator = 0.0

      # 等待并读取交易信号
      file_path = f'./signal/pred30top10/{signal_index}{signal_suffix}'
      logger_trade.info("------寻找信号文件: " + str(file_path) + " ------")
      while not os.path.exists(file_path):
        time.sleep(2)

      alpha_map = {}
      logger_trade.info("------开始处理信号文件: " + str(file_path) + " ------")
      with open(file_path, mode='r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # 跳过表头行
        for row in csv_reader:
          key = row[0]
          value = Decimal(row[1])/Decimal('5')
          value = value.quantize(Decimal('0.00'), rounding=ROUND_HALF_UP)
          alpha_map[key] = value

      logger_trade.info(account_balance(client, logger_error))
      # 时间窗口内重试多次(window/period->100s/3s)
      for i in range(retry_count):
        # 更新最新持仓数据(返回True表示失败)
        if (update_positions(client, position_value_map, position_amount_map, logger_error)):
          continue
        # 日志需要:打印出目前不为0的持仓金额和数量
        filtered_values = {k: v for k, v in position_value_map.items() if v != 0.0}
        filtered_amounts = {k: v for k, v in position_amount_map.items() if v != 0.0}
        logger_trade.info("Positions value: " + str(filtered_values))
        logger_trade.info("Positions amount: " + str(filtered_amounts))

        # 根据交易信号进行maker交易(差值)
        logger_trade.info("------开始根据Alpha信号差值交易: retry" + str(i+1)  +  "------")

        # 对要遍历的仓位map做随机化处理,保证处理顺序公平
        random_position_value_items = list(position_value_map.items())
        random.shuffle(random_position_value_items)

        for key, value in random_position_value_items:
          if key in alpha_map: # 有权重(需要交易,包含0)
            # 计算目前持仓和信号权重的差值delta
            alpha_value = Decimal(alpha_map[key])
            position_value = Decimal(position_value_map[key])
            delta = alpha_value - position_value

            positive_threshold = get_threshold(key)
            negative_threshold = positive_threshold * Decimal("-1")
            # logger_trade.info("------positive_threshold:" + str(positive_threshold) + "negative_threshold:"+str(negative_threshold))
            first_divide_delta = Decimal('0.0')
            sch=3
            first_divide_delta = delta * Decimal(min(max(i-sch,0),8) / 8)
            first_divide_delta=first_divide_delta if abs(first_divide_delta) > 2*positive_threshold else Decimal("0")
            second_divide_delta = delta - first_divide_delta
            second_divide_delta=second_divide_delta if abs(second_divide_delta) > 2*positive_threshold else Decimal("0")

            if (delta > positive_threshold): #买入
                logger_trade.info("------" + "开始处理: " + str(key) + "------")
                logger_trade.info("Delta为: " + str(delta)+" fd:"+str(first_divide_delta)+" sd:"+str(second_divide_delta) +" a0:"+str(alpha_value==0))
                symbol_set.add(key)
                if (i == 0):
                  deal_denominator = deal_denominator + 1.0
                if (i == (retry_count-1)):
                  deal_numerator = deal_numerator + 1.0
                trade_tag = True
                #while(trade_tag):
                bid0,bid1 = query_bids(client, key, price_precision_map[key], logger_error)
                logger_trade.info("bid0: " + str(bid0)+" bid1:"+str(bid1))
                if (bid0 > 0) and (bid1 > 0):
                  # bid1 = bid0 * Decimal('0.9998')
                  # bid1 = bid1.quantize(Decimal(f'1.{"0" * price_precision_map[key]}'))
                  # logger_trade.info("nbid0: " + str(bid0)+" nbid1:"+str(bid1))
                  # 平仓单
                  if (alpha_value == 0 and delta < 3*positive_threshold):
                    trade_amount = abs(Decimal(position_amount_map[key]))
                    buy(client, trade_amount, key, bid0, logger_trade, logger_error)
                    continue
                  # 下2个单: 第一个单(i+1/8)
                  if first_divide_delta > 0:
                    trade_amount = first_divide_delta / Decimal(bid0)
                    trade_amount = trade_amount.quantize(Decimal(f'1.{"0" * quantity_precision_map[key]}'))
                    buy(client, trade_amount, key, bid0, logger_trade, logger_error)
                  # 下2个单: 第二个单1-(i+1/8)
                  if (second_divide_delta > 0):
                    trade_amount = second_divide_delta / Decimal(bid1)
                    trade_amount = trade_amount.quantize(Decimal(f'1.{"0" * quantity_precision_map[key]}'))
                    buy(client, trade_amount, key, bid1, logger_trade, logger_error)
            elif (delta < negative_threshold): #卖出
                logger_trade.info("------" + "开始处理: " + str(key) + "------")
                logger_trade.info("Delta为: " + str(delta)+" fd:"+str(first_divide_delta)+" sd:"+str(second_divide_delta) +" a0:"+str(alpha_value==0))
                symbol_set.add(key)
                if (i == 0):
                  deal_denominator = deal_denominator + 1.0
                if (i == (retry_count-1)):
                  deal_numerator = deal_numerator + 1.0
                trade_tag = True
                #while(trade_tag):
                ask0,ask1 = query_asks(client, key, price_precision_map[key], logger_error)
                logger_trade.info("ask0: " + str(ask0)+" ask1:"+str(ask1))
                if (ask0 > 0) and (ask1 > 0):
                  # ask1 = ask0 * Decimal('1.0002')
                  # ask1 = ask1.quantize(Decimal(f'1.{"0" * price_precision_map[key]}'))
                  # logger_trade.info("nask0: " + str(ask0)+" nask1:"+str(ask1))
                  # 平仓单
                  if (alpha_value == 0 and delta > 3*negative_threshold):
                    trade_amount = abs(Decimal(position_amount_map[key]))
                    sell(client, trade_amount, key, ask0, logger_trade, logger_error)
                    continue
                  # 下2个单: 第一个单(i+1/8)
                  if first_divide_delta < 0:
                    trade_amount = first_divide_delta / Decimal(ask0)
                    trade_amount = trade_amount.quantize(Decimal(f'1.{"0" * quantity_precision_map[key]}'))
                    trade_amount = abs(trade_amount)
                    sell(client, trade_amount, key, ask0, logger_trade, logger_error)
                  # 下2个单: 第二个单1-(i+1/8)
                  if (second_divide_delta < 0):
                    trade_amount = second_divide_delta / Decimal(ask1)
                    trade_amount = trade_amount.quantize(Decimal(f'1.{"0" * quantity_precision_map[key]}'))
                    trade_amount = abs(trade_amount)
                    sell(client, trade_amount, key, ask1, logger_trade, logger_error)

        # 等待3S轮询
        time.sleep(10)

        # 取消掉所有Open状态的订单
        logger_trade.info("------取消所有Open状态的订单------")
        for key in symbol_set:
          cancel_status = True
          while(cancel_status):
            cancel_status = cancel_open_orders(client, key, logger_trade, logger_error)

      # 输出交易成交率(参考用)
      if (deal_denominator > 0) and (deal_numerator > 0):
        deal_ratio = round(1 - (deal_numerator/deal_denominator), 2)
        logger_trade.info("交易成交率(最后一轮):" + str(deal_ratio))
      # 更新信号文件下标(等待下一个信号文件)
      signal_time = datetime.strptime(signal_index, "%Y%m%d%H%M%S")
      signal_time = signal_time + time_delta
      signal_index = signal_time.strftime("%Y%m%d%H%M00")
      # 当前账户余额
      logger_trade.info(account_balance(client, logger_error))
    except Exception as e:
      logger_error.info(str(e))
      # 异常通知到钉钉
      notify(str(e))
      exit(1)

main()
