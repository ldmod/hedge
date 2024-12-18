import logging
import datetime
import pytz
from binance.um_futures import UMFutures

# 日志初始化
def get_logger(logger_name,log_file,level=logging.INFO):
  logger = logging.getLogger(logger_name)
  formatter = logging.Formatter('%(asctime)s : %(message)s', "%Y-%m-%d %H:%M:%S")
  fileHandler = logging.FileHandler(log_file, mode='a')
  fileHandler.setFormatter(formatter)

  logger.setLevel(level)
  logger.addHandler(fileHandler)
  return logger

# 当前账户余额查询
def account_balance(client, logger_error):
    try:
      return client.balance()
    except Exception as e:
      logger_error.info(str(e)+"  ")


def convert_timestamp_to_string(timestamp_ms):
  # 将毫秒时间戳转换为秒时间戳
  timestamp_s = timestamp_ms / 1000
  # 创建datetime对象
  dt_object = datetime.datetime.fromtimestamp(timestamp_s, pytz.UTC)
  # 格式化为需要的字符串格式
  formatted_string = dt_object.strftime('%Y-%m-%d %H:%M UTC')
  return formatted_string


# 冲击成本计算
if __name__ == '__main__':
  # 可交易Token
  token_list = [
    'BTCUSDT', 'ETHUSDT', 'BCHUSDT', 'XRPUSDT', 'EOSUSDT', 'LTCUSDT', 'TRXUSDT', 'ETCUSDT', 'LINKUSDT', 'XLMUSDT',
    'ADAUSDT', 'XMRUSDT', 'DASHUSDT', 'ZECUSDT', 'XTZUSDT', 'BNBUSDT', 'ATOMUSDT', 'ONTUSDT', 'IOTAUSDT', 'BATUSDT',
    'VETUSDT', 'NEOUSDT', 'QTUMUSDT', 'IOSTUSDT', 'THETAUSDT', 'ALGOUSDT', 'ZILUSDT', 'KNCUSDT', 'ZRXUSDT', 'COMPUSDT',
    'OMGUSDT', 'DOGEUSDT', 'SXPUSDT', 'KAVAUSDT', 'BANDUSDT', 'RLCUSDT', 'WAVESUSDT', 'MKRUSDT', 'SNXUSDT', 'DOTUSDT',
    'DEFIUSDT', 'YFIUSDT', 'BALUSDT', 'CRVUSDT', 'TRBUSDT', 'RUNEUSDT', 'SUSHIUSDT', 'EGLDUSDT', 'SOLUSDT', 'ICXUSDT',
    'STORJUSDT', 'BLZUSDT', 'UNIUSDT', 'AVAXUSDT', 'FTMUSDT', 'ENJUSDT', 'FLMUSDT', 'RENUSDT', 'KSMUSDT', 'NEARUSDT',
    'AAVEUSDT', 'FILUSDT', 'RSRUSDT', 'LRCUSDT', 'MATICUSDT', 'OCEANUSDT', 'CVCUSDT', 'BELUSDT', 'CTKUSDT', 'AXSUSDT',
    'ALPHAUSDT', 'ZENUSDT', 'SKLUSDT', 'GRTUSDT', '1INCHUSDT', 'CHZUSDT', 'SANDUSDT', 'ANKRUSDT', 'LITUSDT', 'UNFIUSDT',
    'REEFUSDT', 'RVNUSDT', 'SFPUSDT', 'XEMUSDT', 'BTCSTUSDT', 'COTIUSDT', 'CHRUSDT', 'MANAUSDT', 'ALICEUSDT',
    'HBARUSDT',
    'ONEUSDT', 'LINAUSDT', 'STMXUSDT', 'DENTUSDT', 'CELRUSDT', 'HOTUSDT', 'MTLUSDT', 'OGNUSDT', 'NKNUSDT', 'SCUSDT',
    'DGBUSDT', '1000SHIBUSDT', 'BAKEUSDT', 'GTCUSDT', 'BTCDOMUSDT', 'IOTXUSDT', 'RAYUSDT', 'C98USDT', 'MASKUSDT',
    'ATAUSDT',
    'DYDXUSDT', '1000XECUSDT', 'GALAUSDT', 'CELOUSDT', 'ARUSDT', 'KLAYUSDT', 'ARPAUSDT', 'CTSIUSDT', 'LPTUSDT',
    'ENSUSDT',
    'PEOPLEUSDT', 'ROSEUSDT', 'DUSKUSDT', 'FLOWUSDT', 'IMXUSDT', 'API3USDT', 'GMTUSDT', 'APEUSDT', 'WOOUSDT', 'FTTUSDT',
    'JASMYUSDT', 'DARUSDT', 'GALUSDT', 'OPUSDT', 'INJUSDT', 'STGUSDT', 'SPELLUSDT', '1000LUNCUSDT', 'LUNA2USDT',
    'LDOUSDT',
    'CVXUSDT', 'ICPUSDT', 'APTUSDT', 'QNTUSDT', 'FETUSDT', 'FXSUSDT', 'HOOKUSDT', 'MAGICUSDT', 'TUSDT', 'RNDRUSDT',
    'HIGHUSDT', 'MINAUSDT', 'ASTRUSDT', 'AGIXUSDT', 'PHBUSDT', 'GMXUSDT', 'CFXUSDT', 'STXUSDT', 'BNXUSDT', 'ACHUSDT',
    'SSVUSDT', 'CKBUSDT', 'PERPUSDT', 'TRUUSDT', 'LQTYUSDT', 'USDCUSDT', 'IDUSDT', 'ARBUSDT', 'JOEUSDT', 'TLMUSDT',
    'AMBUSDT', 'LEVERUSDT', 'RDNTUSDT', 'HFTUSDT', 'XVSUSDT', 'BLURUSDT', 'EDUUSDT', 'IDEXUSDT', 'SUIUSDT',
    '1000PEPEUSDT',
    '1000FLOKIUSDT', 'UMAUSDT', 'RADUSDT', 'KEYUSDT', 'COMBOUSDT', 'NMRUSDT', 'MAVUSDT', 'MDTUSDT', 'XVGUSDT',
    'WLDUSDT',
    'PENDLEUSDT', 'ARKMUSDT', 'AGLDUSDT', 'YGGUSDT', 'DODOXUSDT', 'BNTUSDT', 'OXTUSDT', 'SEIUSDT', 'CYBERUSDT',
    'HIFIUSDT',
    'ARKUSDT', 'FRONTUSDT', 'GLMRUSDT', 'BICOUSDT', 'STRAXUSDT', 'LOOMUSDT', 'BIGTIMEUSDT', 'BONDUSDT', 'ORBSUSDT',
    'STPTUSDT',
    'WAXPUSDT', 'BSVUSDT', 'RIFUSDT', 'POLYXUSDT', 'GASUSDT', 'POWRUSDT', 'SLPUSDT', 'TIAUSDT', 'SNTUSDT', 'CAKEUSDT',
    'MEMEUSDT', 'TWTUSDT', 'TOKENUSDT', 'ORDIUSDT', 'STEEMUSDT', 'BADGERUSDT', 'ILVUSDT', 'NTRNUSDT', 'KASUSDT',
    'BEAMXUSDT',
    '1000BONKUSDT', 'PYTHUSDT', 'SUPERUSDT', 'USTCUSDT', 'ONGUSDT', 'ETHWUSDT', 'JTOUSDT', '1000SATSUSDT',
    'AUCTIONUSDT',
    '1000RATSUSDT', 'ACEUSDT', 'MOVRUSDT', 'NFPUSDT', 'AIUSDT', 'XAIUSDT', 'WIFUSDT', 'MANTAUSDT', 'ONDOUSDT',
    'LSKUSDT',
    'ALTUSDT', 'JUPUSDT', 'ZETAUSDT', 'RONINUSDT', 'DYMUSDT', 'OMUSDT', 'PIXELUSDT', 'STRKUSDT', 'MAVIAUSDT', 'GLMUSDT',
    'PORTALUSDT', 'TONUSDT', 'AXLUSDT', 'MYROUSDT', 'METISUSDT', 'AEVOUSDT', 'VANRYUSDT', 'BOMEUSDT', 'ETHFIUSDT',
    'ENAUSDT',
    'WUSDT', 'TNSRUSDT', 'SAGAUSDT', 'TAOUSDT', 'OMNIUSDT', 'REZUSDT', 'BBUSDT', 'NOTUSDT', 'TURBOUSDT', 'IOUSDT'
  ]
  # 初始化只读客户端
  client = UMFutures(key='OugVodq3VF9syMAWI9iOGBwaIb4G0Pv3xVXSUOto24Oc4vImXaZBcpOyqL4uwkxF', secret='tNGt7sG9kMatPYqn1TcX8fcbds4jdgN21TY8TKD60rZRjuKxy2W6yZCLMuOCSyKw')

  # 初始化日志文件
  logfile_info = "/tmp/orders-info.md"
  logfile_error = "/tmp/orders-error.md"
  logger_info = get_logger("logger_info", logfile_info)
  logger_error = get_logger("logger_error", logfile_error)

  # 变量定义
  total_price_qty = 0
  total_qty = 0
  max_order_time = 0
  total_delta = 0
  total_money = 0
  signal_date = 0
  symbol_set = set()
  buyer = False
  win = False

  # 控制参数
  lookback_mins = 48 #长点好
  lookback_count = 0
  vwap_window = 5

  # 选出上一次信号有交易的token
  server_time_resp = client.time()
  server_time = int(server_time_resp['serverTime'])
  server_time = server_time - lookback_count * (1000*60*30)

  print("name, own, all, buyer, win, delta")
  for token in token_list:
    # 变量清空
    max_order_time = 0
    total_price_qty = 0
    total_qty = 0
    win = False
    buyer = None
    order_data = client.get_account_trades(symbol=token, startTime=server_time-1000*60*lookback_mins, endTime=server_time)
    if order_data:
      for order in order_data:
        if (buyer is None):
          buyer = order['buyer']
        if (buyer == order['buyer']):
          price = float(order['price'])
          qty = float(order['qty'])
          total_price_qty += price * qty
          total_qty += qty
          if order['time'] > max_order_time:
            max_order_time = order['time']
      average_price_own = total_price_qty / total_qty if total_qty != 0 else 0

      # 所有成交的订单 (10表示看前十分钟)
      total_qty_own = total_qty
      total_price_qty = 0
      total_qty = 0
      order_data = client.agg_trades(symbol=token, startTime=max_order_time-1000*60*vwap_window, endTime=max_order_time+1000)
      for order in order_data:
        price = float(order['p'])
        qty = float(order['q'])
        total_price_qty += price * qty
        total_qty += qty
      average_price_all = total_price_qty / total_qty if total_qty != 0 else 0
      delta = -abs(round((average_price_own - average_price_all) * total_qty_own, 4))
      if (buyer and (average_price_own < average_price_all)):
        win = True
        delta = -delta
      if ((not buyer) and (average_price_own > average_price_all)):
        win = True
        delta = -delta
      total_delta = total_delta + delta
      total_money = total_money + round(average_price_own * total_qty_own, 4)
      signal_date = max_order_time
      print(str(token)+" , "+str(average_price_own)+" , "+str(average_price_all)+" , "+str(buyer)+" , "+str(win)+" , "+str(delta))
print("Total Delta: " + str(total_delta))
print("Total Money: " + str(total_money))
print("BP: " + str(total_delta/total_money*10000))
print("Data: " + convert_timestamp_to_string(signal_date))