import json
import time
import threading
import pandas as pd
from datetime import datetime, timedelta, timezone
from multiprocessing import Manager, Queue, Process, Lock
import logging
from binance.websocket.um_futures.websocket_client import UMFuturesWebsocketClient
import signal
import sys
from functools import partial

def get_logger(logger_name, log_file, level=logging.INFO):
    logger = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s', "%Y-%m-%d %H:%M:%S")
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setFormatter(formatter)

    logger.setLevel(level)
    logger.addHandler(file_handler)
    return logger

manager = Manager()
data_dict = manager.dict()
latest_t_dict = manager.dict()
available_t_dict = manager.dict()
#data_lock = Lock()
message_queues = [Queue() for _ in range(69)]
clients = [] #TODO thread safe

my_proxies = {
    'http': 'http://127.0.0.1:20890',
    'https': 'http://127.0.0.1:20890'
  }
my_timeout=60

def generate_key_new(data):
    return f"{data['price']}-{data['time']}-{data['isBuyerMaker']}"
def generate_key_old(data):
    return f"{data['p']}-{data['T']}-{data['m']}"
def remove_old_data(symbol, current_time_ms):
    try:
        #with data_lock:
        if symbol in data_dict:
            # 计算5分钟前的时间戳
            cutoff_time_ms = current_time_ms - 5 * 60 * 1000
            if symbol == 'WUSDT':
                print(f"Before Length of data_dict[{symbol}]: {len(data_dict[symbol])}")

            new_data = [item for item in data_dict[symbol] if item['T'] > cutoff_time_ms]
            data_dict[symbol] = manager.list(new_data)

            if symbol == 'WUSDT':
                print(f"After Length of data_dict[{symbol}]: {len(data_dict[symbol])}")
    except Exception as e:
        print(f"Error in remove_old_data: {str(e)}")

def fetch_and_add_data(symbol):
    if symbol not in latest_t_dict:
        return
    um_futures_client = UMFutures(key='OugVodq3VF9syMAWI9iOGBwaIb4G0Pv3xVXSUOto24Oc4vImXaZBcpOyqL4uwkxF',
                                  secret='tNGt7sG9kMatPYqn1TcX8fcbds4jdgN21TY8TKD60rZRjuKxy2W6yZCLMuOCSyKw',
                                  proxies=my_proxies, timeout=my_timeout)
    try:
        start_time = latest_t_dict[symbol]
        end_time = int(datetime.now(timezone.utc).timestamp() * 1000)
        fetch_data = um_futures_client.agg_trades(symbol=symbol, startTime=start_time, endTime=end_time)
        if symbol == 'WUSDT':
            print(f"Fetched data for {symbol}: {fetch_data}")
        #with data_lock:
        if symbol not in data_dict:
            data_dict[symbol] = manager.list()

        for data in fetch_data:
            #data_frame = pd.DataFrame([data])
            new_key = generate_key_new(data)
            if new_key not in (generate_key_old(item) for item in data_dict[symbol]):
                data_dict[symbol].append(data)

        if symbol == 'WUSDT':
            print(f"Data added for {symbol}. Current length: {len(data_dict[symbol])}")
    except Exception as e:
        print(f"Error fetching data for {symbol}: {str(e)}")
    finally:
        um_futures_client.close()

def process_message_queue(symbols, index):
    while True:
        symbol, data = message_queues[index].get()
        if symbol not in symbols:
            print("CAO!!!!!!!!!!!!")
            continue
        try:
            #with data_lock:
            if symbol not in data_dict:
                data_dict[symbol] = manager.list()

            #data_frame = pd.DataFrame([data])
            data_dict[symbol].append(data)
            if symbol not in latest_t_dict or data['T'] > latest_t_dict[symbol]:
                latest_t_dict[symbol] = data['T']

            # 获取当前时间戳(UTC标准时间)的毫秒时间戳
            current_time_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
            remove_old_data(symbol, current_time_ms)

            if symbol == 'WUSDT':
                print(f"Processed data for {symbol}: {data} : {current_time_ms}")
                print(f"Delta is {current_time_ms - int(data['T'])}")
                print(f"Length of data_dict[{symbol}]: {len(data_dict[symbol])}")
        except Exception as e:
            print(f"Error in process_message_queue: {str(e)}")

def message_handler(index, _, message):
    try:
        msg = json.loads(message)
        # 过滤掉包含 "result": null 的消息
        if "result" in msg and msg["result"] is None and "id" in msg:
            return
        # 检查消息中是否包含 'stream' 和 'data' 字段
        if 'stream' not in msg or 'data' not in msg:
            return
        stream = msg['stream']
        data = msg['data']
        symbol = stream.split('@')[0].upper()
        # 放入Queue(线程安全)
        message_queues[index].put((symbol, data))

    except Exception as e:
        print(f"Error in message_handler with lock: {str(e)}")
def on_error(error, symbols=None):
    print(f"WebSocket encountered an error: {error}")

def on_close(*args, symbols=None, index=0):
    print("WebSocket closed. Reconnecting...")
    create_websocket_client(symbols, index)
def create_websocket_client(symbols, index):
    while True:
        print("Start create websocket client.")
        client = UMFuturesWebsocketClient(
            on_message=partial(message_handler, index),
            on_close=partial(on_close, symbols=symbols, index=index),
            on_error=partial(on_error, symbols=symbols),
            is_combined=True,
            proxies=my_proxies
        )
        clients.append(client)  # 将客户端添加到全局列表中
        try:
            for symbol in symbols:
                client.agg_trade(symbol=symbol)
            for symbol in symbols:
                fetch_and_add_data(symbol)
            while True:
                time.sleep(1)  # 保持程序运行并允许捕获终止信号
                current_time_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
                for symbol in symbols:
                    available_t_dict[symbol] = current_time_ms - 1000

        except Exception as e:
            print(f"WebSocket connection lost for symbols: {symbols}. Reconnecting...")
            print(f"Error: {str(e)}")
        finally:
            print(f"Stop client for: {symbols}")
            client.stop()

def signal_handler(sig, frame):
    print("Exiting program... Closing all WebSocket clients.")
    for client in clients:
        client.stop()
    sys.exit(0)

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logfile_info = "/tmp/aggtrades-info.log"
    logfile_error = "/tmp/aggtrades-error.log"
    logger_info = get_logger("logger_info", logfile_info)
    logger_error = get_logger("logger_error", logfile_error)

    available_tokens = ['BTCUSDT', 'ETHUSDT', 'BCHUSDT', 'XRPUSDT', 'EOSUSDT', 'LTCUSDT', 'TRXUSDT', 'ETCUSDT',
                        'LINKUSDT',
                        'XLMUSDT', 'ADAUSDT', 'XMRUSDT', 'DASHUSDT', 'ZECUSDT', 'XTZUSDT', 'ATOMUSDT', 'ONTUSDT',
                        'IOTAUSDT', 'BATUSDT', 'VETUSDT', 'NEOUSDT', 'QTUMUSDT', 'IOSTUSDT', 'THETAUSDT', 'ALGOUSDT',
                        'ZILUSDT', 'KNCUSDT', 'ZRXUSDT', 'COMPUSDT', 'OMGUSDT', 'DOGEUSDT', 'SXPUSDT', 'KAVAUSDT',
                        'BANDUSDT', 'RLCUSDT', 'WAVESUSDT', 'MKRUSDT', 'SNXUSDT', 'DOTUSDT', 'DEFIUSDT', 'YFIUSDT',
                        'BALUSDT', 'CRVUSDT', 'TRBUSDT', 'RUNEUSDT', 'SUSHIUSDT', 'EGLDUSDT', 'SOLUSDT', 'ICXUSDT',
                        'STORJUSDT', 'BLZUSDT', 'UNIUSDT', 'AVAXUSDT', 'FTMUSDT', 'ENJUSDT', 'FLMUSDT', 'RENUSDT',
                        'KSMUSDT', 'NEARUSDT', 'AAVEUSDT', 'FILUSDT', 'RSRUSDT', 'LRCUSDT', 'MATICUSDT', 'OCEANUSDT',
                        'CVCUSDT', 'BELUSDT', 'CTKUSDT', 'AXSUSDT', 'ALPHAUSDT', 'ZENUSDT', 'SKLUSDT', 'GRTUSDT',
                        '1INCHUSDT', 'CHZUSDT', 'SANDUSDT', 'ANKRUSDT', 'LITUSDT', 'UNFIUSDT', 'REEFUSDT', 'RVNUSDT',
                        'SFPUSDT', 'XEMUSDT', 'BTCSTUSDT', 'COTIUSDT', 'CHRUSDT', 'MANAUSDT', 'ALICEUSDT', 'HBARUSDT',
                        'ONEUSDT', 'LINAUSDT', 'STMXUSDT', 'DENTUSDT', 'CELRUSDT', 'HOTUSDT', 'MTLUSDT', 'OGNUSDT',
                        'NKNUSDT', 'SCUSDT', 'DGBUSDT', '1000SHIBUSDT', 'BAKEUSDT', 'GTCUSDT', 'BTCDOMUSDT', 'IOTXUSDT',
                        'RAYUSDT', 'C98USDT', 'MASKUSDT', 'ATAUSDT', 'DYDXUSDT', '1000XECUSDT', 'GALAUSDT', 'CELOUSDT',
                        'ARUSDT', 'KLAYUSDT', 'ARPAUSDT', 'CTSIUSDT', 'LPTUSDT', 'ENSUSDT', 'PEOPLEUSDT', 'ROSEUSDT',
                        'DUSKUSDT', 'FLOWUSDT', 'IMXUSDT', 'API3USDT', 'GMTUSDT', 'APEUSDT', 'WOOUSDT', 'FTTUSDT',
                        'JASMYUSDT', 'DARUSDT', 'GALUSDT', 'OPUSDT', 'INJUSDT', 'STGUSDT', 'SPELLUSDT', '1000LUNCUSDT',
                        'LUNA2USDT', 'LDOUSDT', 'CVXUSDT', 'ICPUSDT', 'APTUSDT', 'QNTUSDT', 'FETUSDT', 'FXSUSDT',
                        'HOOKUSDT', 'MAGICUSDT', 'TUSDT', 'RNDRUSDT', 'HIGHUSDT', 'MINAUSDT', 'ASTRUSDT', 'AGIXUSDT',
                        'PHBUSDT', 'GMXUSDT', 'CFXUSDT', 'STXUSDT', 'BNXUSDT', 'ACHUSDT', 'SSVUSDT', 'CKBUSDT',
                        'PERPUSDT',
                        'TRUUSDT', 'LQTYUSDT', 'USDCUSDT', 'IDUSDT', 'ARBUSDT', 'JOEUSDT', 'TLMUSDT', 'AMBUSDT',
                        'LEVERUSDT', 'RDNTUSDT', 'HFTUSDT', 'XVSUSDT', 'BLURUSDT', 'EDUUSDT', 'IDEXUSDT', 'SUIUSDT',
                        '1000PEPEUSDT', '1000FLOKIUSDT', 'UMAUSDT', 'RADUSDT', 'KEYUSDT', 'COMBOUSDT', 'NMRUSDT',
                        'MAVUSDT',
                        'MDTUSDT', 'XVGUSDT', 'WLDUSDT', 'PENDLEUSDT', 'ARKMUSDT', 'AGLDUSDT', 'YGGUSDT', 'DODOXUSDT',
                        'BNTUSDT', 'OXTUSDT', 'SEIUSDT', 'CYBERUSDT', 'HIFIUSDT', 'ARKUSDT', 'FRONTUSDT', 'GLMRUSDT',
                        'BICOUSDT', 'STRAXUSDT', 'LOOMUSDT', 'BIGTIMEUSDT', 'BONDUSDT', 'ORBSUSDT', 'STPTUSDT',
                        'WAXPUSDT',
                        'BSVUSDT', 'RIFUSDT', 'POLYXUSDT', 'GASUSDT', 'POWRUSDT', 'SLPUSDT', 'TIAUSDT', 'SNTUSDT',
                        'CAKEUSDT', 'MEMEUSDT', 'TWTUSDT', 'TOKENUSDT', 'ORDIUSDT', 'STEEMUSDT', 'BADGERUSDT',
                        'ILVUSDT',
                        'NTRNUSDT', 'KASUSDT', 'BEAMXUSDT', '1000BONKUSDT', 'PYTHUSDT', 'SUPERUSDT', 'USTCUSDT',
                        'ONGUSDT',
                        'ETHWUSDT', 'JTOUSDT', '1000SATSUSDT', 'AUCTIONUSDT', '1000RATSUSDT', 'ACEUSDT', 'MOVRUSDT',
                        'NFPUSDT', 'AIUSDT', 'XAIUSDT', 'WIFUSDT', 'MANTAUSDT', 'ONDOUSDT', 'LSKUSDT', 'ALTUSDT',
                        'JUPUSDT',
                        'ZETAUSDT', 'RONINUSDT', 'DYMUSDT', 'OMUSDT', 'PIXELUSDT', 'STRKUSDT', 'MAVIAUSDT', 'GLMUSDT',
                        'PORTALUSDT', 'TONUSDT', 'AXLUSDT', 'MYROUSDT', 'METISUSDT', 'AEVOUSDT', 'VANRYUSDT',
                        'BOMEUSDT',
                        'ETHFIUSDT', 'ENAUSDT', 'WUSDT', 'TNSRUSDT', 'SAGAUSDT', 'TAOUSDT', 'OMNIUSDT', 'REZUSDT',
                        'BBUSDT',
                        'NOTUSDT', 'TURBOUSDT', 'IOUSDT']
    #available_tokens = ['BTCUSDT', 'ETHUSDT', 'WUSDT']

    # 创建进程池处理消息队列中的数据
    process_pool_size = 69
    processes = []
    batch_size = len(available_tokens) // process_pool_size

    for i in range(process_pool_size):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        if i == process_pool_size - 1:
            end_idx = len(available_tokens)
        batch_symbols = available_tokens[start_idx:end_idx]
        p = Process(target=process_message_queue, args=(batch_symbols, i))
        p.start()
        processes.append(p)

    # 每个WebSocket连接订阅3个symbol
    threads = []
    for i in range(process_pool_size):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        if i == process_pool_size - 1:
            end_idx = len(available_tokens)
        batch_symbols = available_tokens[start_idx:end_idx]
        t = threading.Thread(target=create_websocket_client, args=(batch_symbols, i))
        t.start()
        threads.append(t)

    # 等待所有线程完成
    for t in threads:
        t.join()

    # 等待所有进程完成
    for p in processes:
        p.join()