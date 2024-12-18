import sys
sys.path.append('/home/crypto/sec_klines_env/')
import json
from datetime import datetime, timezone
import threading
import time
import pandas as pd
from binance.um_futures import UMFutures
from multiprocessing import Manager, Process, Queue
from binance.websocket.um_futures.websocket_client import UMFuturesWebsocketClient
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
import cryptoqt.data.updatedata as ud
import cryptoqt.data.tools as tools
import copy
import random
import cryptoqt.data.sec_klines.secklines as skl
import numpy as np
import yaml
import os
#Websocket推送的数据结构:
# {
#   "e": "aggTrade",  // 事件类型
#   "E": 123456789,   // 事件时间
#   "s": "BNBUSDT",    // 交易对
#   "a": 5933014,     // 归集成交 ID
#   "p": "0.001",     // 成交价格
#   "q": "100",       // 成交量
#   "f": 100,         // 被归集的首个交易ID
#   "l": 105,         // 被归集的末次交易ID
#   "T": 123456785,   // 成交时间
#   "m": true         // 买方是否是做市方。如true，则此次成交是一个主动卖出单，否则是一个主动买入单。
# }

#我们的数据结构DataFrame
# 事件时间 | 归集成交ID | 成交价格 | 成交量 | 被归集的首个交易ID | 被归集的末次交易ID | 成交时间 | 买方是否是做市方
# event_time | agg_order_id | price | quantity | first_order_id | last_order_id | time | buy_maker

manager = Manager()
data_dict = manager.dict()
latest_t_dict = manager.dict()
available_t_dict = manager.dict()
webstart_t_dict = manager.dict()
g_log_ratio=0.002

# my_proxies = {
#     'http': 'http://127.0.0.1:33880',
#     'https': 'http://127.0.0.1:33880'
# }
my_proxies = {
    'http': 'http://127.0.0.1:19890',
    'https': 'http://127.0.0.1:19890'
}

def message_handler(_, queue, message):
    try:
        msg = json.loads(message)
        if "result" in msg and msg["result"] is None and "id" in msg:
            return
        if 'stream' not in msg or 'data' not in msg:
            return
        stream = msg['stream']
        data = msg['data']
        symbol = stream.split('@')[0].upper()
        queue.put((symbol, data))
        if random.random() < g_log_ratio/2:
            current_time_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
            print(f"Network Cost {symbol}: {current_time_ms - int(data['E'])} ms", flush=True)

    except Exception as e:
        print(f"Error in message_handler: {str(e)}", flush=True)

def get_available_data_time():
    min_time = float('inf')
    for symbol, time in available_t_dict.items():
        if time < min_time:
            min_time = time
    return min_time

# 记住修改: /home/jywang/miniconda3/lib/python3.11/site-packages/binance/websocket/binance_socket_manager.py
#  except WebSocketException as e:
#     if isinstance(e, WebSocketConnectionClosedException):
#         self.logger.error("Lost websocket connection")
#     else:
#         self.logger.error("Websocket exception: {}".format(e))
#     if self.ws.connected:
#         self.ws.send_close()
#     self.ws.connected = False
#     self._callback(self.on_error, e)
#     break
# except Exception as e:
#     self.logger.error("Exception in read_data: {}".format(e))
#     if self.ws.connected:
#        self.ws.send_close()
#     self.ws.connected = False
#     self._callback(self.on_error, e)
#     break

@retry(stop=stop_after_attempt(50), wait=wait_fixed(5), retry=retry_if_exception_type(Exception))
def create_websocket_client(symbols, queue):
    def on_message_wrapper(_, message):
        message_handler(_, queue, message)

    while True:
        try:
            client = UMFuturesWebsocketClient(
                on_message=on_message_wrapper,
                #on_close=on_close_wrapper,
                # on_error=on_error_wrapper,
                is_combined=True,
                proxies=my_proxies
            )
            
            for symbol in symbols:
                client.agg_trade(symbol)
                tm=int(time.time()+2)*1000   # delay 2 second
                webstart_t_dict[symbol]=tm
            print(f"start client:",
                  threading.get_ident(), symbols, time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())), flush=True)
            
            lasttm=int(time.time()*1000)
            time.sleep(2)  # delay 2 second
            while client.socket_manager.ws.connected:
                valid_tm=int(time.time()*1000-800) #这里的800ms应该是T到本地处理完的delta 
                if valid_tm>int(lasttm/1000+1)*1000: 
                    for symbol in symbols:
                        data=dict(e='tmflag', valid_tm=valid_tm)
                        queue.put((symbol, data))
                    lasttm=valid_tm
                time.sleep(0.2)
            print(f"WebSocket error with. wait Reconnecting...", 
                  threading.get_ident(), symbols, time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())), flush=True)
        except Exception as e:
            print(f"Error of own create_websocket_client: {str(e)}",
                  threading.get_ident(), symbols, time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())), flush=True)
            raise
        #finally:
            #print(f"Stop client for: {symbols}")
            #client.stop()

@retry(stop=stop_after_attempt(50), wait=wait_fixed(5), retry=retry_if_exception_type(Exception))
def fetch_and_add_data(queue, symbol, latest_t_dict_copy):
    if symbol not in latest_t_dict_copy:
        return
    rest_client = UMFutures(key='OugVodq3VF9syMAWI9iOGBwaIb4G0Pv3xVXSUOto24Oc4vImXaZBcpOyqL4uwkxF',secret='tNGt7sG9kMatPYqn1TcX8fcbds4jdgN21TY8TKD60rZRjuKxy2W6yZCLMuOCSyKw',proxies=my_proxies)
    try:
        start_time = latest_t_dict_copy[symbol]
        end_time = int(datetime.now(timezone.utc).timestamp() * 1000)
        all_fix_data = []
        fetch_data = []
        # whil fetch_data超过1000条
        while True:
            fetch_data = rest_client.agg_trades(symbol=symbol, startTime=start_time, endTime=end_time, limit=1000)
            if len(fetch_data)> 0:
                all_fix_data+=fetch_data
                start_time=fetch_data[-1]['T']
            if len(fetch_data) < 1000:
                break
            
        if symbol not in data_dict:
            data_dict[symbol] = pd.DataFrame(columns=[
                'event_time', 'agg_order_id', 'price', 'quantity',
                'first_order_id', 'last_order_id', 'transact_time', 'is_buyer_maker'
            ])

        new_data = pd.DataFrame([{
            'event_time': item['T'],
            'agg_order_id': item['a'],
            'price': item['p'],
            'quantity': item['q'],
            'first_order_id': item['f'],
            'last_order_id': item['l'],
            'transact_time': item['T'],
            'is_buyer_maker': item['m']
        } for item in all_fix_data])

        data=dict(e="fixdata", new_data=new_data)
        queue.put((symbol, data))
        # data_dict[symbol] = pd.concat([data_dict[symbol], new_data]).drop_duplicates(subset='agg_order_id').reset_index(drop=True)

    except Exception as e:
        if symbol == 'WUSDT':
          print(f"Error fetching data for {symbol}: {str(e)}", flush=True)
        raise
    finally:
        rest_client.close()

def process_function(queue, sec_data_queue, secdata):
    while True:
        symbol, data=queue.get()
        valid_tm_dict={}
        try:
            if symbol not in data_dict:
                data_dict[symbol] = pd.DataFrame(columns=[
                    'event_time', 'agg_order_id', 'price', 'quantity',
                    'first_order_id', 'last_order_id', 'transact_time', 'is_buyer_maker'
                ])
            if data['e']=='aggTrade':
                if random.random() < g_log_ratio*2:
                    current_time_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
                    print(f"Current-T {symbol}: {current_time_ms - int(data['T'])} ms", flush=True)
                    print(f"Current-E {symbol}: {current_time_ms - int(data['E'])} ms", flush=True)
                    print(f"Length of data_dict[{symbol}]: {data_dict[symbol].shape[0]}  queue_size:", queue.qsize(), flush=True)
    
                # add new data
                start_time = time.time()
                new_row = pd.DataFrame([{
                    'event_time': data['E'],
                    'agg_order_id': data['a'],
                    'price': data['p'],
                    'quantity': data['q'],
                    'first_order_id': data['f'],
                    'last_order_id': data['l'],
                    'transact_time': data['T'],
                    'is_buyer_maker': data['m']
                }])
                data_dict[symbol] = pd.concat([data_dict[symbol], new_row], ignore_index=True)
                
                valid_tm=data_dict[symbol]['transact_time'].max()
                if symbol in valid_tm_dict:
                    valid_tm=max(valid_tm_dict[symbol], valid_tm)
                
                if random.random() < g_log_ratio/2:
                    end_time = time.time()
                    elapsed_time_ms = (end_time - start_time) * 1000
                    print(f"Add Cost: {symbol}: {elapsed_time_ms:.2f} ms.", 
                          "valid_tm:", tools.tmu2ms(valid_tm), tools.tmu2ms(data_dict[symbol]['transact_time'].max()), 
                          tools.tmu2ms(valid_tm_dict[symbol]),
                          flush=True)
            else:
                valid_tm=data['valid_tm']
                         
            if (symbol not in valid_tm_dict) or int(valid_tm_dict[symbol]/1000) < int(valid_tm/1000):
                # start_tm=webstart_t_dict[symbol]
                start_tm=int(int(valid_tm/1000)*1000)-1000
                df=data_dict[symbol]
    
                tmpdf=df[(df['transact_time']>=start_tm) & (df['transact_time']<valid_tm)].copy()
                
                tmpdf['price']=tmpdf['price'].astype(np.float32)
                tmpdf['quantity']=tmpdf['quantity'].astype(np.float32)
                tmpdf['transact_time']=tmpdf['transact_time'].astype(int)
                tmpdf['is_buyer_maker']=tmpdf['is_buyer_maker'].astype(bool)
      
                # skl.merge_and_write(symbol, tmpdf, secdata)
                sec_data_queue.put((symbol, tmpdf))
                #todo
                data_dict[symbol] = df[df['transact_time'] >= valid_tm] 
                valid_tm_dict[symbol]=valid_tm
                if random.random() < g_log_ratio:
                    cur_tm=int(time.time()*1000)
                    print("data_merge:", symbol, tools.tmu2ms(valid_tm), tools.tmu2ms(start_tm), 
                          "delay(ms):", cur_tm-valid_tm, 
                          "queue len:", queue.qsize(),
                          "len:", len(tmpdf), len(data_dict[symbol]), 
                          flush=True)
            else:
                if random.random() < g_log_ratio/2:
                    print("skip merge:", symbol, tools.tmu2ms(valid_tm), data['e'], flush=True)

        except Exception as e:
            print(f"Error of process_function: {str(e)}", flush=True)
                
def getsids():
    timeout=60
    um_futures_client = UMFutures(proxies=my_proxies, timeout=timeout)
    exchange_info=um_futures_client.exchange_info()
    ud.readuniverse(ud.g_data)
    symbols=[]
    for item in exchange_info["symbols"]:
        sid=item['symbol']
        if sid in ud.g_data['sids']:
            symbols.append(sid)
    return symbols
  
g_sec_data_queue=Queue(maxsize=100000)
def start_realtime_update(sec_data_queue, secdata):
    # WebSocket线程数
    n_threads = 32
    # 进程数
    n_processes = n_threads
    g_symbols=getsids()
    # g_symbols=g_symbols[:50]
    try:
        # 创建queues
        queues = [Queue(maxsize=100000) for _ in range(n_processes)]
        
        # 启动进程
        processes = []
        for q in queues:
            p = Process(target=process_function, args=(q, sec_data_queue, secdata))
            processes.append(p)
            p.start()
        time.sleep(1.0)
        # 启动线程
        threads = []
        symbols_set=[]
        symbols_set.append(g_symbols[0:1])
        symbols_set.append(g_symbols[1:2])
        start=len(symbols_set)
        symbols=g_symbols[start:]
        symbols_set+=[symbols[i::(n_threads-start)] for i in range(n_threads-start)]
        for i in range(n_threads):
            # thread_symbols = g_symbols[i::n_threads]
            thread_symbols=symbols_set[i]
            t = threading.Thread(target=create_websocket_client, args=(thread_symbols, queues[i]))
            time.sleep(0.01)
            threads.append(t)
            t.start()

        # for t in threads:
        #     t.join()

        # # for q in queues:
        # #     q.put(None)

        # for p in processes:
        #     p.join()

    except Exception as e:
        print(f"Error of main: {str(e)}", flush=True)
        
def realtime_write(sec_data_queue, secdata):
    sec_data_queue=g_sec_data_queue
    start_realtime_update(sec_data_queue, secdata)
    cnt=0
    while True:
        symbol, df=sec_data_queue.get()
        sidx=ud.g_data['sidmap'][symbol]
        if df['transact_time'].shape[0]>0:
            dfagg=skl.merge_and_write(symbol, df, secdata)
        if cnt%100==0:
            secdata.flush()
        cnt=(cnt+1)%100000000
        if random.random() < g_log_ratio:
            print(symbol,"realtime_write queue len:", sec_data_queue.qsize(), flush=True)
        
    
if __name__ == "__main__":
    
    cfgpath="./config/sec_trades.yaml"
    with open(cfgpath) as f:
        g_cfg = yaml.load(f, Loader=yaml.FullLoader)
    os.environ["http_proxy"]=g_cfg["http_proxy"]
    os.environ["https_proxy"]=g_cfg["https_proxy"]
    ud.readuniverse(ud.g_data)
    g_secdata=skl.SecondData(g_cfg["secdata_path"], ud.g_data["sids"], skl.kline_fields, mode="r+")
    realtime_write(g_sec_data_queue, g_secdata)
