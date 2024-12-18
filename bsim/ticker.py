import time
import json
import logging
from binance.websocket.um_futures.websocket_client import UMFuturesWebsocketClient


class BookTickerClient:
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
                           'COTIUSDT': 5, 'CHRUSDT': 4, 'MANAUSDT': 4, 'ALICEUSDT': 3, 'HBARUSDT': 5, 'ONEUSDT': 5,
                           'LINAUSDT': 5, 'STMXUSDT': 5, 'DENTUSDT': 6, 'CELRUSDT': 5, 'HOTUSDT': 6, 'MTLUSDT': 4,
                           'OGNUSDT': 4, 'NKNUSDT': 5, 'SCUSDT': 6, 'DGBUSDT': 5, '1000SHIBUSDT': 6, 'BAKEUSDT': 4,
                           'GTCUSDT': 3, 'BTCDOMUSDT': 1, 'IOTXUSDT': 5, 'RAYUSDT': 3, 'C98USDT': 4, 'MASKUSDT': 4,
                           'ATAUSDT': 4, 'DYDXUSDT': 3, '1000XECUSDT': 5, 'GALAUSDT': 5, 'CELOUSDT': 3, 'ARUSDT': 3,
                           'KLAYUSDT': 4, 'ARPAUSDT': 5, 'CTSIUSDT': 4, 'LPTUSDT': 3, 'ENSUSDT': 3, 'PEOPLEUSDT': 5,
                           'ROSEUSDT': 5, 'DUSKUSDT': 5, 'FLOWUSDT': 3, 'IMXUSDT': 4, 'API3USDT': 4, 'GMTUSDT': 5,
                           'APEUSDT': 4, 'WOOUSDT': 5, 'FTTUSDT': 4, 'JASMYUSDT': 6, 'DARUSDT': 4, 'OPUSDT': 4,
                           'INJUSDT': 3, 'STGUSDT': 4, 'SPELLUSDT': 7, '1000LUNCUSDT': 5, 'LUNA2USDT': 4, 'LDOUSDT': 4,
                           'CVXUSDT': 3, 'ICPUSDT': 3, 'APTUSDT': 3, 'QNTUSDT': 2, 'FETUSDT': 4, 'FXSUSDT': 4,
                           'HOOKUSDT': 4, 'MAGICUSDT': 4, 'TUSDT': 5, 'HIGHUSDT': 4, 'MINAUSDT': 4, 'ASTRUSDT': 5,
                           'AGIXUSDT': 4, 'PHBUSDT': 4, 'GMXUSDT': 3, 'CFXUSDT': 5, 'STXUSDT': 4, 'BNXUSDT': 4,
                           'ACHUSDT': 6, 'SSVUSDT': 3, 'CKBUSDT': 6, 'PERPUSDT': 4, 'TRUUSDT': 5, 'LQTYUSDT': 4,
                           'USDCUSDT': 6, 'IDUSDT': 5, 'ARBUSDT': 4, 'JOEUSDT': 4, 'TLMUSDT': 6, 'AMBUSDT': 6,
                           'LEVERUSDT': 7, 'RDNTUSDT': 5, 'HFTUSDT': 5, 'XVSUSDT': 3, 'BLURUSDT': 4, 'EDUUSDT': 4,
                           'IDEXUSDT': 5, 'SUIUSDT': 4, '1000PEPEUSDT': 7, '1000FLOKIUSDT': 5, 'UMAUSDT': 3,
                           'RADUSDT': 4, 'KEYUSDT': 6, 'COMBOUSDT': 4, 'NMRUSDT': 3, 'MAVUSDT': 5, 'MDTUSDT': 5,
                           'XVGUSDT': 6, 'WLDUSDT': 4, 'PENDLEUSDT': 4, 'ARKMUSDT': 4, 'AGLDUSDT': 4, 'YGGUSDT': 4,
                           'DODOXUSDT': 6, 'BNTUSDT': 5, 'OXTUSDT': 5, 'SEIUSDT': 4, 'CYBERUSDT': 3, 'HIFIUSDT': 4,
                           'ARKUSDT': 4, 'FRONTUSDT': 4, 'GLMRUSDT': 5, 'BICOUSDT': 4, 'STRAXUSDT': 4, 'LOOMUSDT': 5,
                           'BIGTIMEUSDT': 4, 'BONDUSDT': 3, 'ORBSUSDT': 5, 'STPTUSDT': 5, 'WAXPUSDT': 5, 'BSVUSDT': 2,
                           'RIFUSDT': 5, 'POLYXUSDT': 5, 'GASUSDT': 3, 'POWRUSDT': 4, 'SLPUSDT': 6, 'TIAUSDT': 4,
                           'SNTUSDT': 5, 'CAKEUSDT': 4, 'MEMEUSDT': 6, 'TWTUSDT': 4, 'TOKENUSDT': 5, 'ORDIUSDT': 3,
                           'STEEMUSDT': 5, 'BADGERUSDT': 4, 'ILVUSDT': 2, 'NTRNUSDT': 4, 'KASUSDT': 5, 'BEAMXUSDT': 6,
                           '1000BONKUSDT': 6, 'PYTHUSDT': 4, 'SUPERUSDT': 4, 'USTCUSDT': 5, 'ONGUSDT': 5, 'ETHWUSDT': 4,
                           'JTOUSDT': 4, '1000SATSUSDT': 7, 'AUCTIONUSDT': 3, '1000RATSUSDT': 5, 'ACEUSDT': 4,
                           'MOVRUSDT': 3, 'NFPUSDT': 4, 'AIUSDT': 5, 'XAIUSDT': 4, 'WIFUSDT': 4, 'MANTAUSDT': 4,
                           'ONDOUSDT': 4, 'LSKUSDT': 4, 'ALTUSDT': 5, 'JUPUSDT': 4, 'ZETAUSDT': 4, 'RONINUSDT': 4,
                           'DYMUSDT': 4, 'OMUSDT': 5, 'PIXELUSDT': 4, 'STRKUSDT': 4, 'MAVIAUSDT': 4, 'GLMUSDT': 4,
                           'PORTALUSDT': 4, 'TONUSDT': 4, 'AXLUSDT': 4, 'MYROUSDT': 5, 'METISUSDT': 2, 'AEVOUSDT': 4,
                           'VANRYUSDT': 5, 'BOMEUSDT': 6, 'ETHFIUSDT': 3, 'ENAUSDT': 4, 'WUSDT': 4, 'TNSRUSDT': 4,
                           'SAGAUSDT': 4, 'TAOUSDT': 2, 'OMNIUSDT': 3, 'REZUSDT': 5, 'BBUSDT': 4, 'NOTUSDT': 6,
                           'TURBOUSDT': 6, 'IOUSDT': 3, 'ZKUSDT': 5, 'MEWUSDT': 6, 'LISTAUSDT': 4, 'ZROUSDT': 3,
                           'RENDERUSDT': 3, 'BANANAUSDT': 3, 'RAREUSDT': 4, 'GUSDT': 5, 'SYNUSDT': 4, 'SYSUSDT': 4,
                           'VOXELUSDT': 4, 'BRETTUSDT': 5, 'ALPACAUSDT': 5, 'POPCATUSDT': 4, 'SUNUSDT': 6}

    ticker_price_map = {'BTCUSDT': '0.10', 'ETHUSDT': '0.01', 'BCHUSDT': '0.01', 'XRPUSDT': '0.0001',
                        'EOSUSDT': '0.001', 'LTCUSDT': '0.01', 'TRXUSDT': '0.00001', 'ETCUSDT': '0.001',
                        'LINKUSDT': '0.001', 'XLMUSDT': '0.00001', 'ADAUSDT': '0.00010', 'XMRUSDT': '0.01',
                        'DASHUSDT': '0.01', 'ZECUSDT': '0.01', 'XTZUSDT': '0.001', 'BNBUSDT': '0.010',
                        'ATOMUSDT': '0.001', 'ONTUSDT': '0.0001', 'IOTAUSDT': '0.0001', 'BATUSDT': '0.0001',
                        'VETUSDT': '0.000001', 'NEOUSDT': '0.001', 'QTUMUSDT': '0.001', 'IOSTUSDT': '0.000001',
                        'THETAUSDT': '0.0001', 'ALGOUSDT': '0.0001', 'ZILUSDT': '0.00001', 'KNCUSDT': '0.00010',
                        'ZRXUSDT': '0.0001', 'COMPUSDT': '0.01', 'OMGUSDT': '0.0001', 'DOGEUSDT': '0.000010',
                        'SXPUSDT': '0.0001', 'KAVAUSDT': '0.0001', 'BANDUSDT': '0.0001', 'RLCUSDT': '0.0001',
                        'WAVESUSDT': '0.0001', 'MKRUSDT': '0.10', 'SNXUSDT': '0.001', 'DOTUSDT': '0.001',
                        'DEFIUSDT': '0.1', 'YFIUSDT': '1', 'BALUSDT': '0.001', 'CRVUSDT': '0.001', 'TRBUSDT': '0.001',
                        'RUNEUSDT': '0.0010', 'SUSHIUSDT': '0.0001', 'EGLDUSDT': '0.001', 'SOLUSDT': '0.0010',
                        'ICXUSDT': '0.0001', 'STORJUSDT': '0.0001', 'BLZUSDT': '0.00001', 'UNIUSDT': '0.0010',
                        'AVAXUSDT': '0.0010', 'FTMUSDT': '0.000100', 'ENJUSDT': '0.00001', 'FLMUSDT': '0.0001',
                        'RENUSDT': '0.00001', 'KSMUSDT': '0.001', 'NEARUSDT': '0.0010', 'AAVEUSDT': '0.010',
                        'FILUSDT': '0.001', 'RSRUSDT': '0.000001', 'LRCUSDT': '0.00001', 'MATICUSDT': '0.00010',
                        'OCEANUSDT': '0.00010', 'CVCUSDT': '0.00001', 'BELUSDT': '0.00010', 'CTKUSDT': '0.00010',
                        'AXSUSDT': '0.00100', 'ALPHAUSDT': '0.00001', 'ZENUSDT': '0.001', 'SKLUSDT': '0.00001',
                        'GRTUSDT': '0.00001', '1INCHUSDT': '0.0001', 'CHZUSDT': '0.00001', 'SANDUSDT': '0.00010',
                        'ANKRUSDT': '0.000010', 'LITUSDT': '0.001', 'UNFIUSDT': '0.001', 'REEFUSDT': '0.000001',
                        'RVNUSDT': '0.00001', 'SFPUSDT': '0.0001', 'XEMUSDT': '0.0001', 'COTIUSDT': '0.00001',
                        'CHRUSDT': '0.0001', 'MANAUSDT': '0.0001', 'ALICEUSDT': '0.001', 'HBARUSDT': '0.00001',
                        'ONEUSDT': '0.00001', 'LINAUSDT': '0.00001', 'STMXUSDT': '0.00001', 'DENTUSDT': '0.000001',
                        'CELRUSDT': '0.00001', 'HOTUSDT': '0.000001', 'MTLUSDT': '0.0001', 'OGNUSDT': '0.0001',
                        'NKNUSDT': '0.00001', 'SCUSDT': '0.000001', 'DGBUSDT': '0.00001', '1000SHIBUSDT': '0.000001',
                        'BAKEUSDT': '0.0001', 'GTCUSDT': '0.001', 'BTCDOMUSDT': '0.1', 'IOTXUSDT': '0.00001',
                        'RAYUSDT': '0.001', 'C98USDT': '0.0001', 'MASKUSDT': '0.0001', 'ATAUSDT': '0.0001',
                        'DYDXUSDT': '0.001', '1000XECUSDT': '0.00001', 'GALAUSDT': '0.00001', 'CELOUSDT': '0.001',
                        'ARUSDT': '0.001', 'KLAYUSDT': '0.0001', 'ARPAUSDT': '0.00001', 'CTSIUSDT': '0.0001',
                        'LPTUSDT': '0.001', 'ENSUSDT': '0.001', 'PEOPLEUSDT': '0.00001', 'ROSEUSDT': '0.00001',
                        'DUSKUSDT': '0.00001', 'FLOWUSDT': '0.001', 'IMXUSDT': '0.0001', 'API3USDT': '0.0001',
                        'GMTUSDT': '0.00001', 'APEUSDT': '0.0001', 'WOOUSDT': '0.00001', 'FTTUSDT': '0.0001',
                        'JASMYUSDT': '0.000001', 'DARUSDT': '0.0001', 'OPUSDT': '0.0001000', 'INJUSDT': '0.001000',
                        'STGUSDT': '0.0001000', 'SPELLUSDT': '0.0000001', '1000LUNCUSDT': '0.0000100',
                        'LUNA2USDT': '0.0001000', 'LDOUSDT': '0.000100', 'CVXUSDT': '0.001000', 'ICPUSDT': '0.001000',
                        'APTUSDT': '0.00100', 'QNTUSDT': '0.010000', 'FETUSDT': '0.0001000', 'FXSUSDT': '0.000100',
                        'HOOKUSDT': '0.000100', 'MAGICUSDT': '0.000100', 'TUSDT': '0.0000100', 'HIGHUSDT': '0.000100',
                        'MINAUSDT': '0.0001000', 'ASTRUSDT': '0.0000100', 'AGIXUSDT': '0.0001000',
                        'PHBUSDT': '0.0001000', 'GMXUSDT': '0.001000', 'CFXUSDT': '0.0000100', 'STXUSDT': '0.0001000',
                        'BNXUSDT': '0.000100', 'ACHUSDT': '0.0000010', 'SSVUSDT': '0.001000', 'CKBUSDT': '0.0000010',
                        'PERPUSDT': '0.000100', 'TRUUSDT': '0.0000100', 'LQTYUSDT': '0.000100', 'USDCUSDT': '0.0000010',
                        'IDUSDT': '0.0000100', 'ARBUSDT': '0.000100', 'JOEUSDT': '0.0001000', 'TLMUSDT': '0.0000010',
                        'AMBUSDT': '0.0000010', 'LEVERUSDT': '0.0000001', 'RDNTUSDT': '0.0000100',
                        'HFTUSDT': '0.0000100', 'XVSUSDT': '0.001000', 'BLURUSDT': '0.0001000', 'EDUUSDT': '0.0001000',
                        'IDEXUSDT': '0.0000100', 'SUIUSDT': '0.000100', '1000PEPEUSDT': '0.0000001',
                        '1000FLOKIUSDT': '0.0000100', 'UMAUSDT': '0.001000', 'RADUSDT': '0.000100',
                        'KEYUSDT': '0.0000010', 'COMBOUSDT': '0.000100', 'NMRUSDT': '0.001000', 'MAVUSDT': '0.0000100',
                        'MDTUSDT': '0.0000100', 'XVGUSDT': '0.0000010', 'WLDUSDT': '0.0001000',
                        'PENDLEUSDT': '0.0001000', 'ARKMUSDT': '0.0001000', 'AGLDUSDT': '0.0001000',
                        'YGGUSDT': '0.0001000', 'DODOXUSDT': '0.0000010', 'BNTUSDT': '0.0000100',
                        'OXTUSDT': '0.0000100', 'SEIUSDT': '0.0001000', 'CYBERUSDT': '0.001000',
                        'HIFIUSDT': '0.0001000', 'ARKUSDT': '0.0001000', 'FRONTUSDT': '0.0001000',
                        'GLMRUSDT': '0.0000100', 'BICOUSDT': '0.0001000', 'STRAXUSDT': '0.0001000',
                        'LOOMUSDT': '0.0000100', 'BIGTIMEUSDT': '0.0001000', 'BONDUSDT': '0.001000',
                        'ORBSUSDT': '0.0000100', 'STPTUSDT': '0.0000100', 'WAXPUSDT': '0.0000100', 'BSVUSDT': '0.01000',
                        'RIFUSDT': '0.0000100', 'POLYXUSDT': '0.0000100', 'GASUSDT': '0.001000',
                        'POWRUSDT': '0.0001000', 'SLPUSDT': '0.0000010', 'TIAUSDT': '0.0001000', 'SNTUSDT': '0.0000100',
                        'CAKEUSDT': '0.0001000', 'MEMEUSDT': '0.0000010', 'TWTUSDT': '0.000100',
                        'TOKENUSDT': '0.0000100', 'ORDIUSDT': '0.001000', 'STEEMUSDT': '0.000010',
                        'BADGERUSDT': '0.000100', 'ILVUSDT': '0.01000', 'NTRNUSDT': '0.000100', 'KASUSDT': '0.0000100',
                        'BEAMXUSDT': '0.0000010', '1000BONKUSDT': '0.0000010', 'PYTHUSDT': '0.0001000',
                        'SUPERUSDT': '0.0001000', 'USTCUSDT': '0.0000100', 'ONGUSDT': '0.0000100',
                        'ETHWUSDT': '0.000100', 'JTOUSDT': '0.000100', '1000SATSUSDT': '0.0000001',
                        'AUCTIONUSDT': '0.001000', '1000RATSUSDT': '0.0000100', 'ACEUSDT': '0.000100',
                        'MOVRUSDT': '0.001000', 'NFPUSDT': '0.0001000', 'AIUSDT': '0.000010', 'XAIUSDT': '0.0001000',
                        'WIFUSDT': '0.0001000', 'MANTAUSDT': '0.0001000', 'ONDOUSDT': '0.0001000',
                        'LSKUSDT': '0.000100', 'ALTUSDT': '0.0000100', 'JUPUSDT': '0.0001000', 'ZETAUSDT': '0.000100',
                        'RONINUSDT': '0.000100', 'DYMUSDT': '0.000100', 'OMUSDT': '0.0000100', 'PIXELUSDT': '0.0001000',
                        'STRKUSDT': '0.0001000', 'MAVIAUSDT': '0.0001000', 'GLMUSDT': '0.0001000',
                        'PORTALUSDT': '0.0001000', 'TONUSDT': '0.0001000', 'AXLUSDT': '0.0001000',
                        'MYROUSDT': '0.0000100', 'METISUSDT': '0.0100', 'AEVOUSDT': '0.0001000',
                        'VANRYUSDT': '0.0000100', 'BOMEUSDT': '0.0000010', 'ETHFIUSDT': '0.0010000',
                        'ENAUSDT': '0.0001000', 'WUSDT': '0.0001000', 'TNSRUSDT': '0.0001000', 'SAGAUSDT': '0.0001000',
                        'TAOUSDT': '0.01', 'OMNIUSDT': '0.0010', 'REZUSDT': '0.0000100', 'BBUSDT': '0.0001000',
                        'NOTUSDT': '0.0000010', 'TURBOUSDT': '0.0000010', 'IOUSDT': '0.0010000', 'ZKUSDT': '0.0000100',
                        'MEWUSDT': '0.0000010', 'LISTAUSDT': '0.0001000', 'ZROUSDT': '0.0010000',
                        'RENDERUSDT': '0.0010000', 'BANANAUSDT': '0.0010000', 'RAREUSDT': '0.0001000',
                        'GUSDT': '0.0000100', 'SYNUSDT': '0.0001000', 'SYSUSDT': '0.0001000', 'VOXELUSDT': '0.0001000',
                        'BRETTUSDT': '0.0000100', 'ALPACAUSDT': '0.0000100', 'POPCATUSDT': '0.0001000',
                        'SUNUSDT': '0.0000010'}

    def __init__(self, proxies=None):
        self.ticker_data = {}
        self.client = None
        self.proxies = proxies

        # Set up the logger
        self.logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def _message_handler(self, _, message):
        try:
            result = json.loads(message)
            symbol = result.get('s')
            if symbol in self.price_precision_map:
                bid_price = round(float(result.get('b')), self.price_precision_map[symbol])
                ask_price = round(float(result.get('a')), self.price_precision_map[symbol])

                if symbol and bid_price and ask_price:
                    self.ticker_data[symbol] = {'bid_price': float(bid_price), 'ask_price': float(ask_price)}
            # self.logger.info(f"Number of symbols tracked: {len(self.ticker_data)}")
        except Exception as e:
            self.logger.error(f"Failed to parse message: {e}")

    def start(self):
        retry_delay = 5
        while True:
            if self.client:
                self.logger.warning("Client is already running")
                return
            try:
                self.client = UMFuturesWebsocketClient(on_message=self._message_handler, proxies=self.proxies)
                self.client.book_ticker(symbol=None)
                self.logger.info("Client started and subscription initiated")
                break
            except Exception as e:
                self.logger.error(f"Error starting client, retrying in {retry_delay} seconds: {e}")
                self.client = None
                time.sleep(retry_delay)

    def stop(self):
        if not self.client:
            self.logger.warning("Client is not running")
            return
        try:
            self.client.stop()
            self.client = None
            self.logger.info("Client stopped")
        except Exception as e:
            self.logger.error(f"Error stopping client: {e}")
            self.client = None

    def get_ask(self, symbol):
        try:
            data = self.ticker_data.get(symbol)
            if not data:
                return None
            return data.get('ask_price')
        except Exception as e:
            self.logger.error(f"Failed to get ask price for {symbol}: {e}")
            return None

    def get_bid(self, symbol):
        try:
            data = self.ticker_data.get(symbol)
            if not data:
                return None
            return data.get('bid_price')
        except Exception as e:
            self.logger.error(f"Failed to get bid price for {symbol}: {e}")
            return None

    def get_ask_limit(self, symbol, limit):
        ask_price = self.get_ask(symbol)
        if ask_price is None or symbol not in self.ticker_price_map:
            self.logger.warning(f"No data available or ticker price missing for {symbol}")
            return None

        ticker_price = float(self.ticker_price_map[symbol])
        precision = self.price_precision_map.get(symbol, 2)
        return [round(ask_price + i * ticker_price, precision) for i in range(limit)]

    def get_bid_limit(self, symbol, limit):
        bid_price = self.get_bid(symbol)
        if bid_price is None or symbol not in self.ticker_price_map:
            self.logger.warning(f"No data available or ticker price missing for {symbol}")
            return None

        ticker_price = float(self.ticker_price_map[symbol])
        precision = self.price_precision_map.get(symbol, 2)
        return [round(bid_price - i * ticker_price, precision) for i in range(limit)]