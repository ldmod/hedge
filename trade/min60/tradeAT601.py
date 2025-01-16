import sys
import os

sys.path.append(os.path.abspath(__file__ + "../../../../../"))
import yaml
import random
import copy
import time
import logging
from multiprocessing import Lock
from binance.um_futures import UMFutures
from datetime import datetime, timedelta
from dingtalkchatbot.chatbot import DingtalkChatbot
# import cryptoqt.bsim.gorders as gorders
import cryptoqt.trade.min60.gorders_at as gorders
import cryptoqt.bsim.ticker as ticker
import cryptoqt.data.updatedata as ud
import cryptoqt.data.sec_klines.sec_klines as sk
import cryptoqt.data.tools as tools
import concurrent.futures
import json
import pandas as pd
# Post Maker Order Fail
SUB_STR = "Post Only order will be rejected"
CANCEL_STR = "Unknown order sent"
# Suffix for signal file
signal_suffix = '_book.csv'
# prefix for local order id
PREFIX_ORDER_ID = "at"

available_token_map = {'BTCUSDT': 0.0, 'ETHUSDT': 0.0, 'BCHUSDT': 0.0, 'XRPUSDT': 0.0, 'EOSUSDT': 0.0,'BNBUSDT': 0.0,
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
                       'RENDERUSDT': 3, 'BANANAUSDT': 3, 'RAREUSDT': 4, 'GUSDT': 5, 'SYNUSDT': 4}

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
                          'COTIUSDT': 0, 'CHRUSDT': 0, 'MANAUSDT': 0, 'ALICEUSDT': 1, 'HBARUSDT': 0, 'ONEUSDT': 0,
                          'LINAUSDT': 0, 'STMXUSDT': 0, 'DENTUSDT': 0, 'CELRUSDT': 0, 'HOTUSDT': 0, 'MTLUSDT': 0,
                          'OGNUSDT': 0, 'NKNUSDT': 0, 'SCUSDT': 0, 'DGBUSDT': 0, '1000SHIBUSDT': 0, 'BAKEUSDT': 0,
                          'GTCUSDT': 1, 'BTCDOMUSDT': 3, 'IOTXUSDT': 0, 'RAYUSDT': 1, 'C98USDT': 0, 'MASKUSDT': 0,
                          'ATAUSDT': 0, 'DYDXUSDT': 1, '1000XECUSDT': 0, 'GALAUSDT': 0, 'CELOUSDT': 1, 'ARUSDT': 1,
                          'KLAYUSDT': 1, 'ARPAUSDT': 0, 'CTSIUSDT': 0, 'LPTUSDT': 1, 'ENSUSDT': 1, 'PEOPLEUSDT': 0,
                          'ROSEUSDT': 0, 'DUSKUSDT': 0, 'FLOWUSDT': 1, 'IMXUSDT': 0, 'API3USDT': 1, 'GMTUSDT': 0,
                          'APEUSDT': 0, 'WOOUSDT': 0, 'FTTUSDT': 1, 'JASMYUSDT': 0, 'DARUSDT': 1, 'OPUSDT': 1,
                          'INJUSDT': 1, 'STGUSDT': 0, 'SPELLUSDT': 0, '1000LUNCUSDT': 0, 'LUNA2USDT': 0,
                          'LDOUSDT': 0, 'CVXUSDT': 0, 'ICPUSDT': 0, 'APTUSDT': 1, 'QNTUSDT': 1, 'FETUSDT': 0,
                          'FXSUSDT': 1, 'HOOKUSDT': 1, 'MAGICUSDT': 1, 'TUSDT': 0, 'HIGHUSDT': 1, 'MINAUSDT': 0,
                          'ASTRUSDT': 0, 'AGIXUSDT': 0, 'PHBUSDT': 0, 'GMXUSDT': 2, 'CFXUSDT': 0, 'STXUSDT': 0,
                          'BNXUSDT': 1, 'ACHUSDT': 0, 'SSVUSDT': 2, 'CKBUSDT': 0, 'PERPUSDT': 1, 'TRUUSDT': 0,
                          'LQTYUSDT': 1, 'USDCUSDT': 0, 'IDUSDT': 0, 'ARBUSDT': 1, 'JOEUSDT': 0, 'TLMUSDT': 0,
                          'AMBUSDT': 0, 'LEVERUSDT': 0, 'RDNTUSDT': 0, 'HFTUSDT': 0, 'XVSUSDT': 1, 'BLURUSDT': 0,
                          'EDUUSDT': 0, 'IDEXUSDT': 0, 'SUIUSDT': 1, '1000PEPEUSDT': 0, '1000FLOKIUSDT': 0,
                          'UMAUSDT': 0, 'RADUSDT': 0, 'KEYUSDT': 0, 'COMBOUSDT': 1, 'NMRUSDT': 1, 'MAVUSDT': 0,
                          'MDTUSDT': 0, 'XVGUSDT': 0, 'WLDUSDT': 0, 'PENDLEUSDT': 0, 'ARKMUSDT': 0, 'AGLDUSDT': 0,
                          'YGGUSDT': 0, 'DODOXUSDT': 0, 'BNTUSDT': 0, 'OXTUSDT': 0, 'SEIUSDT': 0, 'CYBERUSDT': 1,
                          'HIFIUSDT': 0, 'ARKUSDT': 0, 'FRONTUSDT': 0, 'GLMRUSDT': 0, 'BICOUSDT': 0, 'STRAXUSDT': 0,
                          'LOOMUSDT': 0, 'BIGTIMEUSDT': 0, 'BONDUSDT': 1, 'ORBSUSDT': 0, 'STPTUSDT': 0,
                          'WAXPUSDT': 0, 'BSVUSDT': 1, 'RIFUSDT': 0, 'POLYXUSDT': 0, 'GASUSDT': 1, 'POWRUSDT': 0,
                          'SLPUSDT': 0, 'TIAUSDT': 0, 'SNTUSDT': 0, 'CAKEUSDT': 0, 'MEMEUSDT': 0, 'TWTUSDT': 0,
                          'TOKENUSDT': 0, 'ORDIUSDT': 1, 'STEEMUSDT': 0, 'BADGERUSDT': 0, 'ILVUSDT': 1,
                          'NTRNUSDT': 0, 'KASUSDT': 0, 'BEAMXUSDT': 0, '1000BONKUSDT': 0, 'PYTHUSDT': 0,
                          'SUPERUSDT': 0, 'USTCUSDT': 0, 'ONGUSDT': 0, 'ETHWUSDT': 0, 'JTOUSDT': 0,
                          '1000SATSUSDT': 0, 'AUCTIONUSDT': 2, '1000RATSUSDT': 0, 'ACEUSDT': 2, 'MOVRUSDT': 2,
                          'NFPUSDT': 1, 'AIUSDT': 0, 'XAIUSDT': 0, 'WIFUSDT': 1, 'MANTAUSDT': 1, 'ONDOUSDT': 1,
                          'LSKUSDT': 0, 'ALTUSDT': 0, 'JUPUSDT': 0, 'ZETAUSDT': 0, 'RONINUSDT': 1, 'DYMUSDT': 1,
                          'OMUSDT': 1, 'PIXELUSDT': 0, 'STRKUSDT': 1, 'MAVIAUSDT': 1, 'GLMUSDT': 0, 'PORTALUSDT': 1,
                          'TONUSDT': 1, 'AXLUSDT': 1, 'MYROUSDT': 0, 'METISUSDT': 2, 'AEVOUSDT': 1, 'VANRYUSDT': 0,
                          'BOMEUSDT': 0, 'ETHFIUSDT': 1, 'ENAUSDT': 0, 'WUSDT': 1, 'TNSRUSDT': 1, 'SAGAUSDT': 1,
                          'TAOUSDT': 3, 'OMNIUSDT': 2, 'REZUSDT': 0, 'BBUSDT': 0, 'NOTUSDT': 0, 'TURBOUSDT': 0,
                          'IOUSDT': 1, 'ZKUSDT': 0, 'MEWUSDT': 0, 'LISTAUSDT': 0, 'ZROUSDT': 1, 'RENDERUSDT': 1,
                          'BANANAUSDT': 1, 'RAREUSDT': 0, 'GUSDT': 0, 'SYNUSDT': 0}

# DingDing Notify
def notify(msg):
    webhook = "https://oapi.dingtalk.com/robot/send?access_token=08981a30db7ee421f6f910cb1dbe9b722bb18420b1c54d9b5cdad300470d2cda"
    xiaoding = DingtalkChatbot(webhook)
    xiaoding.send_text(msg, is_at_all=False)


# Logger Init
def get_logger(logger_name, log_file, level=logging.INFO):
    logger = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s', "%Y-%m-%d %H:%M:%S")
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)

    logger.setLevel(level)
    logger.addHandler(fileHandler)
    return logger


# Create sell order(maker)
def sell(client, amount, symbol, price, order_id, reduce, logger_trade, logger_error):
    # return False
    try:
        params = {
            'symbol': symbol,
            'side': 'SELL',
            'reduceOnly': reduce,
            'type': 'LIMIT',
            'quantity': amount,
            'price': price,
            'newClientOrderId': order_id,
            'timeInForce': 'GTX'
        }
        # logger_trade.info(f"Sell: {symbol}, amount: {amount}, price: {price}")
        client.new_order(**params)
        logger_trade.info(f"Sell order succ: {symbol}, amount: {amount}, price: {price}")
        return False
    except Exception as e:
        if SUB_STR not in str(e):
            logger_error.info(
                "Sell error: " + str(symbol) + ", " + str(order_id) + ", amount: " + str(amount) + ", price: " + str(
                    price) + str(e))
        return True


# Create buy order(maker)
def buy(client, amount, symbol, price, order_id, reduce, logger_trade, logger_error):
    # return False
    try:
        params = {
            'symbol': symbol,
            'side': 'BUY',
            'reduceOnly': reduce,
            'type': 'LIMIT',
            'quantity': amount,
            'price': price,
            'newClientOrderId': order_id,
            'timeInForce': 'GTX'
        }
        # logger_trade.info(f"Buy: {symbol}, amount: {amount}, price: {price}")
        client.new_order(**params)
        logger_trade.info(f"Buy order succ: {symbol}, amount: {amount}, price: {price}")
        return False
    except Exception as e:
        if SUB_STR not in str(e):
            logger_error.info(
                "Buy error: " + str(symbol) + ", " + str(order_id) + ", amount: " + str(amount) + ", price: " + str(
                    price) + str(e))
        return True


# cancel order by local id
def cancel_open_order_by_id(client, symbol, order_id, logger_trade, logger_error):
    try:
        logger_trade.info(f"Cancel order by id: {symbol} {order_id}")
        client.cancel_order(symbol=symbol, origClientOrderId=order_id, recvWindow=2000)
    except Exception as e:
        if CANCEL_STR not in str(e):
            logger_error.info("cancel_open_order_by_id: " + str(e))


# Cancel all symbol set open orders
def cancel_orders_in_parallel(clients, symbol_set, logger_trade, logger_error):
    def cancel_order(client, key):
        cancel_status = True
        retry_num = 10
        while cancel_status and retry_num > 0:
            try:
                cancel_status = cancel_open_orders(client, key, logger_trade, logger_error)
                logger_trade.info(f"Cancel order by symbol: {key} cancel_status:{cancel_status}")
            except Exception as e:
                logger_error.error(f"Error canceling order for {key}: {str(e)}")
            retry_num -= 1

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(clients)) as executor:
        futures = [executor.submit(cancel_order, clients[i % len(clients)], key) for i, key in enumerate(symbol_set)]
        concurrent.futures.wait(futures)


# Cancel order by local order id
def cancel_order_by_id_in_parallel(clients, order_id_set, logger_trade, logger_error):
    def cancel_order_by_id(client, order_id):
        try:
            symbol = order_id.split("-")[-1]
            cancel_open_order_by_id(client, symbol, order_id, logger_trade, logger_error)
        except Exception as e:
            logger_error.error(f"Error canceling order {order_id}: {str(e)}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(clients)) as executor:
        futures = [executor.submit(cancel_order_by_id, clients[i % len(clients)], order_id) for i, order_id in
                   enumerate(order_id_set)]
        concurrent.futures.wait(futures)


# Post orders(buy/sell) parallel
def post_orders_in_parallel(clients, orders, logger_trade, logger_error, price_precision_map, quantity_precision_map,
                            next_tm):
    def place_order(client, order, request_ask_bid=False):
        try:
            symbol, _, order_price, order_money, order_info = order
            order_id = order_info["order_id"]
            # logger_trade.info(f"Order is: {symbol}, {order_price}, {order_money}, {order_id}")
            if abs(order_money) < get_threshold(symbol):
                return False
            result=False
            if order_money > 0:
                if request_ask_bid:
                    bids = query_bids(client, symbol, price_precision_map[symbol], logger_error)
                    bid0=bids[0]
                    shiftBp = order_price
                    order_price = max(bid0*(1-shiftBp/10000), bids[round(shiftBp)])
                    # order_price=bids[round(shiftBp)]
                    logger_trade.info(f"use best buy price is: {symbol} money:{order_money}, price:{order_price}, bid:{bids}, {shiftBp}")
                    # order_price = bid0
                    order_info["order_price"] = order_price
                trade_price = round(order_price, price_precision_map[symbol])
                trade_amount = round(abs(order_money / order_price), quantity_precision_map[symbol])
                reduce = "false"
                # if abs(order_info["target_money"])<1.0 or (order_info["target_money"] * order_info["begin_money"]) > 0 :
                #     reduce='true'
                if (trade_amount < 0) or (round(trade_price * trade_amount, 2) < get_threshold(symbol)):
                    return True
                result = buy(client, trade_amount, symbol, trade_price, order_id, reduce, logger_trade, logger_error)

            else:
                if request_ask_bid:
                    asks = query_asks(client, symbol, price_precision_map[symbol], logger_error)
                    ask0=asks[0]
                    shiftBp = order_price
                    order_price = min(ask0*(1+shiftBp/10000), asks[round(shiftBp)])
                    # order_price=asks[round(shiftBp)]
                    logger_trade.info(f"use best sell price is: {symbol} money:{order_money}, price:{order_price}, ask:{asks}, {shiftBp}")
                    # order_price = ask0
                    order_info["order_price"] = order_price
                trade_price = round(order_price, price_precision_map[symbol])
                trade_amount = round(abs(order_money / order_price), quantity_precision_map[symbol])
                reduce = "false"
                # if abs(order_info["target_money"])<1.0 or (order_info["target_money"] * order_info["begin_money"]) > 0 :
                #     reduce='true'
                if (trade_amount < 0) or (round(trade_price * trade_amount, 2) < get_threshold(symbol)):
                    return True
                result = sell(client, trade_amount, symbol, trade_price, order_id, reduce, logger_trade, logger_error)

            return result
        except Exception as e:
            logger_error.error(f"Error placing order for: {str(e)}")
            return False

    order_status = {json.dumps(order[:4]): True for order in orders}
    request_ask_bid = True
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(clients)) as executor:
        while True:
            futures = []
            for i, order in enumerate(orders):
                if not order_status[json.dumps(order[:4])]:
                    continue
                futures.append(
                    (executor.submit(place_order, clients[i % len(clients)], order, request_ask_bid), order))

            for future, order in futures:
                if not future.result():
                    order_status[json.dumps(order[:4])] = False

            time.sleep(1.0)
            request_ask_bid = True
            # Step8: Until next 5s time
            if get_current_tm() > next_tm:
                break

def query_bids(client, symbol, precision, logger_error):
    tmpres = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    try:
        depth = client.depth(symbol=symbol, limit=10)
        if depth['bids'] is None:
            return tmpres
        if depth['bids'][0] is None:
            return tmpres
        return [round(float(depth['bids'][0][0]), precision), round(float(depth['bids'][1][0]), precision), round(
            float(depth['bids'][2][0]), precision), round(float(depth['bids'][3][0]), precision), round(
            float(depth['bids'][4][0]), precision), round(float(depth['bids'][5][0]), precision), round(float(depth['bids'][6][0]), precision),
                round(float(depth['bids'][7][0]), precision), round(float(depth['bids'][8][0]), precision), round(float(depth['bids'][9][0]), precision)]
    except Exception as e:
        logger_error.info("query_bids " + str(e))
        return tmpres


def query_asks(client, symbol, precision, logger_error):
    tmpres = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    try:
        depth = client.depth(symbol=symbol, limit=10)
        if depth['asks'] is None:
            return tmpres
        if depth['asks'][0] is None:
            return tmpres
        return [round(float(depth['asks'][0][0]), precision), round(float(depth['asks'][1][0]), precision), round(
            float(depth['asks'][2][0]), precision), round(float(depth['asks'][3][0]), precision), round(
            float(depth['asks'][4][0]), precision), round(float(depth['asks'][5][0]), precision), round(float(depth['asks'][6][0]), precision),
                 round(float(depth['asks'][7][0]), precision), round(float(depth['asks'][8][0]), precision), round(float(depth['asks'][9][0]), precision)]
    except Exception as e:
        logger_error.info("query_asks: " + str(e))
        return tmpres

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
        logger_trade.info(f"Cancel order: {symbol}")
        client.cancel_open_orders(symbol=symbol, recvWindow=2000)
        return False
    except Exception as e:
        logger_error.info("modify_position_side: " + str(e))
        return True


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
        time.sleep(1)
        return True

# Change leverage(1)
def change_leverage(client, symbol, logger_error, leverage=20):
    try:
        if symbol in ["NMRUSDT"]:
            leverage=10
        client.change_leverage(symbol=symbol, leverage=leverage)
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


# Get delta time by minutes
def get_tm_by_min(start_tm, mins):
    start_time = datetime.strptime(start_tm, "%Y%m%d%H%M%S")
    start_time = start_time + timedelta(minutes=mins)
    start_index = start_time.strftime("%Y%m%d%H%M%S")
    return int(start_index)


# Get delta time by seconds
def get_tm_by_second(start_tm, seconds):
    start_time = datetime.strptime(start_tm, "%Y%m%d%H%M%S")
    start_time = start_time + timedelta(seconds=seconds)
    start_index = start_time.strftime("%Y%m%d%H%M%S")
    return int(start_index)


# Get delta time based on current
def get_current_tm():
    start_index = time.strftime("%Y%m%d%H%M%S",
                                time.localtime(int(time.time())))
    return int(start_index)


# Clean sets
def clear_sets(sets):
    for s in sets:
        s.clear()

def clearSymbolPos(client, position_value_map, position_amount_map, logger_error, moneyLimit = 50.0):
    update_positions(client, position_value_map, position_amount_map, logger_error)
    for key in position_value_map:
        try:
            if abs(position_value_map[key])<moneyLimit and abs(position_value_map[key]) > 0:
                money=position_value_map[key]
                pos=position_amount_map[key]
                params = {
                    'symbol': key,
                    'side': 'SELL' if money > 0 else 'BUY',
                    'type': 'MARKET',
                    'quantity': abs(pos),
                    "reduceOnly": "true",
                    # 'timeInForce': 'GTX'
                }
                # logger_trade.info(f"Sell: {symbol}, amount: {amount}, price: {price}")
                client.new_order(**params)
        except Exception as e:
            logger_error.info(str(e))
    return
            
            
    
    
# Trading process
def process_task(position_value_map, position_amount_map, api_key, api_secret, cfg, logger_trade, logger_error):
    # Signal file path
    base_path = cfg['signal']['base_path']
    pred_dir = cfg['signal']['pred_dir']

    # retry count for signal
    retry_count = cfg["retry_count"]
    # set for store orderId to cancel order
    cancel_order_id_sets = [set() for _ in range(retry_count)]

    # Init gorder
    ud.readuniverse(ud.g_data)
    gorder1 = gorders.GenerateOrders(cfg["gorder"])
    gorder2 = gorders.GenerateOrders(cfg["gorder2"])
    # Init client
    proxies = cfg["proxy"] if "proxy" in cfg else None
    client = UMFutures(key=api_key, secret=api_secret, proxies=proxies, timeout=cfg["timeout"])
    clients = [UMFutures(key=api_key, secret=api_secret, proxies=proxies,
                         timeout=cfg["timeout"]) for _ in range(10)]

    logger_trade.info(f"value map is: {position_value_map}")
    logger_trade.info(f"amount map is: {position_amount_map}")
    logger_trade.info(f"------ process cfg: {cfg} ------")

    # Start index of signal file(based on current timestamp)
    signal_index = time.strftime("%Y%m%d%H%M00", time.localtime(int(time.time() / (cfg["minutes"]*60) + cfg["waitSingalCnt"]) * (cfg["minutes"]*60 )))

    # Percision for price and quantity(3 means .00X)

    # Just one time
    # modify_position_side(client, logger_error)
    # leverage_brackets=client.leverage_brackets()
    # lbDict = {}
    # for lb in leverage_brackets:
    #     lbDict[lb["symbol"]]=lb
    # for key, value in position_value_map.items():
    #     lb=lbDict[key]
    #     targetItme = None
    #     moneylimit=200000
    #     for item in lb["brackets"]:
    #         if item['notionalCap'] >=moneylimit:
    #             targetItme=item
    #             break
    #     change_leverage(client, key, logger_error, item['initialLeverage'])

    signal_wait_max = 60 * cfg["signal_wait_max"]  # 60s*10min

    while True:
        try:
            symbol_set = set()
            close_position_set = set()
            clear_sets(cancel_order_id_sets)
            # Step1: Wait signal file and handle (before trade, once)
            file_path = f'{base_path}{pred_dir}/{signal_index}{signal_suffix}'
            logger_trade.info(f"------ Wait signal file: {file_path} ------")
            signal_wait_count = 0
            while not os.path.exists(file_path):
                if signal_wait_count > signal_wait_max:
                    signal_time = datetime.strptime(signal_index, "%Y%m%d%H%M%S")
                    signal_time = signal_time + time_delta
                    signal_index = signal_time.strftime("%Y%m%d%H%M00")
                    file_path = f'{base_path}{pred_dir}/{signal_index}{signal_suffix}'
                    notify("miss signal file, skip!")
                    logger_trade.info(f"------ Wait signal file: {file_path} ------")
                    signal_wait_count = 0
                    continue
                time.sleep(1)
                signal_wait_count = signal_wait_count + 1

            if int((int(signal_index) % 1000000) / 10000) in cfg["order_hours"]:
                gorder = gorder1
                logger_trade.info(f"------ gorder-1 : {signal_index} ------")
            else:
                gorder = gorder2
                logger_trade.info(f"------ gorder-2 : {signal_index} ------")
            alpha_map = {}
            pred_alpha_map = {}
            retry_count_signal = 2
            attempt_signal = 0
            while attempt_signal <= retry_count_signal:
                try:
                    cur_book=pd.read_csv(file_path)
                    cur_book["alpha"]=(cur_book["alpha"]-cur_book["alpha"].mean())/cur_book["alpha"].std()
                    cur_book["alpha"]=cur_book["alpha"].fillna(0)
                    for idx in range(cur_book.shape[0]):
                        row=cur_book.iloc[idx]
                        key = row["sid"]
                        value = round(float(row["bookw"]) * cfg["signal"]["money_scale"], 2)
                        alpha_map[key] = value
                        pred_alpha_map[key]=float(row["alphamin1"])
                    break
                except Exception as e:
                    logger_error.info(str(e))
                    attempt_signal += 1
                    if attempt_signal <= retry_count_signal:
                        time.sleep(2)
                    else:
                        notify("at-bad signal file, skip!")
                        continue

            # Step2: Restart algo order module (before trade, once)
            while True:
                if not update_positions(client, position_value_map, position_amount_map, logger_error):
                    break
            first_delta_map = {}
            sm_pairs = {}
            for key, value in position_value_map.items():
                if key in alpha_map:
                    alpha_value = alpha_map[key]
                    if alpha_value == 0:
                        close_position_set.add(key)
                    position_value = position_value_map[key]
                    real_delta = round(alpha_value - position_value, 2)
                    first_delta_map[key] = real_delta
                    sm_pairs[key] = (alpha_value, position_value, pred_alpha_map[key])
            logger_trade.info(f"First delta map: {first_delta_map}")
            logger_trade.info(f"First sm_pairs: {sm_pairs}")

            start_tm = get_current_tm()
            end_tm = get_tm_by_min(signal_index, cfg["trade_min"])
            gorder.restart(start_tm, end_tm, cfg["trade_delta"], sm_pairs)
            logger_trade.info(f"Restart algo order module: {start_tm} - {end_tm}\n gorder_smpairs:{gorder.smpairs}")

            canceled_sids=[]
            delta_map = {}
            for i in range(retry_count):
                # stop trading after end time
                if get_current_tm() > end_tm:
                    # Step9: Cancel all open orders
                    cancel_orders_in_parallel(clients, gorder.smpairs.keys(), logger_trade, logger_error)
                    break
                logger_trade.info(f"------ Round {i + 1} ------")
                # Step3: Update position data(True: fail)
                if update_positions(client, position_value_map, position_amount_map, logger_error):
                    continue
                filtered_values = {k: v for k, v in position_value_map.items() if abs(v) > 10.0}
                filtered_amounts = {k: v for k, v in position_amount_map.items() if abs(v) > 0.0001}
                logger_trade.info(f"Positions value:\n {filtered_values}\n")
                # logger_trade.info(f"Positions amount: {filtered_amounts}")

                # Step4: Random the symbols for fairness
                random_position_value_items = list(position_value_map.items())
                # random.shuffle(random_position_value_items)

                # Step5: Calculate delta_map
                delta_map = {}
                for key, value in random_position_value_items:
                    if key in alpha_map:
                        alpha_value = alpha_map[key]
                        position_value = position_value_map[key]
                        real_delta = round(alpha_value - position_value, 2)
                        if key in gorder.smpairs:
                            delta_map[key] = real_delta
                        symbol_set.add(key)

                logger_trade.info(f"Delta map: \n{delta_map}\n")

                # Step6: Get orders from algo order module
                current_tm = get_current_tm()
                tradeDeltaMs = cfg["trade_delta"]*1000
                next_tm = tools.tmu2i(int(tools.tmi2u(current_tm) / tradeDeltaMs) * tradeDeltaMs + tradeDeltaMs + 1000)
                orders = gorder.update_and_gorders(current_tm, position_value_map)
                logger_trade.info(f"{current_tm} - {next_tm} \n\n gorder_smpairs:{gorder.smpairs}")
                logger_trade.info(f"{current_tm} - {next_tm} \n\n order details:{orders}")
                completed_sids=gorder.get_completed_symbols()
                # Step6B: Handle order ids (queue)
                for order_index, order in enumerate(orders):
                    symbol, _, _, _, order_info = order
                    if symbol in canceled_sids:
                        canceled_sids.remove(symbol)
                    order_id = f"{PREFIX_ORDER_ID}-{i}-{order_index}-{symbol}"
                    cancel_order_id_sets[i].add(order_id)
                    order_info["order_id"] = order_id

                logger_trade.info(f"{current_tm} - {next_tm} completed_sids:{completed_sids}  canceled_sids:{canceled_sids}")
                completed_sids=[x for x in completed_sids if x not in canceled_sids]
                canceled_sids+=completed_sids
                if len(completed_sids) > 0:
                    cancel_orders_in_parallel(clients, completed_sids, logger_trade, logger_error)

                # Step7: Post maker orders
                post_orders_in_parallel(clients, orders, logger_trade, logger_error, price_precision_map,
                                        quantity_precision_map, next_tm)

                # Step9: Cancel pre round-10 orders
                if i >= (gorder.cancel_delay - 1):
                    cancel_order_by_id_in_parallel(clients, cancel_order_id_sets[i - (gorder.cancel_delay - 1)],
                                                   logger_trade, logger_error)

            # Step10: Update algo order module tratio (after trade, once)
            cancel_orders_in_parallel(clients, gorder.smpairs.keys(), logger_trade, logger_error)
            
            # Step11: Update next signal file index (after trade, once)
            signal_time = datetime.strptime(signal_index, "%Y%m%d%H%M%S")
            signal_time = signal_time + time_delta
            signal_index = signal_time.strftime("%Y%m%d%H%M00")

            # Step12: Print trade success ratio
            target_value = 0.0
            real_value = 0.0
            update_positions(client, position_value_map, position_amount_map, logger_error)
            logger_trade.info(
                f"\nend update_positions {position_value_map}.\n\n alpha_map:{alpha_map}\n\n gorder-keys:{gorder.smpairs.keys()}")
            for key in alpha_map.keys():
                value = alpha_map[key]
                f_value = abs(float(value))
                f_position_value = abs(float(position_value_map[key]))
                if (f_value > 5.0):
                    real_value = real_value + abs(f_value - f_position_value)
                    target_value = target_value + abs(f_value)
                    individual_ratio = f_position_value / f_value
                    logger_trade.info(
                        f"Key: {key}, Real money: {f_position_value}, Target money: {f_value}, Ratio: {individual_ratio}")
            if target_value > 0:
                deal_ratio = 1 - real_value / target_value
                logger_trade.info(f"Real money: {real_value}, Target money: {target_value}, Ratio: {deal_ratio}")

            logger_trade.info(f"\n\n--------calc trade completed ratio------")
            target_value = 0.0
            real_value = 0.0
            for key in gorder.smpairs.keys():
                value = first_delta_map[key]
                f_value = float(value)
                if abs(f_value) > 5:
                    f_position_value = float(delta_map[key])
                    target_value += abs(f_value)
                    real_value += abs(f_position_value)
                    individual_ratio = 1 - f_position_value / f_value
                    logger_trade.info(
                        f"Key: {key}, Delta money: {f_position_value}, Target money: {f_value}, alpha:{alpha_map[key]}  cur:{position_value_map[key]} Ratio: {individual_ratio}")
            if target_value > 0:
                deal_ratio = 1 - real_value / target_value
                logger_trade.info(f"all Delta money: {real_value}, Target money: {target_value}, Ratio: {deal_ratio}")
            clearSymbolPos(client, position_value_map, position_amount_map, logger_error, moneyLimit=cfg["moneyLimit"])

        except Exception as e:
            logger_error.info(str(e))
            # raise
            notify(str(e))



if __name__ == "__main__":
    items = list(available_token_map.items())
    # random.shuffle(items)
    # Init config file
    with open('./config/trade.yaml', 'r') as file:
        cfg = yaml.safe_load(file)

    # Signal file index delta
    time_delta = timedelta(minutes=cfg["minutes"])

    api_key = ''
    api_secret = ''
    for process_config in cfg['processes']:
        api_key = process_config['api_key']
        api_secret = process_config['api_secret']
        break
    # Init logger
    logfile_error = f"./logs/alpha-error.log"
    logger_error = get_logger(f"logger_error", logfile_error)
    logfile_trade = f"./logs/alpha-trade.log"
    logger_trade = get_logger(f"logger_trade", logfile_trade)
    # Start process
    position_value_map = copy.deepcopy(available_token_map)
    position_amount_map = copy.deepcopy(available_token_map)
    process_task(position_value_map, position_amount_map, api_key, api_secret, cfg, logger_trade, logger_error)



