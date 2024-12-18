from binance.um_futures import UMFutures
import math
# Exchange Info
def get_exchange_info_price(client):
    try:
      data = client.exchange_info()
      symbol_map = {}
      for symbol_info in data.get("symbols", []):
          symbol = symbol_info.get("symbol")
          contractType = symbol_info.get("contractType")
          filters = symbol_info.get("filters")
          if symbol and symbol.endswith("USDT") and (contractType == "PERPETUAL"):
              for filter in filters:
                  if filter.get("filterType") == "PRICE_FILTER":
                    price_precision = filter.get("tickSize")
                    symbol_map[symbol] = price_precision
      power_of_tens = {key: 1 / float(value) for key, value in symbol_map.items()}
      exponents = {key: int(math.log10(power_of_tens[key])) for key in power_of_tens}

      print(exponents)
      print(symbol_map)
    except Exception as e:
      print(str(e))


def get_exchange_info_volume(client):
  try:
    data = client.exchange_info()
    symbol_map = {}
    for symbol_info in data.get("symbols", []):
      symbol = symbol_info.get("symbol")
      contractType = symbol_info.get("contractType")
      filters = symbol_info.get("filters")
      if symbol and symbol.endswith("USDT") and (contractType == "PERPETUAL"):
        for filter in filters:
          if filter.get("filterType") == "LOT_SIZE":
            price_precision = filter.get("stepSize")
            symbol_map[symbol] = price_precision
    power_of_tens = {key: 1 / float(value) for key, value in symbol_map.items()}
    exponents = {key: int(math.log10(power_of_tens[key])) for key in power_of_tens}
    print(exponents)
    print(symbol_map)
  except Exception as e:
    print(str(e))

def get_exchange_info_token(client):
  try:
    data = client.exchange_info()
    symbol_map = {}
    for symbol_info in data.get("symbols", []):
      symbol = symbol_info.get("symbol")
      contractType = symbol_info.get("contractType")
      if symbol and symbol.endswith("USDT") and (contractType == "PERPETUAL"):
        symbol_map[symbol] = 0.0
    print(symbol_map)
  except Exception as e:
    print(str(e))
def main():

  client = UMFutures(key='OugVodq3VF9syMAWI9iOGBwaIb4G0Pv3xVXSUOto24Oc4vImXaZBcpOyqL4uwkxF', secret='tNGt7sG9kMatPYqn1TcX8fcbds4jdgN21TY8TKD60rZRjuKxy2W6yZCLMuOCSyKw')

  get_exchange_info_token(client)

main()
