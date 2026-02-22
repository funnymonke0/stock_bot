from pathlib import Path
import pandas as pd
import numpy as np
import json

import torch
from StockModel import StockModel

from alpaca.data.live import StockDataStream, CryptoDataStream
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

EPSILON = 2**-52 # the most tiniest didyblud
PATH_TO_MODELS = Path(r"models")
PATH_TO_KEYS = Path(r"config")
API_KEY_FILE = PATH_TO_KEYS / r"public_key.txt"
SECRET_KEY_FILE = PATH_TO_KEYS / r"secret_key.txt"
MODEL_NAME = r"crypto_model1.6_weights.pth"
MODEL_PATH = PATH_TO_MODELS / MODEL_NAME
PATH_TO_PRECOMPUTE = Path(r"precompute_cache")
PATH_TO_DATASETS = Path(r"datasets")
DATASET_NAME = r"5_crypto_txt"
EMBEDDING_LOOKUP = PATH_TO_PRECOMPUTE/(DATASET_NAME+r"_embedding_lookup.json")
X_TENSOR = PATH_TO_PRECOMPUTE/ (DATASET_NAME+r"_x_tensor.pt")
X_ID_TENSOR = PATH_TO_PRECOMPUTE/ (DATASET_NAME+r"_x_id_tensor.pt")
Y_TENSOR = PATH_TO_PRECOMPUTE/ (DATASET_NAME+r"_y_tensor.pt")

TICKERS = ["BTC", "ETH", "BCH", "LTC", "UNI", "SOL", "AVAX", "XRP", "DOGE", "USDC", "USDT","BCH","AAVE", "DOT", "LINK", "CRV", "XTZ", "YFI"] # List of tickers to subscribe to
X_FEATURE_COLUMNS = ["norm_open", "norm_high", "norm_low", "log_volume", "momentum"]


class Trader():
    def __init__(self):
        self.bid_ask_map = {ticker+"/USD":[-1,-1] for ticker in TICKERS}
        self.embedding_map = {}
        self.bars_window = pd.DataFrame() # Initialize an empty DataFrame to store incoming bars data
        with open(EMBEDDING_LOOKUP, 'r') as f:
            self.embedding_map = json.load(f)
            self.embedding_map = {v:int(k) for k,v in self.embedding_map.items()}# Invert the embedding map to get a mapping from ticker symbols to their corresponding IDs
            print(f"Loaded embedding map: {self.embedding_map}")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x_id_tensor = torch.load(X_ID_TENSOR)
        embed_size = x_id_tensor.max().item() + 1
        self.model = StockModel(feature_size=len(X_FEATURE_COLUMNS), embed_size=embed_size)
        state_dict = torch.load(MODEL_PATH, map_location=self.device,weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model = torch.compile(self.model)
        self.model.eval()

        with open(API_KEY_FILE, "r") as f:
            API_KEY = f.readline()
        with open(SECRET_KEY_FILE, "r") as f:
            SECRET_KEY = f.readline()
        self.stream = CryptoDataStream(API_KEY, SECRET_KEY, url_override=r"wss://stream.data.alpaca.markets/v1beta3/crypto/us")
        self.client = TradingClient(API_KEY, SECRET_KEY, paper=True)
        self.account = self.client.get_account()
        print(f"Initialized trader with account status: {self.account.crypto_status}")

    
    def stream_data(self):
        # Connect to the Alpaca data stream and subscribe to the desired tickers
        print("Connecting to Alpaca data stream...")
        
        self.stream.subscribe_bars(self.handle_data, *[ticker+"/USD" for ticker in TICKERS])
        print("Subscribed to tickers: ", TICKERS)
        self.stream.subscribe_quotes(self.handle_quote, *[ticker+"/USD" for ticker in TICKERS])
        print("Subscribed to quotes: ", TICKERS)
        self.stream.run()


    def signal_generator(self, ticker_id:torch.Tensor, features:torch.Tensor):
        signal = [0,0,0] # Default to hold
        with torch.no_grad():
            prediction = self.model(ticker_id.to(self.device), features.to(self.device))
        signal = torch.softmax(prediction, dim=1).squeeze().tolist()  #buy / sell / hold, 0, 1, 2 respectively. returns a list

        return signal # No signal for the first data point since we don't have a previous close price
    
    def process(self, data):
        new_row = pd.DataFrame([data.__dict__])
        self.bars_window = pd.concat([self.bars_window, new_row]).sort_values(['symbol', 'timestamp']).dropna(inplace=False).iloc[-(50*len(TICKERS)):] 
        self.bars_window['symbol'] = self.bars_window['symbol'].astype('category')
        if data.symbol not in self.bars_window['symbol'].cat.categories:
            print(f"Symbol {data.symbol} not in bars window categories yet. Skipping processing for this data point.")
            return None, None
        symbol_group = self.bars_window.groupby('symbol', sort=False, observed=False).get_group(data.symbol) # Get the group of rows corresponding to the symbol of the incoming data point
        
        #for each symbol...
        if len(symbol_group) < 3: # We need at least 3 data points to compute the features
            return None, None
        
        norm = symbol_group['close'].iloc[-2] 
        v_ma = symbol_group['volume'].shift(1).rolling(window=20, min_periods=1).mean().iloc[-1]

        norm_open = np.log((data.open+EPSILON)/(norm+EPSILON))
        norm_high = np.log((data.high+EPSILON)/(norm+EPSILON))
        norm_low = np.log((data.low+EPSILON)/(norm+EPSILON))
        log_volume = np.log((data.volume + 1) / (v_ma + 1))
        momentum = np.log((data.close+EPSILON)/(norm+EPSILON))
        ticker_id = torch.as_tensor(self.embedding_map[data.symbol], dtype=torch.int64).unsqueeze(0) # Get the ticker ID from the embedding map and add a batch dimension
        features = torch.as_tensor([norm_open, norm_high, norm_low, log_volume, momentum], dtype=torch.float32).unsqueeze(0) # Create a tensor of features and add a batch dimension
        return ticker_id, features
    
    async def handle_quote(self, data):
        self.bid_ask_map[data.symbol] = [data.bid_price, data.ask_price]
        # print(f"quotes recieved {data.symbol}, bid: {data.bid_price}, ask: {data.ask_price}")
        # print(self.bid_ask_map)

    async def handle_data(self, data):
        # print(f"data received: {data}")
        data.symbol = data.symbol.replace("/USD", ".V") # Remove the "/USD" suffix from the symbol to match the ticker format in the embedding map
        ticker_id, features = self.process(data)
        if ticker_id is not None and features is not None:
            # print(f"Ticker ID: {ticker_id}, Features: {features}")
            signal = self.signal_generator(ticker_id=ticker_id, features=features)
            print(f"Generated signal: {signal} for data {data}")
            with open(PATH_TO_PRECOMPUTE / "signal_log.txt", "a") as f:
                f.write(f"{data.timestamp}: {data.symbol} - Signal: {signal}\n")
            order = self.portfolio_management(signal, data.symbol)
            if order is not None:
                try:
                    self.client.submit_order(order)
                    print(f"Order submitted: {order}")
                except Exception as e:
                    print(f"Error submitting order: {e}")
        else:
            # print("Not enough data to generate features and signal yet.")
            pass

    def portfolio_management(self, signal, symbol):

        symbol = symbol.replace(".V", "/USD") # Convert symbol to match Alpaca's format for crypto trading
        print("test1")
        asset = self.client.get_asset(symbol) # Check if the asset is tradable
        if not asset.tradable:
            print(f"Asset {symbol} is not tradable.")
            return
        ask, bid = self.bid_ask_map[symbol]
        if ask == -1 or bid ==-1:
            print(f"need to get quote data for {symbol}")
            return
        print("test2")
        buying_power = float(self.account.buying_power)*0.99
        limit = 0.05 * buying_power # per trade limit
        direction = signal[0]-signal[2]
        qty = lambda limit_price: min(int(limit*abs(direction) // limit_price), int(buying_power//limit_price)) if not asset.fractionable else min(limit*abs(direction)/limit_price, buying_power/limit_price)

        order = None
        if direction > 0: # Buy signal (limit order placed slightly above current price)
            limit_price = ask * (1 + 0.001)
            quantity =qty(limit_price)
            if buying_power > limit_price * quantity: #extra
                print(f"Placing limit buy order for {symbol} at limit price {limit_price} with quantity {quantity}")
                order = LimitOrderRequest(
                    symbol=symbol,
                    limit_price = limit_price,
                    qty=quantity,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.GTC # Cancel if not filled by end of day
                )
        else: # Sell signal
            if symbol in self.client.get_all_positions():
                limit_price = bid * (1 - 0.001)
                quantity =qty(limit_price)
                print(f"Placing limit sell order for {symbol} at price {limit_price} with quantity {quantity}")
                qty = min(self.client.get_open_position(symbol).qty, quantity)
                order = LimitOrderRequest(
                    symbol=symbol,
                    limit_price = limit_price,
                    qty=quantity,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.GTC # Cancel if not filled by end of day
                )

        return order
            
        
            


if __name__ == "__main__":
    trader = Trader()
    trader.stream_data()