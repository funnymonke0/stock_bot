from pathlib import Path
import pandas as pd
import numpy as np
import json

import torch
from StockModel import StockModel

from alpaca.data.live import StockDataStream, CryptoDataStream
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest, GetAssetsRequest
from alpaca.trading.enums import AssetClass
from alpaca.trading.enums import OrderSide, TimeInForce

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer

EPSILON = 2**-52 # the most tiniest didyblud
PATH_TO_MODELS = Path(r"models")
PATH_TO_KEYS = Path(r"config")
API_KEY_FILE = PATH_TO_KEYS / r"public_key.txt"
SECRET_KEY_FILE = PATH_TO_KEYS / r"secret_key.txt"
MODEL_NAME = r"crypto_model2.0.3_weights.pth"
MODEL_PATH = PATH_TO_MODELS / MODEL_NAME
PATH_TO_PRECOMPUTE = Path(r"precompute_cache")
PATH_TO_DATASETS = Path(r"datasets")
DATASET_NAME = r"5_crypto_txt"
EMBEDDING_LOOKUP = PATH_TO_PRECOMPUTE/(DATASET_NAME+r"_embedding_lookup.json")
X_TENSOR = PATH_TO_PRECOMPUTE/ (DATASET_NAME+r"_x_tensor.pt")
X_ID_TENSOR = PATH_TO_PRECOMPUTE/ (DATASET_NAME+r"_x_id_tensor.pt")
Y_TENSOR = PATH_TO_PRECOMPUTE/ (DATASET_NAME+r"_y_tensor.pt")

RSI_PERIOD = 9
Z_PERIOD = 20

# TICKERS = ["BTC", "ETH", "BCH", "LTC", "UNI", "SOL", "AVAX", "XRP", "DOGE", "USDC", "USDT","BCH","AAVE", "DOT", "LINK", "CRV", "XTZ", "YFI"] # List of tickers to subscribe to
X_FEATURE_COLUMNS = ["vol_z", "vwap_z", "return_z", "rsi9_norm"]
TICKERS = []
class Trader():
    def __init__(self):
        self.tickers = TICKERS
        self.embedding_map = {}
        self.dtypes = {
            'symbol': 'category',
            'timestamp': 'datetime64[ns]',
            'open': 'float64',
            'high': 'float64',
            'low': 'float64',
            'close': 'float64',
            'volume': 'int32',
            "trade_count": "int32",
            "vwap" : "float32"
        }
        self.bars_window = pd.DataFrame({k: pd.Series(dtype=v) for k, v in self.dtypes.items()})

        with open(EMBEDDING_LOOKUP, 'r') as f:
            self.embedding_map = json.load(f)
            self.embedding_map = {v:int(k) for k,v in self.embedding_map.items()}# Invert the embedding map to get a mapping from ticker symbols to their corresponding IDs
        print(self.embedding_map)
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
        self.cached_buying_power = float(self.account.buying_power)
        all_assets = self.client.get_all_assets(GetAssetsRequest(asset_class=AssetClass.CRYPTO))
        self.tradable = [asset.symbol for asset in all_assets if asset.tradable]
        self.frac = [asset.symbol for asset in all_assets if asset.fractionable]
        self.tickers = [ticker.replace("/USD", "") for ticker in self.tradable if ticker.replace("/USD", ".V") in self.embedding_map.keys()]
        self.bars_window["symbol"] = pd.Series(self.tickers, dtype="category")
        print(self.bars_window)
        self.bid_ask_map = {ticker+"/USD":[-1,-1] for ticker in self.tickers}
        
        print(f"Initialized trader with account status: {self.account.crypto_status}")

    
    def stream_data(self):
        # Connect to the Alpaca data stream and subscribe to the desired tickers
        print("Connecting to Alpaca data stream...")
        
        self.stream.subscribe_bars(self.handle_data, *[ticker+"/USD" for ticker in self.tickers])
        print("Subscribed to tickers: ", self.tickers)
        self.stream.subscribe_quotes(self.handle_quote, *[ticker+"/USD" for ticker in self.tickers])
        print("Subscribed to quotes: ", self.tickers)
        self.stream.run()


    def signal_generator(self, ticker_id:torch.Tensor, features:torch.Tensor):
        ticker_id = ticker_id.to(torch.int64).reshape(-1)
        features = features.to(torch.float32)
        if features.dim() == 1:
            features = features.reshape(1, -1)
        elif features.dim() > 2:
            features = features.reshape(features.size(0), -1)

        if features.size(0) != ticker_id.size(0):
            if features.size(0) == 1:
                features = features.expand(ticker_id.size(0), -1)
            elif ticker_id.size(0) == 1:
                ticker_id = ticker_id.expand(features.size(0))
            else:
                raise ValueError(f"Batch mismatch: ticker_id={ticker_id.size(0)}, features={features.size(0)}")

        if features.size(1) != len(X_FEATURE_COLUMNS):
            raise ValueError(f"Feature width mismatch: got {features.size(1)}, expected {len(X_FEATURE_COLUMNS)}")

        signal = [0,0,0] # Default to hold
        with torch.no_grad():
            prediction = self.model(ticker_id.to(self.device), features.to(self.device))
        signal = torch.softmax(prediction, dim=1).squeeze().tolist()  #buy / sell / hold, 0, 1, 2 respectively. returns a list

        return signal # No signal for the first data point since we don't have a previous close price
    
    def process(self, data):
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value=0, keep_empty_features=True)), #for the nan values of the z scores
            ('squasher', FunctionTransformer(np.tanh)),
        ])

        data.symbol = data.symbol.replace("/USD", ".V") # Remove the "/USD" suffix from the symbol to match the ticker format in the embedding map
        print(data.symbol)
        new_row = pd.DataFrame([data.__dict__])
        self.bars_window = pd.concat([self.bars_window, new_row], ignore_index=True).dropna(subset=['symbol', 'timestamp',"open", "high", "low", "close", "volume"],inplace=False).groupby("symbol", sort=False, observed=True).tail(50) #21 minimum but keeping this in case of some faulty bars.
        
        symbol_group = self.bars_window.groupby('symbol', sort=False, observed=True).get_group(data.symbol).copy() # Get the group of rows corresponding to the symbol of the incoming data point
        
        #for each symbol...
        if len(symbol_group) < 21: # We need at least 21 data points to compute the features
            print(f"not enough data for symbol {data.symbol} yet. {len(symbol_group)}/21 data points available. skipping.")
            return None, None
        
        #VWAP price volume

        tpv = symbol_group['volume']*((symbol_group['high'] + symbol_group['low'] + symbol_group['close']) / 3)
        sum_tpv = tpv.shift(1).rolling(window=Z_PERIOD).sum().reset_index(level=0, drop=True)
        sum_v = symbol_group['volume'].shift(1).rolling(window=Z_PERIOD).sum().reset_index(level=0, drop=True)
        symbol_group.loc[:, 'p_vwap'] = np.log((symbol_group['close']/(sum_tpv / (sum_v + EPSILON))).reset_index(0, drop=True)) #price/vwap

        #speed of change
        diff = symbol_group['close'].diff()
        gain = diff.clip(lower=0)
        loss = -diff.clip(upper=0)
        ma_gain = gain.ewm(alpha=1/RSI_PERIOD, adjust=False).mean().reset_index(level=0, drop=True)
        ma_loss = loss.ewm(alpha=1/RSI_PERIOD, adjust=False).mean().reset_index(level=0, drop=True)
        symbol_group.loc[:, 'rsi9'] = 100 - (100 / (1 + ma_gain / (ma_loss + EPSILON)))
        
        #return
        symbol_group.loc[:, "return"] = np.log((symbol_group["close"]+EPSILON)/(symbol_group["close"].shift(1)+EPSILON)) #interbar momentum, basically the return from the previous close to the current close. this is what we will be trying to predict the direction of, so it's not included in the features.

        #final
        vol_z = (symbol_group['volume'] - symbol_group['volume'].rolling(Z_PERIOD).mean().reset_index(0, drop=True)) / symbol_group['volume'].rolling(Z_PERIOD).std().reset_index(0, drop=True)
        vwap_z = (symbol_group['p_vwap'] - symbol_group['p_vwap'].rolling(Z_PERIOD).mean().reset_index(0, drop=True)) / symbol_group['p_vwap'].rolling(Z_PERIOD).std().reset_index(0, drop=True)
        return_z = (symbol_group['return'] - symbol_group['return'].rolling(Z_PERIOD).mean().reset_index(0, drop=True)) / symbol_group['return'].rolling(Z_PERIOD).std().reset_index(0, drop=True)
        rsi9_norm = (symbol_group['rsi9'] - 50) / 50
        feature_row = pd.DataFrame([{
            "vol_z": vol_z.iloc[-1],
            "vwap_z": vwap_z.iloc[-1],
            "return_z": return_z.iloc[-1],
            "rsi9_norm": rsi9_norm.iloc[-1],
        }])
        feature_row = feature_row.replace([np.inf, -np.inf], np.nan)
        x_features = pipeline.fit_transform(feature_row)
        ticker_id = torch.tensor([self.embedding_map[data.symbol]], dtype=torch.int64) # batch size of 1
        features = torch.as_tensor(x_features, dtype=torch.float32).reshape(1, -1) # shape: (1, feature_size)
        return ticker_id, features
    
    async def handle_quote(self, data):
        self.bid_ask_map[data.symbol] = [data.bid_price, data.ask_price]
        # print(f"quotes recieved {data.symbol}, bid: {data.bid_price}, ask: {data.ask_price}")
        # print(self.bid_ask_map)

    async def handle_data(self, data):
        print(f"data received: {data}")
        symbol = data.symbol
        ticker_id, features = self.process(data)
        if ticker_id is not None and features is not None:
            # print(f"Ticker ID: {ticker_id}, Features: {features}")
            signal = self.signal_generator(ticker_id=ticker_id, features=features)
            print(f"Generated signal: {signal} for data {data}")
            with open(PATH_TO_PRECOMPUTE / "signal_log.txt", "a") as f:
                f.write(f"{data.timestamp}: {symbol} - Signal: {signal}\n")
            self.cached_buying_power = self.account.buying_power

            self.portfolio_management(signal, symbol)

        else:
            # print("Not enough data to generate features and signal yet.")
            pass

    def portfolio_management(self, signal, symbol):
        print(symbol)
        if symbol not in self.tradable:
            print(f"Asset {symbol} is not tradable.")
            return
        
        ask, bid = self.bid_ask_map[symbol]
        if ask <=0 or bid <=0:
            print(f"need to get quote data for {symbol}")
            return
        
        direction = signal[0]-signal[1]
        buying_power = float(self.cached_buying_power)*0.95
        limit = max(0.005 * buying_power * abs(direction), 10) # per trade limit
        
        order = None
        if signal[0] > 1/3 and direction > 0: # Buy signal (limit order placed slightly above current price)
            limit_price = ask * (1 + 0.001)
            quantity =min(int(limit // limit_price), int(buying_power//limit_price)) if symbol not in self.frac else min(limit/limit_price, buying_power/limit_price)
            if buying_power > limit_price * quantity: #extra
                print(f"Placing limit buy order for {symbol} at limit price {limit_price} with quantity {quantity}")
                order = LimitOrderRequest(
                    symbol=symbol,
                    limit_price = limit_price,
                    qty=quantity,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.GTC,
                    extended_hours = True
                )

        elif signal[1] > 1/3 and direction < 0: # Sell signal
            limit_price = bid * (1 - 0.001)
            # limit_price = bid
            quantity =min(int(limit // limit_price), int(buying_power//limit_price)) if symbol not in self.frac else min(limit/limit_price, buying_power/limit_price)
            try:
                position = self.client.get_open_position(symbol.replace("/USD", "USD")) 
                current_qty = float(position.qty)
                quantity = min(current_qty, quantity)
                print(f"Placing limit buy order for {symbol} at limit price {limit_price} with quantity {quantity}")
                # print(f"Placing instant sell order for {symbol} at price {limit_price} with quantity {quantity}")
                order = LimitOrderRequest(
                    symbol=symbol,
                    limit_price = limit_price,
                    qty=quantity,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.GTC,
                    extended_hours = True
                )
            except Exception as e:
                print(f"no open position for symbol: {symbol}")
        else:
            print("Holding")
        #add something to sell if the current close is lower, regardless of signal, to prevent getting stuck with a losing position. maybe a market order if the price drops more than 1% below the last close or something like that. also maybe add a stop loss to the limit orders.
        if order is not None:
            try:
                self.client.submit_order(order)
                print(f"Order submitted: {order}")
            except Exception as e:
                print(f"Error submitting order: {e}")


if __name__ == "__main__":
    trader = Trader()
    trader.stream_data()