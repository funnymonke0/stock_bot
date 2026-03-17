import pandas as pd
import numpy as np
import json

import torch
from StockModel import StockModel

from alpaca.data.live import StockDataStream, CryptoDataStream
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest, GetAssetsRequest, ClosePositionRequest
from alpaca.trading.enums import AssetClass
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.common.exceptions import APIError

from AppConfig import (
    EPSILON,
    RSI_PERIOD,
    Z_PERIOD,
    EMBEDDING_LOOKUP,
    TRADER_MODEL_VERSION,
    TRADER_MODEL_PATH,
    TRADER_MODEL_CONFIG,
    PIPELINE,
    load_api_keys,
)

TICKERS = []
class Trader():
    @staticmethod
    def to_stream_symbol(base_symbol: str) -> str:
        return f"{base_symbol}/USD"

    @staticmethod
    def to_model_symbol(stream_symbol: str) -> str:
        return stream_symbol.replace("/USD", ".V")

    @staticmethod
    def to_order_symbol(stream_symbol: str) -> str:
        return stream_symbol.replace("/USD", "USD")

    @staticmethod
    def to_base_symbol(symbol: str) -> str:
        return symbol.replace("/USD", "").replace("USD", "")

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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = StockModel(feature_size=len(TRADER_MODEL_CONFIG["features"]), embed_size=len(self.embedding_map), embedding_dims=TRADER_MODEL_CONFIG["embedding_dims"], output_size=TRADER_MODEL_CONFIG["output_size"], dropout=TRADER_MODEL_CONFIG["dropout"], hidden_layers=TRADER_MODEL_CONFIG["hidden_layers"])
        state_dict = torch.load(TRADER_MODEL_PATH, map_location=self.device,weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print(f"Using trader model version {TRADER_MODEL_VERSION}: {TRADER_MODEL_PATH.name}")

        API_KEY, SECRET_KEY = load_api_keys()
        self.stream = CryptoDataStream(API_KEY, SECRET_KEY, url_override=r"wss://stream.data.alpaca.markets/v1beta3/crypto/us")
        self.client = TradingClient(API_KEY, SECRET_KEY, paper=True)
        self.account = self.client.get_account()
        assets = self.client.get_all_assets(GetAssetsRequest(asset_class=AssetClass.CRYPTO))
        self.asset_map = {
            self.to_base_symbol(asset.symbol): asset
            for asset in assets
            if self.to_model_symbol(asset.symbol) in self.embedding_map.keys()
        }
        self.tickers = list(self.asset_map.keys())
        self.bars_window["symbol"] = pd.Series([f"{ticker}.V" for ticker in self.tickers], dtype="category")
        self.bid_ask_map = {self.to_stream_symbol(ticker): [-1, -1] for ticker in self.tickers} # Initialize bid-ask map with default values
        print(self.bars_window)
        
        print(f"Initialized trader with account status: {self.account.crypto_status}")

    
    def stream_data(self):
        # Connect to the Alpaca data stream and subscribe to the desired tickers
        print("Connecting to Alpaca data stream...")
        
        stream_symbols = [self.to_stream_symbol(ticker) for ticker in self.tickers]
        self.stream.subscribe_bars(self.handle_data, *stream_symbols)
        print("Subscribed to tickers: ", self.tickers)
        self.stream.subscribe_quotes(self.handle_quote, *stream_symbols)
        print("Subscribed to quotes: ", self.tickers)
        self.stream.run()


    def signal_generator(self, ticker_id:torch.Tensor, features:torch.Tensor) -> float:
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

        if features.size(1) != len(TRADER_MODEL_CONFIG["features"]):
            raise ValueError(f"Feature width mismatch: got {features.size(1)}, expected {len(TRADER_MODEL_CONFIG['features'])}")

        # signal = [0,0,0] # Default to hold
        signal = 0
        with torch.no_grad():
            signal = self.model(ticker_id, features).item() # Get the single prediction value
            # 
        # signal = torch.softmax(prediction, dim=1).squeeze().tolist()  #buy / sell / hold, 0, 1, 2 respectively. returns a list

        return signal # No signal for the first data point since we don't have a previous close price
    
    def process(self, data): #eventually we can use separate files that compute each model's features, and just import the correct one based on the model version. This way we can have different features for different models.
        stream_symbol = data.symbol
        model_symbol = self.to_model_symbol(stream_symbol)
        print(model_symbol)
        new_row = pd.DataFrame([data.__dict__])
        new_row.loc[:, "symbol"] = model_symbol
        self.bars_window = pd.concat([self.bars_window, new_row], ignore_index=True).dropna(subset=['symbol', 'timestamp',"open", "high", "low", "close", "volume"],inplace=False).groupby("symbol", sort=False, observed=True).tail(50) #21 minimum but keeping this in case of some faulty bars.
        
        symbol_group = self.bars_window.groupby('symbol', sort=False, observed=True).get_group(model_symbol).copy() # Get the group of rows corresponding to the symbol of the incoming data point
        
        #for each symbol...
        if len(symbol_group) < 21: # We need at least 21 data points to compute the features
            print(f"not enough data for symbol {model_symbol} yet. {len(symbol_group)}/21 data points available. skipping.")
            return None, None, None
        

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
        volatility = symbol_group['return'].rolling(window=Z_PERIOD).std().reset_index(0, drop=True)
        mean = symbol_group['return'].rolling(window=Z_PERIOD).mean().reset_index(0, drop=True)
        rsi9_norm = (symbol_group['rsi9'] - 50) / 50
        feature_row = pd.DataFrame([{
            "vol_z": vol_z.iloc[-1],
            "vwap_z": vwap_z.iloc[-1],
            "return_z": return_z.iloc[-1],
            "rsi9_norm": rsi9_norm.iloc[-1],
        }])
        feature_row = feature_row.replace([np.inf, -np.inf], np.nan)
        x_features = PIPELINE.fit_transform(feature_row)
        ticker_id = torch.tensor([self.embedding_map[model_symbol]], dtype=torch.int64) # batch size of 1
        features = torch.as_tensor(x_features, dtype=torch.float32).reshape(1, -1) # shape: (1, feature_size)
        return ticker_id, features, [volatility.iloc[-1], mean.iloc[-1]]
    
    async def handle_quote(self, data):
        self.bid_ask_map[data.symbol] = [float(data.bid_price), float(data.ask_price)]
        # print(f"quotes recieved {data.symbol}, bid: {data.bid_price}, ask: {data.ask_price}")
        # print(self.bid_ask_map)

    async def handle_data(self, data):
        print(f"data received: {data}")
        stream_symbol = data.symbol
        ticker_id, features, stats = self.process(data)
        if ticker_id is not None and features is not None:
            # print(f"Ticker ID: {ticker_id}, Features: {features}")
            signal = self.signal_generator(ticker_id=ticker_id, features=features)
            print(f"Generated signal: {signal} for data {data}")
            # with open(PATH_TO_PRECOMPUTE / "signal_log.txt", "a") as f:
            #     f.write(f"{data.timestamp}: {symbol} - Signal: {signal}\n")

            self.portfolio_management(signal, stats, stream_symbol)
            
        else:
            # print("Not enough data to generate features and signal yet.")
            pass



    def portfolio_management(self, signal, stats, symbol):
        base_symbol = self.to_base_symbol(symbol)
        order_symbol = self.to_order_symbol(symbol)
        bid, ask = self.bid_ask_map[symbol]
        
        if not self.asset_map[base_symbol].tradable:
            print(f"Asset {base_symbol} is not tradable. Cannot place order.")
            return 
        if ask <=0 or bid <=0:
            print(f"need to get quote data for {symbol}")
            return
        try:
            position = self.client.get_open_position(order_symbol)
            available_qty = float(position.qty_available)
        except APIError as e:
            position = None
            available_qty = 0
        # direction = signal[0]-signal[1]
        buying_power = float(self.account.non_marginable_buying_power)*0.95
        # limit = max(0.005 * buying_power * abs(direction), 10) # limit order size
        limit = max(0.05 * buying_power, 10) # limit order size
        simple_return = np.exp(signal*stats[0]+stats[1])-1
        print(f"signal: {signal}, volatility: {stats[0]}, mean: {stats[1]}, simple_return: {simple_return}, limit: {limit}, buying_power: {buying_power}, available_qty: {available_qty}")

        if available_qty > 0 and float(position.unrealized_plpc) > 0.005: # take profit if the unrealized profit exceeds 0.5% of the cost basis
            print(f"Taking profit on {symbol} with unrealized P/L of {float(position.unrealized_plpc)}% at price {bid}")
            try:
                self.client.close_position(order_symbol)
            except APIError as e:
                print(f"Error closing position for {symbol}: {e}")
        elif available_qty > 0 and float(position.unrealized_plpc) < -0.001: # stop loss if the unrealized loss exceeds 0.1% of the cost basis
            print(f"Stopping loss on {symbol} with unrealized P/L of {float(position.unrealized_plpc)}% at price {bid}")
            try:
                self.client.close_position(order_symbol)
            except APIError as e:
                print(f"Error closing position for {symbol}: {e}")
        # elif limit > 0 and signal[0] > 1/3 and direction > 0: # Buy signal (limit order placed slightly above current price)
        elif limit > 0 and simple_return > 0.005: # Buy signal, greater than 0.5% expected return
            limit_price = ask * (1 + 0.001)
            quantity = min(int(limit // limit_price), int(buying_power//limit_price)) if not self.asset_map[base_symbol].fractionable else min(limit/limit_price, buying_power/limit_price)
            if buying_power > limit_price * quantity and quantity > 0: #redundant
                print(f"Placing limit buy order for {symbol} at limit price {limit_price} with quantity {quantity}")
                order = LimitOrderRequest(
                    symbol=order_symbol,
                    limit_price = limit_price,
                    qty=quantity,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.GTC,
                    extended_hours = True
                )
                try:
                    self.client.submit_order(order)
                    print(f"Order submitted: {order}")
                except APIError as e:
                    print(f"Error submitting order: {e}")
        #no point in selling anyways if the p/l is affected by fees
        # elif position and signal[1] > 1/3 and direction < 0: # Sell signal (market order to sell the entire position to prevent getting stuck with a losing position, since limit orders can fail to execute if the price keeps dropping)
        #     limit_price = bid * (1 - 0.001)
        #     quantity = min(int(limit // limit_price), int(buying_power//limit_price)) if not self.asset_map[base_symbol].fractionable else min(limit/limit_price, buying_power/limit_price)
        #     quantity = min(quantity, max_sell)
        #     if buying_power > limit_price * quantity and quantity > 0: #redundant
        #         print(f"Taking profit on {symbol} with unrealized P/L of {float(position.unrealized_plpc)}% at limit price {limit_price}")
        #         order = LimitOrderRequest(
        #             symbol=order_symbol,
        #             limit_price=limit_price,
        #             qty=quantity,
        #             side=OrderSide.SELL,
        #             time_in_force=TimeInForce.GTC,
        #             extended_hours = True
        #         )
        
        else:
            print("Holding")

        
        # elif signal[1] > 1/3 and direction < 0: # only for short selling, which is currently disabled since not all assets are marginable and it adds extra risk. Sell signal (limit order placed slightly below current price to prevent getting stuck with a losing position)
        #     limit_price = bid * (1 - 0.001)
        #     # limit_price = bid
        #     quantity =min(int(limit // limit_price), int(buying_power//limit_price)) if not self.asset_map[symbol].fractionable else min(limit/limit_price, buying_power/limit_price)
        #     try:
        #         position = self.client.get_open_position(symbol.replace("/USD", "USD")) 
        #         current_qty = float(position.qty)
        #         quantity = min(current_qty, quantity)
        #         print(f"Placing limit buy order for {symbol} at limit price {limit_price} with quantity {quantity}")
        #         # print(f"Placing instant sell order for {symbol} at price {limit_price} with quantity {quantity}")
        #         order = LimitOrderRequest(
        #             symbol=symbol,
        #             limit_price = limit_price,
        #             qty=quantity,
        #             side=OrderSide.SELL,
        #             time_in_force=TimeInForce.GTC,
        #             extended_hours = True
        #         )
        #     except Exception as e:
        #         print(f"no open position for symbol: {symbol}")
        # else:
        #     print("Holding")

        


if __name__ == "__main__":
    trader = Trader()
    trader.stream_data()