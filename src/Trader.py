from pathlib import Path
from math import log
import json
from alpaca.data.live import StockDataStream
from alpaca.trading.client import TradingClient
import torch
from StockModel import StockModel

PATH_TO_DATASETS = r"datasets\5_us_txt\data\5 min\us"

PATH_TO_MODELS = Path(r"models")
PATH_TO_KEYS = Path(r"config")
API_KEY_FILE = PATH_TO_KEYS / r"public_key.txt"
SECRET_KEY_FILE = PATH_TO_KEYS / r"secret_key.txt"
MODEL_NAME = r"model1_weights.pth"
PATH_TO_PRECOMPUTE = Path(r"precompute_cache")
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "UNH"]


class Trader():
    def __init__(self):
        self.prev_close = None
        self.embedding_map = {}
        with open(PATH_TO_PRECOMPUTE/'embedding_lookup.json', 'r') as f:
            self.embedding_map = json.load(f)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x_id_tensor = torch.load(PATH_TO_PRECOMPUTE / r"x_id_tensor.pt")
        embed_size = x_id_tensor.max().item() + 1
        self.model = StockModel(feature_size=5, embed_size=embed_size)
        state_dict = torch.load(PATH_TO_MODELS / MODEL_NAME, map_location=self.device,weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model = torch.compile(self.model)
        self.model.eval()

        with open(API_KEY_FILE, "r") as f:
            API_KEY = f.readline()
        with open(SECRET_KEY_FILE, "r") as f:
            SECRET_KEY = f.readline()
        self.stream = StockDataStream(API_KEY, SECRET_KEY)
        self.trading_client = TradingClient(API_KEY, SECRET_KEY)

    
    def stream_data(self):
        # Connect to the Alpaca data stream and subscribe to the desired tickers
        print("Connecting to Alpaca data stream...")
        self.stream.subscribe_bars(self.preprocess, *TICKERS, )
        print("Subscribed to tickers: ", TICKERS)
        self.stream.run()


    def signal_generator(self, symbol, features):
        ticker_id = self.embedding_map[symbol+".US"]
        x_idx = torch.tensor([ticker_id], dtype=torch.long)
        x_features = torch.tensor(features, dtype=torch.float32)
        with torch.no_grad():
            prediction = self.model(x_idx.to(self.device), x_features.to(self.device))
        signal = torch.argmax(prediction).item()
        return signal

    async def preprocess(self, data):
        print(f"Received data for {data.symbol}: open={data.open}, high={data.high}, low={data.low}, close={data.close}, volume={data.volume}")
        ohlcv = [data.open, data.high, data.low, data.close, data.volume]
        
        if self.prev_close is not None:
            norm_open = log(ohlcv[0] / ohlcv[3])  # Normalize by close price
            norm_high = log(ohlcv[1] / ohlcv[3]) # Normalize by close price
            norm_low = log(ohlcv[2] / ohlcv[3]) # Normalize by close price
            norm_volume = log(ohlcv[4])
            momentum = log(ohlcv[3] / self.prev_close) # Normalize by close price
            features = [norm_open, norm_high, norm_low, norm_volume, momentum]
            signal = self.signal_generator(data.symbol, features)
            print(f"Signal for {data.symbol}: {signal}")
        self.prev_close = ohlcv[3]  # Update previous close price
            


if __name__ == "__main__":
    trader = Trader()

    trader.stream_data()