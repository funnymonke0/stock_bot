from alpaca.trading.client import TradingClient
from pathlib import Path
import torch
from StockModel import StockModel
API_KEY = ""
SECRET_KEY = ""
PATH_TO_DATASETS = Path(r"datasets\5_us_txt\data\5 min\us")
PATH_TO_MODELS = Path(r"models")
MODEL_NAME = Path(r"model5.pth")
class Trader():
    def __init__(self):
        self.trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
        self.model = StockModel(hidden_layers=[64,64,32], embedding_dims=8)

    def load_model(self):
        state_dict = torch.load(Path(PATH_TO_MODELS, MODEL_NAME))
        self.model.load_state_dict(state_dict)
    
    def signal_generator(self):
        self.model.eval()
        