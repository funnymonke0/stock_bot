# stock_bot
A machine learning model trained on historical market data to trade using papermoney
## Overview
* Data Ingestion: Fetches historical bars (price, volume) using Alpaca's market data endpoints.
* Signal Generation: Processes data through a simple MLP network to predict price movement directions.
* Order Execution: Places buy/sell orders automatically based on model confidence using the Alpaca Python SDK.
## Dependencies
* Pytorch
* Pandas
* Numpy
* Alpaca API
* Version: Python 3.x
## Prereqs
* An Alpaca Account (Paper trading is recommended for testing).
## Installation
1. **Clone Repo**
```
  git clone https://github.com/funnymonke0/stock_bot
```
2. **Install Dependencies**
```
  cd simple-isef-scraper
  pip install -r requirements.txt
```
3. **Try Out**
```
  cd src
  python3 Trader.py #runs the papertrade live trading side
  python3 TrainModel.py #trains the model on dataset
```

## Model
The core of this bot is a Multi-Layer Perceptron. It is trained on historical features
* Log Volume (20 bar normalized volume)
* Momentum/return (log normalized return from the previous bar)
* Log OHLV (log normalized open, high, low, close based on previous close)
* I chose to keep these inputs simple normalized ohlcv values, but will update at a later date

##Results
* check the results folder. Will add more concrete results to this page at a later date.
