import pandas as pd
import numpy as np
import torch
import json
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

from pathlib import Path
from StockModel import StockModel


RELOAD = True

PATH_TO_DATASETS = Path(r"datasets\5_us_txt\data\5 min\us")
PATH_TO_MODELS = Path(r"models")
PATH_TO_PRECOMPUTE = Path(r"precompute_cache")
MODEL_NAME = Path(r"model1_weights.pth")

PATH_TO_PRECOMPUTE.mkdir(parents=True, exist_ok=True)
PATH_TO_MODELS.mkdir(parents=True, exist_ok=True)
PATH_TO_DATASETS.mkdir(parents=True, exist_ok=True)

print(f"Path to datasets: {PATH_TO_DATASETS}")

# SUBDIRS = [r"nasdaq etfs", r"nasdaq stocks\1", r"nasdaq stocks\2", r"nasdaq stocks\3", r"nyse etfs\1", r"nyse etfs\2", r"nyse stocks\1", r"nyse stocks\2", r"nysemkt stocks"]
SUBDIRS = [r"nysemkt stocks",r"nyse stocks\1", r"nyse stocks\2", r"nasdaq stocks\1", r"nasdaq stocks\2", r"nasdaq stocks\3"]

#hyperparams
BATCH_SIZE = 512
EPOCHS = 10
# X_FEATURE_COLUMNS = ["<OPEN>", "<CLOSE>", "<HIGH>", "<LOW>", "<VOL>",
#                     "time_sin", "time_cos", "day_sin", "day_cos",
#                     "month_sin", "month_cos", "day_of_month_sin", "day_of_month_cos",
#                     "start_of_quarter", "end_of_quarter", "start_of_year", "end_of_year", "quarter_of_year",
#                     "log_volume", "return", "range", "abs_return", "momentum", "rolling_std"]

#45.94%  embedding_dims = 32, hidden_layers = [1024, 512, 256] model6

#%46.52  embedding_dims = 32, hidden_layers = [2048, 1024, 512] model7

#%47.04  embedding_dims = 32, hidden_layers = [2048, 2048, 1024] model8

X_FEATURE_COLUMNS = ["norm_open", "norm_high", "norm_low", "log_volume", "momentum"]

# time normalization constants
OPEN = 9*60 + 30
CLOSE = 16*60

class TrainModel:
    def __init__(self):
        self.x_id_tensor = None
        self.x_tensor = None
        self.y_tensor = None
        self.testloader = None
        self.trainloader = None
        self.criterion = None
        self.optimizer = None
        self.dataframe = pd.DataFrame()
        self.model = None
        self.optim_model = None
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.load_tensors()
        if self.x_tensor is None or self.y_tensor is None or self.x_id_tensor is None or RELOAD:
            self.load_data()
            print("next")
            self.preprocess_data()
            print("next")
            self.format_tensors(save=True)
        print("next")
        self.prep_loaders()
        print("model ready to train.")
        

    def load_data(self):
        df_list = []
        total = len(SUBDIRS)
        counter = 1
        if (PATH_TO_DATASETS/r"datasets.parquet").exists():
            self.dataframe = pd.read_parquet(PATH_TO_DATASETS/r"datasets.parquet")
            return
        
        if not PATH_TO_DATASETS.exists() or total < 1:
            print("dataset not found")
            return
        for subdir in SUBDIRS:
            subdir_path = PATH_TO_DATASETS/subdir #join dataset path object and subdir path object
            if not subdir_path.exists():
                print("subdir not found")
                return 
            print(f"loading {subdir_path}; Set: {counter}/{total}")
            counter+=1
            for file_path in subdir_path.glob("*.txt"):
                if file_path.stat().st_size:
                    df = pd.read_csv(file_path,sep=',', engine="pyarrow")
                    df_list.append(df)
        if not len(df_list) > 0:
            print("no csv found or loaded")
            return
        self.dataframe = pd.concat(df_list, ignore_index=True)# basically, since the data is split into multiple files, we read each file and concatenate all the separate dataframes into a single dataframe ignoring their local indexes in the files.
        self.dataframe.dropna(inplace=True)
        self.dataframe.sort_values(by=["<TICKER>", "<DATE>", "<TIME>"], inplace=True) #sort it so everything is in order first by ticker, then date, then time

        if not (PATH_TO_DATASETS/r"datasets.parquet").exists():
            self.dataframe.to_parquet(PATH_TO_DATASETS/r"datasets.parquet", index=False) #save the concatenated dataframe as a parquet file for faster loading later
        print("load done.")


    def preprocess_data(self):
        if self.dataframe is None:
            print("no dataframe found")
            return
        print("Preprocessing data...")
        self.dataframe['<TICKER>'] = self.dataframe['<TICKER>'].astype('category')
        if not (PATH_TO_PRECOMPUTE/r"embedding_lookup.json").exists():
            embedding_lookup = dict(enumerate(self.dataframe['<TICKER>'].cat.categories))
            with open(PATH_TO_PRECOMPUTE/r"embedding_lookup.json", "w") as f:
                json.dump(embedding_lookup, f)
            print("embedding lookup saved.")

        self.dataframe["norm_open"]=np.log(self.dataframe["<OPEN>"]/self.dataframe["<CLOSE>"])
        self.dataframe["norm_high"]=np.log(self.dataframe["<HIGH>"]/self.dataframe["<CLOSE>"])
        self.dataframe["norm_low"]=np.log(self.dataframe["<LOW>"]/self.dataframe["<CLOSE>"])
        self.dataframe["log_volume"]=np.log(self.dataframe["<VOL>"])
        self.dataframe["momentum"] = np.log(self.dataframe["<CLOSE>"]/self.dataframe.groupby('<TICKER>', sort=False, observed=False)["<CLOSE>"].shift(1)) #interbar momentum, basically the return from the previous close to the current close. this is what we will be trying to predict the direction of, so it's not included in the features.
        #ticker generation
        self.dataframe["ticker_id"] = self.dataframe["<TICKER>"].cat.codes
        #label generation
        future_return = self.dataframe.groupby("<TICKER>")["momentum"].shift(-1) #shift the momentum; if the previous close was lower than the current close, the PREVIOUS entry was a buy.
        thresh = 0.001 #threshold for buy/sell signals, can be tuned as a hyperparameter. this means we only want to buy if the momentum is greater than 0.1% and sell if it's less than -0.1%, otherwise hold.
        self.dataframe["label"] = np.select(
            [future_return > thresh, future_return < -thresh],
            [0, 1],
            default=2
        ) #buy / sell / hold, 0, 1, 2 
        self.dataframe.dropna(inplace=True)
        self.dataframe = self.dataframe[["ticker_id"]+X_FEATURE_COLUMNS+["label"]] #we just pass all the relevant fields (ticker id, features, and labels) in 1 dataframe to parse later in format_tensors
        print("preprocess done.")


    def format_tensors(self, save = False): #does all the converting dataframe to tensor stuff. the rest of the values like feature size and embed size is in preprocess
        if self.dataframe.empty:
            print("Dataframe is empty. No data to format tensors.")
            return
        print("Formatting tensors...")
        print(self.dataframe.head())
        # Separate features and labels
        y_labels = self.dataframe["label"].to_numpy(dtype=np.int64)
        self.y_tensor = torch.from_numpy(y_labels).to(torch.int64)   
        
        x_ids = self.dataframe["ticker_id"].to_numpy(dtype=np.int64)
        self.x_id_tensor = torch.from_numpy(x_ids).to(torch.int64)  

        x_features = self.dataframe[X_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        self.x_tensor = torch.from_numpy(x_features).to(torch.float32)  
        if save:
            torch.save(self.x_tensor, Path(PATH_TO_PRECOMPUTE, r"x_tensor.pt"))
            torch.save(self.x_id_tensor, Path(PATH_TO_PRECOMPUTE,r"x_id_tensor.pt"))
            torch.save(self.y_tensor, Path(PATH_TO_PRECOMPUTE, r"y_tensor.pt"))
            print("saving formatted tensors.")
        print("formatting tensors done.")


    def load_tensors(self):
        if not(Path(PATH_TO_PRECOMPUTE / r"y_tensor.pt").exists() and Path(PATH_TO_PRECOMPUTE/ "x_tensor.pt").exists() and Path(PATH_TO_PRECOMPUTE , r"x_id_tensor.pt").exists()):
            print("no saved tensors found")
            return
        self.y_tensor = torch.load(PATH_TO_PRECOMPUTE / r"y_tensor.pt")
        self.x_tensor = torch.load(PATH_TO_PRECOMPUTE / r"x_tensor.pt")
        self.x_id_tensor = torch.load(PATH_TO_PRECOMPUTE / r"x_id_tensor.pt")
        print("saved tensors loaded successfully.")


    def prep_loaders(self):
        if self.x_tensor is None or self.y_tensor is None or self.x_id_tensor is None:
            print("Tensors are not properly initialized. Cannot train model.")
            return
        print("Preparing data loaders and model...")
        embed_size = self.x_id_tensor.max().item() + 1
        feature_size = len(X_FEATURE_COLUMNS)#number of features/inputs
        self.model = StockModel(feature_size=feature_size, embed_size=embed_size).to(self.device)
        self.optim_model = torch.compile(self.model)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr = 1e-3)
        dataset = TensorDataset(self.x_id_tensor, self.x_tensor, self.y_tensor)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        # Set num_workers>0 for better data loading
        self.trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=4, prefetch_factor=2, persistent_workers=True)
        self.testloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=4, persistent_workers=True)


    def training_loop(self):
        if self.criterion is None or self.optimizer is None or self.trainloader is None or self.optim_model is None:
            print("Data loaders or model not properly initialized. Cannot train model.")
            return
        print("training model...")
        self.optim_model.train()
        for epoch in range(EPOCHS):
            total_loss = 0.0
            counter = 1
            for x_id_batch, x_batch, y_batch in self.trainloader:
                x_id_batch = x_id_batch.to(self.device, non_blocking=True)
                x_batch = x_batch.to(self.device,non_blocking=True)
                y_batch = y_batch.to(self.device,non_blocking=True)
                self.optimizer.zero_grad()
                outputs = self.optim_model(x_id_batch, x_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                total_loss+=loss.item()
                counter+=1
            print(f"loss for epoch {epoch} / {EPOCHS}: {(total_loss/counter):.4f}")
        torch.save(self.model.state_dict(), PATH_TO_MODELS / MODEL_NAME)
        print("training complete. Model saved.")


    def evaluate(self):
        if self.testloader is None or self.optim_model is None:
            print("Test loader not properly initialized. Cannot evaluate model.")
            return
        print("testing model...")
        self.optim_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x_id_batch, x_batch, y_batch in self.testloader:
                x_id_batch = x_id_batch.to(self.device, non_blocking=True)
                x_batch = x_batch.to(self.device, non_blocking=True)
                y_batch = y_batch.to(self.device, non_blocking=True)
                outputs = self.optim_model(x_id_batch, x_batch) # forward pass
                predicted = torch.argmax(outputs, dim=1) #predicted class labels
                correct += (predicted == y_batch).sum().item() # count correct predictions
                total += y_batch.size(0) # total number of labels
        if total == 0:
            print('No test samples available to evaluate accuracy.')
        else:
            print(f'Accuracy of the model on the test data: {correct} correct predictions out of {total} total samples; {100 * correct / total:.2f}%')  


    def load_model(self):
        if not (PATH_TO_MODELS/MODEL_NAME).exists():
            print("model state dicts could not be loaded.")
            return
        state_dict = torch.load(PATH_TO_MODELS/MODEL_NAME)
        self.model.load_state_dict(state_dict)
        print("model state dicts loaded.")

if __name__ == "__main__":

    stock_model = TrainModel()
    # stock_model.training_loop()
    # stock_model.load_model()
    # stock_model.evaluate()
    

