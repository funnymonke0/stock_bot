import pandas as pd
import numpy as np
import torch
import json
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from pathlib import Path
from StockModel import StockModel
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
import matplotlib.pyplot as plt

# Put in config at some point
RELOAD = True # Set to True to reload preprocessed tensors if they exist, False to load raw data and preprocess again. 
EPSILON = 2**-52 # the most tiniest didyblud
RSI_PERIOD = 9
Z_PERIOD = 20
# PATH_TO_DATASETS = Path(r"datasets\5_us_txt\data\5 min")
PATH_TO_DATASETS = Path(r"datasets")
PATH_TO_MODELS = Path(r"models")
PATH_TO_PRECOMPUTE = Path(r"precompute_cache")
MODEL_NAME = r"crypto_model2.0.3_weights.pth"
MODEL_PATH = PATH_TO_MODELS / MODEL_NAME
# DATASET_NAME = r"us"
DATASET_NAME = r"5_crypto_txt"
DATASET = PATH_TO_DATASETS/DATASET_NAME
PARQUET = PATH_TO_DATASETS/(DATASET_NAME+r".parquet")
EMBEDDING_LOOKUP = PATH_TO_PRECOMPUTE/(DATASET_NAME+r"_embedding_lookup.json")
X_TENSOR = PATH_TO_PRECOMPUTE/ (DATASET_NAME+r"_x_tensor.pt")
X_ID_TENSOR = PATH_TO_PRECOMPUTE/ (DATASET_NAME+r"_x_id_tensor.pt")
Y_TENSOR = PATH_TO_PRECOMPUTE/ (DATASET_NAME+r"_y_tensor.pt")
WEIGHTS_TENSOR = PATH_TO_PRECOMPUTE/ (DATASET_NAME+r"_weights_tensor.pt")

PATH_TO_PRECOMPUTE.mkdir(parents=True, exist_ok=True)
PATH_TO_MODELS.mkdir(parents=True, exist_ok=True)
PATH_TO_DATASETS.mkdir(parents=True, exist_ok=True)

print(f"Loading from: {DATASET}")

# SUBDIRS = [r"nysemkt stocks",r"nyse stocks\1", r"nyse stocks\2", r"nasdaq stocks\1", r"nasdaq stocks\2", r"nasdaq stocks\3"]
SUBDIRS = [r"cryptocurrencies"]

#hyperparams
BATCH_SIZE = 512
EPOCHS = 20

# X_FEATURE_COLUMNS = ["norm_open", "norm_high", "norm_low", "log_volume", "momentum"]
X_FEATURE_COLUMNS = ["vol_z", "vwap_z", "return_z", "rsi9_norm"]

BUY_THRESH = 0 #threshold for buy signals, can be tuned as a hyperparameter. this means we only want to buy if the momentum is greater than 0.5%, otherwise hold.
SELL_THRESH = 0 #threshold for sell signals, can be tuned as a hyperparameter. this means we only want to sell if the momentum is less than 0%, otherwise hold. we realistically should not hold it even for small fluctuations since it can keep building up in a slow decline.

# time normalization constants
# OPEN = 9*60 + 30
# CLOSE = 16*60

#1.1 embedding_dims = 32, hidden_layers = [2048, 2048, 1024] 3 logit output
#1.2 embedding_dims = 32, hidden_layers = [2048, 2048, 1024] 3 logit output
#1.3 embedding_dims = 32, hidden_layers = [2048, 2048, 1024], sigmoid now
#1.4 embedding_dims = 8, hidden_layers = [64, 32],  sigmoid bad
#1.5 embedding_dims = 8, hidden_layers = [64, 32], back to 3 logit cross entropy (its worse than guessing apparently)
#1.6 embedding_dims = 8, hidden_layers = [128, 64], around same
#2.0.1 embedding_dims = 16, hidden_layers = [64,32,16], dropout=0.2, [ (Price/VWAP_rolling), (RSI_9), (Log_Return_1m), (Volume_ZScore) ]
#2.0.2 embedding_dims = 8, hidden_layers = [128, 64], dropout = 0.5
#2.0.3 embedding_dims = 8, hidden_layers = [64, 32], dropout = 0.2


class TrainModel:
    def __init__(self):
        self.x_id_tensor = None
        self.x_tensor = None
        self.y_tensor = None
        self.testloader = None
        self.trainloader = None
        self.optimizer = None
        self.weights_tensor = None
        self.model = None
        self.criterion = None
        self.optim_model = None
        self.dataframe = None

        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        if not RELOAD:
            self.load_tensors()
        if self.x_tensor is None or self.y_tensor is None or self.x_id_tensor is None:
            self.load_data()
            self.preprocess_data()
        self.prep_loaders()
        print("model ready to train.")
        

    def load_data(self):#loads all the csv files into a single dataframe, does some basic cleaning, and saves it as a parquet file for faster loading later. also checks if the parquet file already exists before doing all that to save time on subsequent runs.
        df_list = []
        total = len(SUBDIRS)
        counter = 1
        
        if (PARQUET).exists():
            print(f"parquet file found at {PARQUET}, loading parquet...")
            self.dataframe = pd.read_parquet(PARQUET)
            return
        
        print(f"parquet file not found at {PARQUET}, loading raw data from {DATASET}...")
        if not DATASET.exists() or total < 1:
            print("dataset not found")
            return

        for subdir in SUBDIRS:
            subdir_path = DATASET/subdir #join dataset path object and subdir path object
            print (f"checking for subdir at {subdir_path}...")
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
            print("no csvs found or loaded")
            return
        self.dataframe = pd.DataFrame()
        self.dataframe = pd.concat(df_list, ignore_index=True)# basically, since the data is split into multiple files, we read each file and concatenate all the separate dataframes into a single dataframe ignoring their local indexes in the files.
        self.dataframe.dropna(inplace=True)
        self.dataframe.sort_values(by=["<TICKER>", "<DATE>", "<TIME>"], inplace=True) #sort it so everything is in order first by ticker, then date, then time

        if not (PARQUET).exists():
            self.dataframe.to_parquet(PARQUET, index=False) #save the concatenated dataframe as a parquet file for faster loading later
        print("load done.")


    def preprocess_data(self):#does all the feature engineering and label generation. also generates the ticker embeddings and saves the mapping to a json file for later use in the trader. this is where we do all the groupby operations since we need to do them on a per-ticker basis, so we do them all here and then just format tensors later without worrying about groupbys.
        if self.dataframe is None:
            print("no dataframe found")
            return
        print("Preprocessing data...")
        self.dataframe['<TICKER>'] = self.dataframe['<TICKER>'].astype('category')
        if not (EMBEDDING_LOOKUP).exists():
            embedding_lookup = dict(enumerate(self.dataframe['<TICKER>'].cat.categories))
            with open(EMBEDDING_LOOKUP, "w") as f:
                json.dump(embedding_lookup, f)
            print("embedding lookup saved.")
        
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value=0, keep_empty_features=True)), #for the nan values of the z scores
            ('squasher', FunctionTransformer(np.tanh)),
        ])

        #VWAP price volume
        g_by = lambda df: df.groupby(self.dataframe['<TICKER>'], sort=False, observed=False)
        g = g_by(self.dataframe)
        tpv = self.dataframe['<VOL>']*((self.dataframe['<HIGH>'] + self.dataframe['<LOW>'] + self.dataframe['<CLOSE>']) / 3)
        sum_tpv = g_by(tpv).shift(1).rolling(window=Z_PERIOD).sum().reset_index(level=0, drop=True)
        sum_v = g['<VOL>'].shift(1).rolling(window=Z_PERIOD).sum().reset_index(level=0, drop=True)
        self.dataframe['p_vwap'] = np.log((self.dataframe['<CLOSE>']/(sum_tpv / (sum_v + EPSILON))).reset_index(0, drop=True)) #price/vwap

        #speed of change
        diff = g['<CLOSE>'].diff()
        gain = diff.clip(lower=0)
        loss = -diff.clip(upper=0)
        ma_gain = g_by(gain).ewm(alpha=1/RSI_PERIOD, adjust=False).mean().reset_index(level=0, drop=True)
        ma_loss = g_by(loss).ewm(alpha=1/RSI_PERIOD, adjust=False).mean().reset_index(level=0, drop=True)
        self.dataframe['rsi9'] = 100 - (100 / (1 + ma_gain / (ma_loss + EPSILON)))
        
        #return
        self.dataframe["return"] = np.log((self.dataframe["<CLOSE>"]+EPSILON)/(g["<CLOSE>"].shift(1)+EPSILON)) #interbar momentum, basically the return from the previous close to the current close. this is what we will be trying to predict the direction of, so it's not included in the features.

        #final
        self.dataframe['vol_z'] = (self.dataframe['<VOL>'] - g_by(self.dataframe['<VOL>']).rolling(Z_PERIOD).mean().reset_index(0, drop=True)) / g_by(self.dataframe['<VOL>']).rolling(Z_PERIOD).std().reset_index(0, drop=True)
        self.dataframe['vwap_z'] = (self.dataframe['p_vwap'] - g_by(self.dataframe['p_vwap']).rolling(Z_PERIOD).mean().reset_index(0, drop=True)) / g_by(self.dataframe['p_vwap']).rolling(Z_PERIOD).std().reset_index(0, drop=True)
        self.dataframe['return_z'] = (self.dataframe['return'] - g_by(self.dataframe['return']).rolling(Z_PERIOD).mean().reset_index(0, drop=True)) / g_by(self.dataframe['return']).rolling(Z_PERIOD).std().reset_index(0, drop=True)
        self.dataframe['rsi9_norm'] = (self.dataframe['rsi9'] - 50) / 50
        #ticker features
        self.dataframe["ticker_id"] = self.dataframe["<TICKER>"].cat.codes
        
        #og features
        # norm = g["<CLOSE>"].shift(1)
        # self.dataframe["norm_open"]=np.log((self.dataframe["<OPEN>"]+EPSILON)/(norm+EPSILON)) 
        # self.dataframe["norm_high"]=np.log((self.dataframe["<HIGH>"]+EPSILON)/(norm+EPSILON))
        # self.dataframe["norm_low"]=np.log((self.dataframe["<LOW>"]+EPSILON)/(norm+EPSILON))
        # v_ma = g['<VOL>'] .shift(1).rolling(window=20).mean().reset_index(level=0, drop=True)
        # self.dataframe['log_volume'] = np.log((self.dataframe['<VOL>'] + 1) / (v_ma + 1))
        # self.dataframe["momentum"] = np.log((self.dataframe["<CLOSE>"]+EPSILON)/(g["<CLOSE>"].shift(1)+EPSILON)) #interbar momentum, basically the return from the previous close to the current close. this is what we will be trying to predict the direction of, so it's not included in the features.
        
        self.dataframe['future_return'] = g['return'].shift(-1)
        volatility = g['future_return'].rolling(window=Z_PERIOD).std().reset_index(0, drop=True)
        #label generation
        self.dataframe["label"] = np.select(
            [self.dataframe['future_return'] > volatility*1.0, self.dataframe['future_return'] < volatility*-1.0],
            [0, 1],
            default=2
        ) #buy / sell / hold, 0, 1, 2

        self.dataframe.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.dataframe.dropna(subset=['vol_z', 'vwap_z', 'return_z','label'], inplace=True)
        x_features = pipeline.fit_transform(self.dataframe[['vol_z', 'vwap_z', 'return_z', 'rsi9_norm']])
        print(x_features)
        counts = self.dataframe["label"].value_counts().sort_index()
        print(f"Label distribution:\n{counts}")
        #accounts for imbalances in data by weighting each output differently (so if there are a bunch of sell signals, it wont just spam sell and get like 100% accuracy but no actual learning)
        total = counts.sum()
        raw_weights =[total/counts[i] for i in range(3)]
        mean_weight = sum(raw_weights) / len(raw_weights)
        final_weights = [(raw_weights[i]/mean_weight) for i in range(3)]
        print(final_weights)
        self.weights_tensor = torch.as_tensor(final_weights, dtype=torch.float32)
        self.y_tensor = torch.as_tensor(self.dataframe["label"].values, dtype=torch.int64)
        self.x_id_tensor = torch.as_tensor(self.dataframe["ticker_id"].values, dtype=torch.int64)
        self.x_tensor = torch.as_tensor(x_features, dtype=torch.float32)
        torch.save(self.x_tensor, X_TENSOR)
        torch.save(self.x_id_tensor, X_ID_TENSOR)
        torch.save(self.y_tensor, Y_TENSOR)
        torch.save(self.weights_tensor, WEIGHTS_TENSOR)
        print("preprocess done. Tensors saved for future use.")

    def load_tensors(self):#loads the preformatted tensors if they exist to save time on subsequent runs. if they don't exist, it will load the data and preprocess it and format the tensors and save them for next time.
        if not((X_TENSOR).exists() and Path(Y_TENSOR).exists() and Path(X_ID_TENSOR).exists() and Path(WEIGHTS_TENSOR).exists()):
            print("no saved tensors found")
            return
        self.y_tensor = torch.load(Y_TENSOR)
        self.x_tensor = torch.load(X_TENSOR)
        self.x_id_tensor = torch.load(X_ID_TENSOR)
        self.weights_tensor = torch.load(WEIGHTS_TENSOR)
        print("saved tensors loaded successfully.")


    def prep_loaders(self):# prepares the data loaders and model for training. it also sets up the loss function and optimizer. we do this in a separate function so that we can easily reload the model and just prepare the loaders without having to reload and preprocess the data again if we want to continue training or evaluate.
        if self.x_tensor is None or self.y_tensor is None or self.x_id_tensor is None or self.weights_tensor is None:
            print("Tensors are not properly initialized. Cannot train model.")
            return
        print("Preparing data loaders and model...")
        
        embed_size = self.x_id_tensor.max().item() + 1
        feature_size = len(X_FEATURE_COLUMNS)#number of features/inputs
        self.model = StockModel(feature_size=feature_size, embed_size=embed_size).to(self.device)
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    m.bias.data.fill_(0.01)
        self.model.apply(init_weights)
        self.optim_model = torch.compile(self.model, mode="default")
        self.weights_tensor = self.weights_tensor.to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=self.weights_tensor)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr = 1e-3)
        dataset = TensorDataset(self.x_id_tensor, self.x_tensor, self.y_tensor)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        # Set num_workers>0 for better data loading
        self.trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=4, prefetch_factor=2, persistent_workers=True)
        self.testloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=4, persistent_workers=True)

    
    def training_loop(self):# trains the model using the prepared data loaders and model. it also saves the model state dict after training is complete.
        if self.optimizer is None or self.trainloader is None or self.optim_model is None or self.criterion is None:
            print("Data loaders or model not properly initialized. Cannot train model.")
            return
        print("training model...")
        train_losses, train_accuracies = [], []
        self.optim_model.train()
        for epoch in range(EPOCHS):
            
            correct, total, total_loss = 0, 0, 0
            for x_id_batch, x_batch, y_batch in self.trainloader:
                x_id_batch = x_id_batch.to(self.device, non_blocking=True)
                x_batch = x_batch.to(self.device,non_blocking=True)
                y_batch = y_batch.to(self.device,non_blocking=True)
                self.optimizer.zero_grad(set_to_none=True)
                outputs = self.optim_model(x_id_batch, x_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                total_loss+=loss.item()
                total += y_batch.size(0)
                predicted = torch.argmax(outputs, dim=1)
                correct += (predicted == y_batch).sum().item()
            epoch_loss = total_loss / len(self.trainloader)
            epoch_accuracy = correct / total if total else 0.0
            train_losses.append(epoch_loss)
            train_accuracies.append(epoch_accuracy)
            print(f"loss for epoch {epoch} / {EPOCHS}: {epoch_loss:.4f}")
        torch.save(self.model.state_dict(), MODEL_PATH)
        print("training complete. Model saved.")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(train_losses, label="train")
        ax1.set_title("Loss")
        ax1.legend()
        ax2.plot(train_accuracies, label="train")
        ax2.set_title("Accuracy")
        ax2.legend()
        plt.tight_layout()
        plt.show()

    def evaluate(self):# evaluates the model on the test set and prints the accuracy. we do this in a separate function so that we can easily reload the model and just evaluate without having to reload and preprocess the data again if we want to continue training or evaluate.
        if self.testloader is None or self.optim_model is None:
            print("Test loader not properly initialized. Cannot evaluate model.")
            return
        print("testing model...")
        self.optim_model.eval()
        correct, total = 0, 0
        all_y_true = []
        all_y_pred = []
        with torch.no_grad():
            for x_id_batch, x_batch, y_batch in self.testloader:
                x_id_batch = x_id_batch.to(self.device, non_blocking=True)
                x_batch = x_batch.to(self.device, non_blocking=True)
                y_batch = y_batch.to(self.device, non_blocking=True)
                outputs = self.optim_model(x_id_batch, x_batch) # forward pass
                
                predicted = torch.argmax(outputs, dim=1) #predicted class labels
                all_y_true.extend(y_batch.cpu().detach().numpy().tolist())
                all_y_pred.extend(predicted.cpu().detach().numpy().tolist())
                correct += (predicted == y_batch).sum().item() # count correct predictions
                total += y_batch.size(0) # total number of labels
        if total == 0:
            print('No test samples available to evaluate accuracy.')
        else:
            cm = confusion_matrix(all_y_true, all_y_pred, labels=[0, 1, 2])
            print(f'Accuracy of the model on the test data: {correct} correct predictions out of {total} total samples; {100 * correct / total:.2f}%')  
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Buy', 'Sell', 'Hold'])
            disp.plot()
            plt.show()
        


    def load_model(self):# loads the model state dict from the saved file. we do this in a separate function so that we can easily reload the model and just prepare the loaders without having to reload and preprocess the data again if we want to continue training or evaluate.
        if not (MODEL_PATH).exists():
            print("model state dicts could not be loaded.")
            return
        state_dict = torch.load(MODEL_PATH)
        self.model.load_state_dict(state_dict)
        print("model state dicts loaded.")


if __name__ == "__main__":

    stock_model = TrainModel()
    stock_model.training_loop()
    # stock_model.load_model()
    stock_model.evaluate()
    

