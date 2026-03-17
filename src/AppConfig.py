from pathlib import Path
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FunctionTransformer, Pipeline
from sklearn.preprocessing import RobustScaler

def load_api_keys() -> tuple[str, str]:
    with open(API_KEY_FILE, "r", encoding="utf-8") as f:
        api_key = f.readline().strip()
    with open(SECRET_KEY_FILE, "r", encoding="utf-8") as f:
        secret_key = f.readline().strip()
    return api_key, secret_key

def winsorize(X):
    # Calculate limits for each column (axis=0)
    lower = np.percentile(X, 1, axis=0)
    upper = np.percentile(X, 99, axis=0)
    # Clip the array to these limits
    return np.clip(X, lower, upper)


BASE_DIR = Path(__file__).resolve().parents[1]

EPSILON = 2**-52
RSI_PERIOD = 9
Z_PERIOD = 20

DATASET_NAME = "5_crypto_txt"

PATH_TO_MODELS = BASE_DIR / "models"
PATH_TO_KEYS = BASE_DIR / "config"
PATH_TO_PRECOMPUTE = BASE_DIR / "precompute_cache"
PATH_TO_DATASETS = BASE_DIR / "datasets"

API_KEY_FILE = PATH_TO_KEYS / "public_key.txt"
SECRET_KEY_FILE = PATH_TO_KEYS / "secret_key.txt"

EMBEDDING_LOOKUP = PATH_TO_PRECOMPUTE / f"{DATASET_NAME}_embedding_lookup.json"
X_TENSOR = PATH_TO_PRECOMPUTE / f"{DATASET_NAME}_x_tensor.pt"
X_ID_TENSOR = PATH_TO_PRECOMPUTE / f"{DATASET_NAME}_x_id_tensor.pt"
Y_TENSOR = PATH_TO_PRECOMPUTE / f"{DATASET_NAME}_y_tensor.pt"
WEIGHTS_TENSOR = PATH_TO_PRECOMPUTE / f"{DATASET_NAME}_weights_tensor.pt"
SUBDIRS = [r"cryptocurrencies"]

#training specific config
TEST = False # Set to True to run a test training loop with a small subset of the data, False to run full training.
RELOAD = False # Set to True to reload preprocessed tensors if they exist, False to load raw data and preprocess again. 

PIPELINE = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value=0, keep_empty_features=True)), #for the nan values of the z scores
    ('winsorizer', FunctionTransformer(winsorize, validate=False)), #winsorize the features to remove outliers. this is important because we want to avoid the influence of outliers on the scaling. we use the 1st and 99th percentiles to compute the limits. this is important because we want to avoid the influence of outliers on the scaling.
    ('scaler', RobustScaler(quantile_range=(5, 95), unit_variance=True, with_centering=True, with_scaling=True, copy=True)) #robust scaler is used to scale the features to a similar range while being robust to outliers. we use the interquartile range (IQR) to scale the features. this is important because we want to avoid the influence of outliers on the scaling. we use the 25th and 75th percentiles to compute the IQR. this is important because we want to avoid the influence of outliers on the scaling.,
])


MODEL_REGISTRY = {
    "1.1": {
        "file": "model1.1_weights.pth",
        "features" : ["norm_open", "norm_high", "norm_low", "log_volume", "momentum"],
        "embedding_dims": 32,
        "hidden_layers": [2048, 2048, 1024],
        "dropout": 0.3,
        "output_size": 3,
        "notes": "3-logit output",
    },
    "1.2": {
        "file": "model1.2_weights.pth",
        "features" : ["norm_open", "norm_high", "norm_low", "log_volume", "momentum"],
        "embedding_dims": 32,
        "hidden_layers": [2048, 2048, 1024],
        "dropout": 0.3,
        "output_size": 3,
        "notes": "3-logit output",
    },
    "1.3": {
        "file": "model1.3_weights.pth",
        "features" : ["norm_open", "norm_high", "norm_low", "log_volume", "momentum"],
        "embedding_dims": 32,
        "hidden_layers": [2048, 2048, 1024],
        "dropout": 0.3,
        "output_size": 1,
        "notes": "sigmoid",
    },
    "1.4": {
        "file": "model1.4_weights.pth",
        "features" : ["norm_open", "norm_high", "norm_low", "log_volume", "momentum"],
        "embedding_dims": 8,
        "hidden_layers": [64, 32],
        "dropout": 0.3,
        "output_size": 1,
        "notes": "sigmoid (bad)",
    },
    "1.5": {
        "file": "model1.5_weights.pth",
        "features" : ["norm_open", "norm_high", "norm_low", "log_volume", "momentum"],
        "embedding_dims": 8,
        "hidden_layers": [64, 32],
        "dropout": 0.3,
        "output_size": 3,
        "notes": "back to 3-logit cross entropy",
    },
    "1.6": {
        "file": "model1.6_weights.pth",
        "features" : ["norm_open", "norm_high", "norm_low", "log_volume", "momentum"],
        "embedding_dims": 8,
        "hidden_layers": [128, 64],
        "dropout": 0.3,
        "output_size": 3,
        "notes": "around same",
    },
    "2.0.1": {
        "file": "crypto_model2.0.1_weights.pth",
        "features" : ["vol_z", "vwap_z", "return_z", "rsi9_norm"],
        "embedding_dims": 16,
        "hidden_layers": [64, 32, 16],
        "dropout": 0.2,
        "output_size": 3,
        "notes": "features: price/vwap, rsi9, return, volume_z",
    },
    "2.0.2": {
        "file": "crypto_model2.0.2_weights.pth",
        "features" : ["vol_z", "vwap_z", "return_z", "rsi9_norm"],
        "embedding_dims": 8,
        "hidden_layers": [128, 64],
        "dropout": 0.5,
        "output_size": 3,
        "notes": "crypto",
    },
    "2.0.3": {
        "file": "crypto_model2.0.3_weights.pth",
        "features" : ["vol_z", "vwap_z", "return_z", "rsi9_norm"],
        "embedding_dims": 8,
        "hidden_layers": [64, 32],
        "dropout": 0.2,
        "output_size": 3,
        "notes": "crypto",
    },
    "3.0.1": {
        "file": "crypto_model3.0.1_weights.pth",
        "features" : ['vol_z', 'vwap_z', 'return_z', 'rsi9_norm'],
        "embedding_dims": 8,
        "hidden_layers": [64, 32],
        "dropout": 0.2,
        "output_size": 1,
        "notes": "predicting price change (regression) instead of buy/sell/hold (classification). z score prediction of log return. features",
    },
    "3.0.2": {
        "file": "crypto_model3.0.2_weights.pth",
        "features" : ['vol_z', 'vwap_z', 'return_z', 'rsi9_norm'],
        "embedding_dims": 8,
        "hidden_layers": [128, 64, 32],
        "dropout": 0.2,
        "output_size": 1,
        "notes": "same as 3.0.1 but with more hidden layers and larger hidden sizes.\
            Test samples: 1632958 | Huber loss: 0.360818 | MAE: 0.697978",
    },

    "3.0.3": {
        "file": "crypto_model3.0.3_weights.pth",
        "features" : ['vol_z', 'vwap_z', 'return_z', 'rsi9_norm'],
        "embedding_dims": 8,
        "hidden_layers": [256, 128, 64],
        "dropout": 0.2,
        "output_size": 1,
        "notes": "testing larger hidden layers and more hidden layers. same features as 3.0.1 and 3.0.2.",
    },
}

TRAIN_MODEL_VERSION = "3.0.3"
TRADER_MODEL_VERSION = "3.0.1"

TRAIN_MODEL_NAME = MODEL_REGISTRY[TRAIN_MODEL_VERSION]["file"]
TRADER_MODEL_NAME = MODEL_REGISTRY[TRADER_MODEL_VERSION]["file"]

TRAIN_MODEL_PATH = PATH_TO_MODELS / TRAIN_MODEL_NAME
TRADER_MODEL_PATH = PATH_TO_MODELS / TRADER_MODEL_NAME

TRAIN_MODEL_CONFIG = MODEL_REGISTRY[TRAIN_MODEL_VERSION]
TRADER_MODEL_CONFIG = MODEL_REGISTRY[TRADER_MODEL_VERSION]


