from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

EPSILON = 2**-52
RSI_PERIOD = 9
Z_PERIOD = 20

DATASET_NAME = "5_crypto_txt"
X_FEATURE_COLUMNS = ["vol_z", "vwap_z", "return_z", "rsi9_norm"]

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

MODEL_REGISTRY = {
    "1.1": {
        "file": "model1.1_weights.pth",
        "embedding_dims": 32,
        "hidden_layers": [2048, 2048, 1024],
        "dropout": None,
        "notes": "3-logit output",
    },
    "1.2": {
        "file": "model1.2_weights.pth",
        "embedding_dims": 32,
        "hidden_layers": [2048, 2048, 1024],
        "dropout": None,
        "notes": "3-logit output",
    },
    "1.3": {
        "file": "model1.3_weights.pth",
        "embedding_dims": 32,
        "hidden_layers": [2048, 2048, 1024],
        "dropout": None,
        "notes": "sigmoid",
    },
    "1.4": {
        "file": "model1.4_weights.pth",
        "embedding_dims": 8,
        "hidden_layers": [64, 32],
        "dropout": None,
        "notes": "sigmoid (bad)",
    },
    "1.5": {
        "file": "model1.5_weights.pth",
        "embedding_dims": 8,
        "hidden_layers": [64, 32],
        "dropout": None,
        "notes": "back to 3-logit cross entropy",
    },
    "1.6": {
        "file": "model1.6_weights.pth",
        "embedding_dims": 8,
        "hidden_layers": [128, 64],
        "dropout": None,
        "notes": "around same",
    },
    "2.0.1": {
        "file": "crypto_model2.0.1_weights.pth",
        "embedding_dims": 16,
        "hidden_layers": [64, 32, 16],
        "dropout": 0.2,
        "notes": "features: price/vwap, rsi9, return, volume_z",
    },
    "2.0.2": {
        "file": "crypto_model2.0.2_weights.pth",
        "embedding_dims": 8,
        "hidden_layers": [128, 64],
        "dropout": 0.5,
        "notes": "crypto",
    },
    "2.0.3": {
        "file": "crypto_model2.0.3_weights.pth",
        "embedding_dims": 8,
        "hidden_layers": [64, 32],
        "dropout": 0.2,
        "notes": "crypto",
    },
}

TRAIN_MODEL_VERSION = "2.0.3"
TRADER_MODEL_VERSION = "2.0.3"

TRAIN_MODEL_NAME = MODEL_REGISTRY[TRAIN_MODEL_VERSION]["file"]
TRADER_MODEL_NAME = MODEL_REGISTRY[TRADER_MODEL_VERSION]["file"]

TRAIN_MODEL_PATH = PATH_TO_MODELS / TRAIN_MODEL_NAME
TRADER_MODEL_PATH = PATH_TO_MODELS / TRADER_MODEL_NAME

def load_api_keys() -> tuple[str, str]:
    with open(API_KEY_FILE, "r", encoding="utf-8") as f:
        api_key = f.readline().strip()
    with open(SECRET_KEY_FILE, "r", encoding="utf-8") as f:
        secret_key = f.readline().strip()
    return api_key, secret_key
