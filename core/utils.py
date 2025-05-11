import os
import json
from tensorflow.keras.preprocessing.text import tokenizer_from_json

MODEL_PATH = "model/extremism_model.h5"
TOKENIZER_PATH = "model/tokenizer.json"
MAXLEN_PATH = "model/maxlen.txt"

def load_maxlen():
    if not os.path.exists(MAXLEN_PATH):
        return 20
    with open(MAXLEN_PATH, "r") as f:
        return int(f.read().strip())

def save_maxlen(val):
    with open(MAXLEN_PATH, "w") as f:
        f.write(str(val))

def load_tokenizer():
    with open(TOKENIZER_PATH, "r", encoding="utf-8") as f:
        return tokenizer_from_json(f.read())

