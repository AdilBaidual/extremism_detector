import os
import json
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json, Tokenizer
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
from tensorflow.keras.optimizers import Adam
from core.utils import load_maxlen, save_maxlen, load_tokenizer, MODEL_PATH, TOKENIZER_PATH

DATA_PATH = "data/training_model.csv"

# --- Переобучение модели ---
def retrain_model(new_maxlen):
    df = pd.read_csv(DATA_PATH)

    tokenizer = Tokenizer(num_words=1000)
    tokenizer.fit_on_texts(df["text"])
    X_seq = tokenizer.texts_to_sequences(df["text"])
    X_pad = pad_sequences(X_seq, maxlen=new_maxlen)
    y = df["label"]

    model = Sequential([
        Embedding(input_dim=1000, output_dim=16),
        GlobalAveragePooling1D(),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(0.01), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_pad, y, epochs=10, verbose=0)

    os.makedirs("model", exist_ok=True)
    model.save(MODEL_PATH)
    with open(TOKENIZER_PATH, "w") as f:
        f.write(tokenizer.to_json())
    save_maxlen(new_maxlen)

# --- Анализ текста ---
def analyze_text(text):
    current_maxlen = load_maxlen()
    input_len = len(text.split())

    print(f"[INFO] исходный текст: {text.split()}")
    print(f"[INFO] Анализируем текст длинной ({input_len})")

    if input_len > current_maxlen:
        print(f"[INFO] Длина текста ({input_len}) > maxlen ({current_maxlen}). Переобучение модели...")
        retrain_model(input_len)
        current_maxlen = input_len

    # Загрузка обновлённой модели и токенизатора
    model = load_model(MODEL_PATH)
    tokenizer = load_tokenizer()

    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=current_maxlen)
    prob = model.predict(padded)[0][0]
    print(f"[INFO]TEST2: {prob}")
    label = "⚠️ Экстремистский контент" if prob >= 0.5 else "✅ Нейтральный контент"

    return {
        "probability": round(prob * 100, 2),
        "label": label
    }
