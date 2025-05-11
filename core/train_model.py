import os
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
from tensorflow.keras.optimizers import Adam
from utils import save_maxlen, MODEL_PATH, TOKENIZER_PATH

def train_model(data_path, maxlen=50):
    print(data_path)
    df = pd.read_csv(data_path, quotechar='"')

    tokenizer = Tokenizer(num_words=1000)
    tokenizer.fit_on_texts(df["text"])
    X_seq = tokenizer.texts_to_sequences(df["text"])
    X_pad = pad_sequences(X_seq, maxlen=maxlen)
    y = df["label"]

    model = Sequential([
        Embedding(input_dim=1000, output_dim=16),
        GlobalAveragePooling1D(),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(0.01), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_pad, y, epochs=10, verbose=1)

    os.makedirs("model", exist_ok=True)
    model.save(MODEL_PATH)
    with open(TOKENIZER_PATH, "w") as f:
        f.write(tokenizer.to_json())
    save_maxlen(maxlen)

if __name__ == "__main__":
    train_model("data/training_model.csv", maxlen=50)
