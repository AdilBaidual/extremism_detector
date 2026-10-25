import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json

from core.config import BASE_DIR
from core.text import clean_text

logger = logging.getLogger(__name__)

VOCAB_SIZE = 1000
EMBEDDING_DIM = 16
HIDDEN_UNITS = 16
LEARNING_RATE = 0.01
DEFAULT_MAX_LEN = 50
DEFAULT_EPOCHS = 20
SEED = 42

MODEL_FILE = "extremism_model.keras"
TOKENIZER_FILE = "tokenizer.json"
META_FILE = "meta.json"
REQUIRED_COLUMNS = {"text", "label"}


@dataclass(frozen=True)
class ModelInfo:
    data_path: str
    samples: int
    vocabulary_size: int
    max_len: int
    train_accuracy: float


@dataclass(frozen=True)
class Detector:
    model: keras.Model
    tokenizer: Tokenizer
    info: ModelInfo

    def predict(self, texts: list[str]) -> list[float]:
        sequences = self.tokenizer.texts_to_sequences([clean_text(text) for text in texts])
        padded = pad_sequences(sequences, maxlen=self.info.max_len)
        probabilities = self.model(padded, training=False).numpy()[:, 0]
        return [float(probability) for probability in probabilities]


def read_dataset(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{data_path} is missing columns: {', '.join(sorted(missing))}")
    df = df.dropna(subset=list(REQUIRED_COLUMNS))
    if df.empty:
        raise ValueError(f"{data_path} has no rows to train on")
    return df


def build_model(max_len: int) -> keras.Model:
    model = keras.Sequential([
        keras.Input(shape=(max_len,)),
        keras.layers.Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM),
        keras.layers.GlobalMaxPooling1D(),
        keras.layers.Dense(HIDDEN_UNITS, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(LEARNING_RATE), loss="binary_crossentropy", metrics=["accuracy"])
    return model


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(BASE_DIR).as_posix()
    except ValueError:
        return path.name


def train(data_path: Path, model_dir: Path, max_len: int = DEFAULT_MAX_LEN, epochs: int = DEFAULT_EPOCHS) -> Detector:
    df = read_dataset(data_path)
    logger.info("Training model on %s (%d samples, %d epochs)", data_path, len(df), epochs)
    keras.utils.set_random_seed(SEED)

    texts = [clean_text(text) for text in df["text"].astype(str)]
    labels = df["label"].astype("float32").to_numpy()
    tokenizer = Tokenizer(num_words=VOCAB_SIZE)
    tokenizer.fit_on_texts(texts)
    padded = pad_sequences(tokenizer.texts_to_sequences(texts), maxlen=max_len)

    model = build_model(max_len)
    history = model.fit(padded, labels, epochs=epochs, verbose=0)

    info = ModelInfo(
        data_path=display_path(data_path),
        samples=len(df),
        vocabulary_size=min(len(tokenizer.word_index), VOCAB_SIZE - 1),
        max_len=max_len,
        train_accuracy=round(float(history.history["accuracy"][-1]), 4),
    )
    detector = Detector(model=model, tokenizer=tokenizer, info=info)
    save(detector, model_dir)
    return detector


def save(detector: Detector, model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    detector.model.save(model_dir / MODEL_FILE)
    (model_dir / TOKENIZER_FILE).write_text(detector.tokenizer.to_json(ensure_ascii=False), encoding="utf-8")
    (model_dir / META_FILE).write_text(json.dumps(asdict(detector.info), ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Model saved to %s", model_dir)


def load(model_dir: Path) -> Detector:
    model = keras.models.load_model(model_dir / MODEL_FILE)
    tokenizer = tokenizer_from_json((model_dir / TOKENIZER_FILE).read_text(encoding="utf-8"))
    info = ModelInfo(**json.loads((model_dir / META_FILE).read_text(encoding="utf-8")))
    return Detector(model=model, tokenizer=tokenizer, info=info)


def is_saved(model_dir: Path) -> bool:
    return all((model_dir / name).exists() for name in (MODEL_FILE, TOKENIZER_FILE, META_FILE))


def load_or_train(model_dir: Path, data_path: Path) -> Detector:
    if is_saved(model_dir):
        logger.info("Loading model from %s", model_dir)
        return load(model_dir)
    logger.warning("No saved model in %s, training a new one", model_dir)
    return train(data_path, model_dir)
