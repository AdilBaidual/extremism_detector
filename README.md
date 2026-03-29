# Extremism Detector

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python)](https://python.org/)
[![Flask](https://img.shields.io/badge/Flask-API-000000?style=flat-square&logo=flask)](https://flask.palletsprojects.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-DL-FF6F00?style=flat-square&logo=tensorflow)](https://tensorflow.org/)

Сервис для определения экстремистского контента в тексте. Использует нейросеть на Keras с Embedding слоями. Есть REST API.

## Что умеет

- Определять экстремизм в тексте
- Возвращать уверенность предсказания
- REST API
- Обработка текста с лемматизацией

## Структура

```
extremism_detector/
├── app.py                   # Flask API
├── analyzer.py              # ML инференс
├── requirements.txt
│
├── core/                    # Ядро
│   ├── preprocessor.py      # Обработка текста
│   ├── train_model.py       # Обучение
│   └── text_utils.py        # Утилиты
│
├── model/                   # Модели
│   ├── tokenizer.json
│   └── extremism_model.h5
│
└── data/                    # Датасеты
    ├── train.csv
    └── test.csv
```

## Архитектура нейросети

```python
Sequential([
    Embedding(10000, 128, input_length=100),
    GlobalMaxPooling1D(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
```

**Препроцессинг:** удаляем URL, упоминания, спецсимволы → лемматизация → токенизация → padding.

## Запуск

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Обучить модель
python core/train_model.py

# Запустить API
python app.py
```

API на `http://localhost:5000`

## API

### POST `/analyze`

Запрос:
```json
{
  "text": "Текст для анализа"
}
```

Ответ:
```json
{
  "is_extremist": true,
  "confidence": 0.92,
  "processed_text": "..."
}
```

### GET `/health`

Проверка работоспособности.

## Метрики

| Метрика | Значение |
|---------|----------|
| Accuracy | 0.91 |
| Precision | 0.89 |
| Recall | 0.88 |
| F1 | 0.88 |

## Пример использования

```python
import requests

response = requests.post('http://localhost:5000/analyze', json={
    'text': 'Текст для проверки'
})
result = response.json()
print(f"Экстремистский: {result['is_extremist']}")
print(f"Уверенность: {result['confidence']}")
```

## Зависимости

```
tensorflow>=2.13.0
flask>=2.3.0
numpy>=1.24.0
pandas>=2.0.0
pymorphy2>=0.9.0
```
