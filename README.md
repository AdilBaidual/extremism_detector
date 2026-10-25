# Extremism Detector

[![CI](https://github.com/AdilBaidual/extremism_detector/actions/workflows/ci.yml/badge.svg)](https://github.com/AdilBaidual/extremism_detector/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python)](https://python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?style=flat-square&logo=tensorflow)](https://tensorflow.org/)

REST API для проверки русскоязычных текстов на экстремистский контент. Сервис принимает текст и возвращает вероятность того, что он экстремистский, и итоговый вердикт по порогу. Умеет проверять сразу пачку текстов и разбирать текст по предложениям, чтобы найти подозрительное.

**Для кого:** модераторы и разработчики площадок с пользовательским контентом (форумы, комментарии, чаты), которым нужна предварительная автоматическая проверка сообщений перед ручной модерацией.

## Содержание

- [Маршруты API](#маршруты-api)
- [Быстрый старт через Docker](#быстрый-старт-через-docker)
- [Локальный запуск](#локальный-запуск)
- [Переменные окружения](#переменные-окружения)
- [Модель](#модель)
- [Тесты](#тесты)
- [CI](#ci)
- [Диагностика](#диагностика)
- [Структура проекта](#структура-проекта)

## Маршруты API

| Метод | Путь | Что делает |
|-------|------|------------|
| GET | `/` | Название, версия и ссылки на `/docs` и `/health` |
| GET | `/health` | Проверка работоспособности: `200` если модель загружена, иначе `503` |
| GET | `/version` | Название и версия приложения |
| POST | `/analyze` | Проверка одного текста |
| POST | `/analyze/batch` | Проверка списка текстов (до 100 штук) со сводкой |
| POST | `/analyze/sentences` | Оценка всего текста и каждого предложения отдельно |
| GET | `/model` | На каких данных обучена модель, её параметры и текущий порог |

Все маршруты `/analyze*` принимают необязательный query-параметр `threshold` (от 0 до 1). Он переопределяет порог из переменной `THRESHOLD` для конкретного запроса.

Интерактивная документация Swagger: `http://127.0.0.1:8000/docs`.

### Примеры запросов

```bash
curl -X POST http://127.0.0.1:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Свергнем власть силой!"}'
```

```json
{"probability": 0.7258, "is_extremist": true, "threshold": 0.5}
```

Тот же текст со своим порогом:

```bash
curl -X POST "http://127.0.0.1:8000/analyze?threshold=0.9" \
  -H "Content-Type: application/json" \
  -d '{"text": "Свергнем власть силой!"}'
```

Пачка текстов:

```bash
curl -X POST http://127.0.0.1:8000/analyze/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Призываем к борьбе с режимом!", "Погуляем по парку, весна!"]}'
```

```json
{
  "threshold": 0.5,
  "total": 2,
  "extremist_count": 1,
  "results": [
    {"probability": 0.7207, "is_extremist": true},
    {"probability": 0.0588, "is_extremist": false}
  ]
}
```

Разбор по предложениям:

```bash
curl -X POST http://127.0.0.1:8000/analyze/sentences \
  -H "Content-Type: application/json" \
  -d '{"text": "Сегодня хорошая погода. Свергнем власть силой! Погуляем по парку."}'
```

Ошибки валидации (пустой текст, текст длиннее 5000 символов, `threshold` вне диапазона 0–1) возвращают `422` с описанием проблемы.

## Быстрый старт через Docker

Нужен только Docker. Модель обучается на этапе сборки образа, контейнер сразу готов к работе.

```bash
git clone https://github.com/AdilBaidual/extremism_detector.git
cd extremism_detector
docker compose up --build
```

Первая сборка занимает несколько минут: скачивается TensorFlow, образ весит около 2 ГБ. После строки `Service is ready` откройте `http://127.0.0.1:8000/docs`.

Остановить: `Ctrl+C` или `docker compose down`.

Если рядом с `docker-compose.yml` лежит файл `.env`, переменные из него передаются в контейнер.

## Локальный запуск

Нужен Python 3.11.

### 1. Окружение и зависимости

```bash
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Настройки

```bash
cp .env.example .env               # Windows: copy .env.example .env
```

Шаг необязательный: без `.env` используются значения по умолчанию.

### 3. Запуск

```bash
python app_final.py
```

Или напрямую через uvicorn (обычный режим, без `--reload`):

```bash
uvicorn app_final:app --host 127.0.0.1 --port 8000
```

При первом запуске модели ещё нет, поэтому сервис сам обучает её на `data/training_data_mock.csv` (несколько секунд) и сохраняет в `model/`. При следующих запусках модель только загружается.

После запуска:

- Swagger UI: http://127.0.0.1:8000/docs
- Проверка: http://127.0.0.1:8000/health

## Переменные окружения

Читаются из окружения или из файла `.env` в корне проекта. Шаблон лежит в `.env.example`. Настоящий `.env` в репозиторий не коммитится.

| Переменная | По умолчанию | Описание |
|------------|--------------|----------|
| `APP_VERSION` | `1.0.0` | Версия, которую отдаёт `/version` |
| `LOG_LEVEL` | `INFO` | Уровень логов: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `THRESHOLD` | `0.5` | Порог вероятности, начиная с которого текст считается экстремистским (0–1) |
| `MODEL_DIR` | `model` | Папка с сохранённой моделью |
| `TRAINING_DATA_PATH` | `data/training_data_mock.csv` | Датасет, на котором модель обучается, если в `MODEL_DIR` её нет |
| `HOST` | `127.0.0.1` | Адрес для `python app_final.py` (в Docker `0.0.0.0`) |
| `PORT` | `8000` | Порт для `python app_final.py` |

Относительные пути считаются от корня проекта. Если значение некорректное (например, `THRESHOLD=2`), приложение не стартует и пишет в лог, какая переменная неверна.

## Модель

Небольшая нейросеть на Keras:

```python
Sequential([
    Input(shape=(50,)),
    Embedding(1000, 16),
    GlobalMaxPooling1D(),
    Dense(16, activation="relu"),
    Dense(1, activation="sigmoid"),
])
```

**Предобработка:** удаление ссылок и упоминаний (`@user`), токенизация Keras `Tokenizer` (словарь из 1000 самых частых слов), обрезка и дополнение до 50 токенов.

**Данные:**

- `data/training_data_mock.csv` содержит 4 примера и используется по умолчанию. Его хватает, чтобы сервис запустился и отвечал, но предсказания на нём условные.
- `data/training_model.csv` — полный датасет, 14 412 размеченных русскоязычных комментариев ([Russian Language Toxic Comments, Kaggle](https://www.kaggle.com/datasets/blackmoon/russian-language-toxic-comments)).

Модель обучается один раз (при первом старте или командой ниже) и дальше только загружается. Во время запросов она не переобучается.

Обучить модель на полном датасете:

```bash
python -m core.train_model --data data/training_model.csv --epochs 10
```

Параметры CLI: `--data`, `--model-dir`, `--max-len`, `--epochs` (`python -m core.train_model --help`). Метаданные обученной модели сохраняются в `model/meta.json` и отдаются через `GET /model`.

В Docker полный датасет подключается аргументом сборки:

```bash
docker compose build --build-arg TRAINING_DATA=data/training_model.csv
```

## Тесты

```bash
pip install -r requirements-dev.txt
python -m pytest -v
```

Тесты проверяют `/health`, `/version`, все маршруты анализа, параметр `threshold`, ошибки валидации, чтение настроек из окружения, сохранение и загрузку модели, обучение при старте и то, что длинный текст не вызывает переобучения. Основная часть тестов использует заглушку вместо нейросети, поэтому они проходят за несколько секунд.

Покрытие:

```bash
python -m pytest --cov=core --cov=app_final --cov-report=term-missing
```

## CI

Workflow `.github/workflows/ci.yml` запускается на каждый push и pull request:

1. **test**: установка зависимостей, `python -m pytest -v`, обучение модели по умолчанию и загрузка artifact `extremism-detector` (код приложения, обученная модель, `requirements.txt`, `.env.example`, README).
2. **docker**: после успешных тестов собирает Docker-образ, запускает контейнер и проверяет `/health`, `/version` и `/analyze`.

Artifact скачивается на странице запуска во вкладке **Actions → нужный запуск → Artifacts**.

## Диагностика

### Проверка, что сервис жив

```bash
curl http://127.0.0.1:8000/health
# {"status":"ok","model_loaded":true}

curl http://127.0.0.1:8000/version
# {"name":"Extremism Detector","version":"1.0.0"}
```

`/health` отвечает `503` и `"model_loaded": false`, если модель не загружена.

### Где смотреть логи

- Локально логи пишутся в консоль, где запущен `python app_final.py` или `uvicorn`.
- В Docker: `docker compose logs -f api`. Статус healthcheck контейнера: `docker compose ps`.

Подробнее: `LOG_LEVEL=DEBUG`.

Пример нормального старта и запросов:

```
INFO [extremism_detector] Starting Extremism Detector v1.0.0
WARNING [core.detector] No saved model in .../model, training a new one
INFO [core.detector] Training model on .../data/training_data_mock.csv (4 samples, 20 epochs)
INFO [core.detector] Model saved to .../model
INFO [extremism_detector] Service is ready, threshold=0.50
INFO [extremism_detector] Analyzed text of 22 chars: probability=0.7258 extremist=True
```

Сами тексты в лог не пишутся, только их длина и результат.

### Частые проблемы

| Симптом | Причина и что делать |
|---------|----------------------|
| `address already in use` при запуске | Порт занят. Запустите на другом: `PORT=8001 python app_final.py` или `uvicorn ... --port 8001` |
| Приложение не стартует, в логе `ValidationError` | Некорректная переменная окружения. В сообщении указано её имя, исправьте `.env` |
| В логе `Failed to prepare the model` | Модель не загрузилась или не обучилась. Проверьте путь `TRAINING_DATA_PATH` и колонки `text,label` в CSV, удалите папку `model/` и запустите заново |
| Нужно переобучить модель | Удалите `model/` и перезапустите сервис или выполните `python -m core.train_model` |
| `422 Unprocessable Entity` | Неверный запрос: пустой `text`, больше 100 текстов в пачке или `threshold` вне 0–1. Подробности в поле `detail` ответа |
| `503` на `/analyze` | Модель ещё не загружена, смотрите логи старта |

## Структура проекта

```
extremism_detector/
├── app_final.py              # FastAPI-приложение и маршруты
├── test_app.py               # Тесты (pytest)
├── core/
│   ├── config.py             # Настройки из переменных окружения, логирование
│   ├── detector.py           # Обучение, сохранение, загрузка и инференс модели
│   ├── schemas.py            # Pydantic-схемы запросов и ответов
│   ├── text.py               # Очистка текста и разбиение на предложения
│   └── train_model.py        # CLI для обучения модели
├── data/
│   ├── training_data_mock.csv
│   └── training_model.csv
├── model/                    # Обученная модель (создаётся автоматически, не в git)
├── requirements.txt
├── requirements-dev.txt
├── .env.example
├── Dockerfile
├── docker-compose.yml
└── .github/workflows/ci.yml
```
