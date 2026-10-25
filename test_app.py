from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app_final import create_app
from core.config import BASE_DIR, Settings
from core.detector import MODEL_FILE, ModelInfo, is_saved, load, read_dataset, train
from core.text import clean_text, split_sentences
from core.train_model import main as train_cli

MOCK_DATA = BASE_DIR / "data" / "training_data_mock.csv"
EXTREMIST_TEXT = "Свергнем власть силой!"
NEUTRAL_TEXT = "Сегодня хорошая погода."


class KeywordDetector:
    info = ModelInfo(data_path="data/test.csv", samples=4, vocabulary_size=15, max_len=50, train_accuracy=1.0)

    def predict(self, texts: list[str]) -> list[float]:
        return [0.9 if "свергнем" in text.lower() else 0.1 for text in texts]


def make_settings(tmp_path: Path, **overrides: object) -> Settings:
    values = {
        "app_version": "1.2.3",
        "threshold": 0.5,
        "model_dir": tmp_path / "model",
        "training_data_path": MOCK_DATA,
        **overrides,
    }
    return Settings(_env_file=None, **values)


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    app = create_app(make_settings(tmp_path))
    app.state.detector = KeywordDetector()
    return TestClient(app)


@pytest.fixture
def client_without_model(tmp_path: Path) -> TestClient:
    return TestClient(create_app(make_settings(tmp_path)))


def test_health_reports_loaded_model(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_health_returns_503_without_model(client_without_model: TestClient) -> None:
    response = client_without_model.get("/health")

    assert response.status_code == 503
    assert response.json() == {"status": "unavailable", "model_loaded": False}


def test_analyze_returns_503_without_model(client_without_model: TestClient) -> None:
    response = client_without_model.post("/analyze", json={"text": EXTREMIST_TEXT})

    assert response.status_code == 503


def test_version_returns_configured_version(client: TestClient) -> None:
    response = client.get("/version")

    assert response.status_code == 200
    assert response.json() == {"name": "Extremism Detector", "version": "1.2.3"}


def test_root_points_to_docs(client: TestClient) -> None:
    assert client.get("/").json()["docs"] == "/docs"


def test_analyze_flags_extremist_text(client: TestClient) -> None:
    response = client.post("/analyze", json={"text": EXTREMIST_TEXT})

    assert response.status_code == 200
    assert response.json() == {"probability": 0.9, "is_extremist": True, "threshold": 0.5}


def test_analyze_marks_neutral_text(client: TestClient) -> None:
    response = client.post("/analyze", json={"text": NEUTRAL_TEXT})

    assert response.status_code == 200
    assert response.json()["is_extremist"] is False


def test_analyze_threshold_parameter_overrides_default(client: TestClient) -> None:
    response = client.post("/analyze", params={"threshold": 0.95}, json={"text": EXTREMIST_TEXT})

    assert response.status_code == 200
    assert response.json() == {"probability": 0.9, "is_extremist": False, "threshold": 0.95}


@pytest.mark.parametrize("payload", [{}, {"text": ""}, {"text": "   "}, {"text": "а" * 5001}])
def test_analyze_rejects_invalid_text(client: TestClient, payload: dict) -> None:
    assert client.post("/analyze", json=payload).status_code == 422


@pytest.mark.parametrize("threshold", ["-0.1", "1.5", "abc"])
def test_analyze_rejects_invalid_threshold(client: TestClient, threshold: str) -> None:
    response = client.post("/analyze", params={"threshold": threshold}, json={"text": NEUTRAL_TEXT})

    assert response.status_code == 422


def test_batch_counts_extremist_texts(client: TestClient) -> None:
    response = client.post("/analyze/batch", json={"texts": [EXTREMIST_TEXT, NEUTRAL_TEXT, "Погуляем по парку"]})

    body = response.json()
    assert response.status_code == 200
    assert body["total"] == 3
    assert body["extremist_count"] == 1
    assert [result["is_extremist"] for result in body["results"]] == [True, False, False]


@pytest.mark.parametrize("texts", [[], ["текст"] * 101])
def test_batch_rejects_empty_or_too_large_list(client: TestClient, texts: list[str]) -> None:
    assert client.post("/analyze/batch", json={"texts": texts}).status_code == 422


def test_sentences_scores_each_sentence(client: TestClient) -> None:
    text = f"{NEUTRAL_TEXT} {EXTREMIST_TEXT} Погуляем по парку."

    response = client.post("/analyze/sentences", json={"text": text})

    body = response.json()
    assert response.status_code == 200
    assert [item["sentence"] for item in body["sentences"]] == [NEUTRAL_TEXT, EXTREMIST_TEXT, "Погуляем по парку."]
    assert [item["is_extremist"] for item in body["sentences"]] == [False, True, False]
    assert body["extremist_count"] == 1


def test_model_info_describes_loaded_model(client: TestClient) -> None:
    response = client.get("/model")

    assert response.status_code == 200
    assert response.json()["samples"] == 4
    assert response.json()["threshold"] == 0.5


def test_clean_text_removes_links_and_mentions() -> None:
    assert clean_text("Смотри https://example.com и пиши @user  срочно") == "Смотри и пиши срочно"


def test_split_sentences_keeps_punctuation() -> None:
    assert split_sentences("Привет! Как дела? Всё хорошо.") == ["Привет!", "Как дела?", "Всё хорошо."]


def test_settings_are_read_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("THRESHOLD", "0.8")
    monkeypatch.setenv("APP_VERSION", "2.0.0")
    monkeypatch.setenv("LOG_LEVEL", "debug")
    monkeypatch.setenv("MODEL_DIR", "custom_model")

    settings = Settings(_env_file=None)

    assert settings.threshold == 0.8
    assert settings.app_version == "2.0.0"
    assert settings.log_level == "DEBUG"
    assert settings.model_dir == BASE_DIR / "custom_model"


def test_settings_reject_invalid_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("THRESHOLD", "1.5")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_read_dataset_requires_text_and_label_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("comment,toxic\nпривет,0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing columns"):
        read_dataset(csv_path)


def test_trained_model_is_saved_and_loaded_back(tmp_path: Path) -> None:
    texts = [EXTREMIST_TEXT, NEUTRAL_TEXT]

    detector = train(MOCK_DATA, tmp_path, epochs=2)
    restored = load(tmp_path)

    assert is_saved(tmp_path)
    assert restored.info == detector.info
    assert restored.predict(texts) == pytest.approx(detector.predict(texts), abs=1e-5)


def test_service_trains_model_on_startup_when_none_is_saved(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)

    with TestClient(create_app(settings)) as client:
        health = client.get("/health").json()
        response = client.post("/analyze", json={"text": EXTREMIST_TEXT})

    assert health["model_loaded"] is True
    assert response.status_code == 200
    assert 0.0 <= response.json()["probability"] <= 1.0
    assert is_saved(settings.model_dir)


def test_service_loads_saved_model_and_never_retrains(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    settings = make_settings(tmp_path)
    train(MOCK_DATA, settings.model_dir, epochs=1)
    saved_at = (settings.model_dir / MODEL_FILE).stat().st_mtime_ns
    long_text = " ".join(["слово"] * 500)

    with caplog.at_level("INFO"), TestClient(create_app(settings)) as client:
        response = client.post("/analyze", json={"text": long_text})

    assert response.status_code == 200
    assert "Loading model" in caplog.text
    assert "Training model" not in caplog.text
    assert (settings.model_dir / MODEL_FILE).stat().st_mtime_ns == saved_at


def test_train_cli_saves_model(tmp_path: Path) -> None:
    train_cli(["--data", str(MOCK_DATA), "--model-dir", str(tmp_path), "--epochs", "1"])

    assert is_saved(tmp_path)
