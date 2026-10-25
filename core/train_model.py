import argparse
import logging
from pathlib import Path

from core.config import Settings, configure_logging
from core.detector import DEFAULT_EPOCHS, DEFAULT_MAX_LEN, train

logger = logging.getLogger(__name__)


def parse_args(settings: Settings, argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the extremism detection model")
    parser.add_argument("--data", type=Path, default=settings.training_data_path, help="CSV file with text and label columns")
    parser.add_argument("--model-dir", type=Path, default=settings.model_dir, help="where to save the trained model")
    parser.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN, help="number of tokens per text")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    settings = Settings()
    configure_logging(settings.log_level)
    args = parse_args(settings, argv)
    detector = train(args.data, args.model_dir, max_len=args.max_len, epochs=args.epochs)
    logger.info("Done: %d samples, train accuracy %.3f", detector.info.samples, detector.info.train_accuracy)


if __name__ == "__main__":
    main()
