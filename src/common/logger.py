from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional


class JsonFormatter(logging.Formatter):
    """Simple JSON formatter for stage logs and Colab inspection."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    json_logs: bool = False,
) -> None:
    """Configure root logger once, with optional JSON output."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    handlers: list[logging.Handler] = [logging.StreamHandler()]

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    formatter: logging.Formatter
    if json_logs:
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            "%Y-%m-%d %H:%M:%S",
        )

    for handler in handlers:
        handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(numeric_level)
    for handler in handlers:
        root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Get module logger with a stable namespace."""
    return logging.getLogger(f"mangaanimator.{name}")


def colab_default_log_dir() -> Path:
    """Return default path for logs in Colab or local runs."""
    if os.path.exists("/content"):
        return Path("/content/manga_animator/logs")
    return Path("outputs/logs")
