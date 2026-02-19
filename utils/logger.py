"""
Logging Utilities

Configures logging for the MangaVID pipeline.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    format_string: Optional[str] = None,
) -> None:
    """
    Set up logging configuration for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        format_string: Optional custom format string
    """

    if format_string is None:
        format_string = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    # Get numeric logging level
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Create handlers list
    handlers = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(logging.Formatter(format_string))
    handlers.append(console_handler)

    # File handler
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(logging.Formatter(format_string))
        handlers.append(file_handler)

    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        format=format_string,
        handlers=handlers,
        force=True,
    )

    # Silence noisy loggers
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("cv2").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class PipelineLogger:
    """
    Specialized logger for pipeline operations.
    
    Provides structured logging with timing and progress tracking.
    """

    def __init__(self, name: str):
        self._logger = logging.getLogger(name)
        self._start_times = {}

    def start_stage(self, stage_name: str) -> None:
        """Log the start of a pipeline stage."""
        self._start_times[stage_name] = datetime.now()
        self._logger.info(f"[START] {stage_name}")

    def end_stage(self, stage_name: str, success: bool = True) -> None:
        """Log the end of a pipeline stage with duration."""
        start = self._start_times.get(stage_name)

        if start:
            duration = (datetime.now() - start).total_seconds()
            status = "SUCCESS" if success else "FAILED"
            self._logger.info(f"[{status}] {stage_name} ({duration:.2f}s)")
            del self._start_times[stage_name]
        else:
            status = "SUCCESS" if success else "FAILED"
            self._logger.info(f"[{status}] {stage_name}")

    def progress(self, stage_name: str, current: int, total: int) -> None:
        """Log progress within a stage."""
        percent = (current / total) * 100 if total > 0 else 0
        self._logger.debug(f"[PROGRESS] {stage_name}: {current}/{total} ({percent:.1f}%)")

    def info(self, message: str) -> None:
        """Log info message."""
        self._logger.info(message)

    def warning(self, message: str) -> None:
        """Log warning message."""
        self._logger.warning(message)

    def error(self, message: str, exc_info: bool = False) -> None:
        """Log error message."""
        self._logger.error(message, exc_info=exc_info)

    def debug(self, message: str) -> None:
        """Log debug message."""
        self._logger.debug(message)
