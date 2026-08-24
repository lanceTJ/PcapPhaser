from __future__ import annotations

import atexit
import json
import logging
import os
import queue
import re
import sys
from datetime import datetime
from logging.handlers import QueueHandler, QueueListener, TimedRotatingFileHandler
from pathlib import Path
from typing import Optional, Union


ENCRYPTED_DIR_RE = re.compile(r"^encrypted", re.I)


def is_encrypted_dir(path: Path) -> bool:
    return ENCRYPTED_DIR_RE.match(path.name or "") is not None


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def atomic_write_json(path: Path, obj: dict) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, ensure_ascii=False)
    tmp_path.replace(path)


def now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


__all__ = ["setup", "get_logger", "log", "ensure_dir", "atomic_write_json", "is_encrypted_dir", "now_iso"]


_LOGGER_NAME = "pcapperturbator"
log = logging.getLogger(_LOGGER_NAME)
log.addHandler(logging.NullHandler())
log.setLevel(logging.INFO)
log.propagate = False

_queue: Optional[queue.Queue] = None
_listener: Optional[QueueListener] = None
_configured = False


def setup(
    log_dir: Optional[Union[str, os.PathLike]] = "logs",
    level: Union[int, str] = "INFO",
    console: bool = True,
    filename: str = "pcapperturbator.log",
    rotate_when: str = "midnight",
    rotate_backup: int = 7,
    encoding: str = "utf-8",
) -> logging.Logger:
    """Configure asynchronous logging for the package."""
    global _queue, _listener, _configured

    if _configured:
        return log

    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    log.setLevel(level)

    formatter = logging.Formatter(
        fmt="[%(asctime)s] %(levelname).1s %(process)d %(threadName)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handlers: list[logging.Handler] = []
    if console:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)
        handlers.append(console_handler)

    if log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = TimedRotatingFileHandler(
            filename=str(log_path / filename),
            when=rotate_when,
            backupCount=rotate_backup,
            encoding=encoding,
            utc=False,
            delay=True,
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        handlers.append(file_handler)

    _queue = queue.SimpleQueue()
    queue_handler = QueueHandler(_queue)
    queue_handler.setLevel(level)

    _clear_handlers(log)
    log.addHandler(queue_handler)

    _listener = QueueListener(_queue, *handlers, respect_handler_level=True)
    _listener.start()
    atexit.register(_shutdown_listener)

    _configured = True
    return log



def _clear_handlers(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        try:
            logger.removeHandler(handler)
            handler.close()
        except Exception:
            pass



def _shutdown_listener() -> None:
    global _listener
    if _listener is not None:
        try:
            _listener.stop()
        except Exception:
            pass
        _listener = None



def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a package logger or a child logger."""
    if not name:
        return log
    return logging.getLogger(f"{_LOGGER_NAME}.{name}")


setup(log_dir="logs", level="INFO", console=True)
