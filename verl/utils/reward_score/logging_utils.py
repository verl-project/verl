"""Shared logging helpers for verifiers."""

from __future__ import annotations

import logging
from typing import Any


def get_reward_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def _format_value(value: Any) -> str:
    if value is None:
        return "None"
    text = str(value).replace("\n", "\\n")
    if not text:
        return '""'
    if any(char.isspace() for char in text) or '"' in text:
        return f"{text!r}"
    return text


def log_reward_warning(
    logger: logging.Logger,
    verifier: str,
    message: str,
    *,
    data_source: str | None = None,
    exc: BaseException | None = None,
) -> None:
    _log_reward_event(
        logger,
        logging.WARNING,
        "warning",
        verifier,
        message,
        data_source=data_source,
        exc=exc,
    )


def log_reward_info(
    logger: logging.Logger,
    verifier: str,
    message: str,
    *,
    data_source: str | None = None,
    exc: BaseException | None = None,
) -> None:
    _log_reward_event(
        logger,
        logging.INFO,
        "info",
        verifier,
        message,
        data_source=data_source,
        exc=exc,
    )


def log_reward_error(
    logger: logging.Logger,
    verifier: str,
    message: str,
    *,
    data_source: str | None = None,
    exc: BaseException | None = None,
) -> None:
    _log_reward_event(
        logger,
        logging.ERROR,
        "error",
        verifier,
        message,
        data_source=data_source,
        exc=exc,
    )


def _log_reward_event(
    logger: logging.Logger,
    level: int,
    event: str,
    verifier: str,
    message: str,
    *,
    data_source: str | None = None,
    exc: BaseException | None = None,
) -> None:
    parts = [f"[verifier={verifier}]"]
    if data_source is not None:
        parts.append(f"[data_source={data_source}]")
    if exc is not None:
        parts.append(f"error={type(exc).__name__}")
        parts.append(f"message={_format_value(exc)}")
        parts.append(f"detail={_format_value(message)}")
    else:
        parts.append(f"message={_format_value(message)}")
    logger.log(level, " ".join(parts))
