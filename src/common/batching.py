from __future__ import annotations

from collections.abc import Iterable
from typing import TypeVar

T = TypeVar("T")


def chunked(items: Iterable[T], chunk_size: int) -> list[list[T]]:
    """Materialize iterable into batches for predictable Colab memory usage."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    batches: list[list[T]] = []
    current: list[T] = []
    for item in items:
        current.append(item)
        if len(current) >= chunk_size:
            batches.append(current)
            current = []
    if current:
        batches.append(current)
    return batches
